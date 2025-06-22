from heapq import merge
from numpy import dtype
from sqlalchemy import collate
import torch
import torchtuples as tt
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Generalized transformer for competing risk
class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')

# Neural Network declaration
class CauseSpecificNet(torch.nn.Module):
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared, in_features,
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                in_features, num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input) + input
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out



class MultiEncoderCauseSpecificNet(torch.nn.Module):
    def __init__(self, in_features_dict, encoder_nodes, fused_nodes, num_nodes_indiv,
                 num_risks, out_features, batch_norm=True, dropout=None):
        """
        Args:
            in_features_dict (dict): Mapping modality name to its number of input features.
            encoder_nodes (list or int): Architecture for each encoder.
                The final output dimension is forced to match the raw input dimension.
            fused_nodes (list or int): Architecture for the fused encoder network.
            num_nodes_indiv (list or int): Architecture for each risk-specific decoder.
            num_risks (int): Number of competing risks.
            out_features (int): Final output dimension for each risk-specific decoder.
            batch_norm (bool, optional): Whether to use batch normalization.
            dropout (float, optional): Dropout probability.
        """
        super().__init__()
        # Build one encoder per modality.
        # We force the output dimension of each encoder to equal the modality's input dimension.
        #self.encoders = torch.nn.ModuleDict({
        #    key: tt.practical.MLPVanilla(
        #            in_features,                      # input dimension
        #            encoder_nodes,                    # hidden layers architecture
        #            in_features,                      # force output dimension == input dimension
        #            batch_norm, dropout
        #        )
        #    for key, in_features in in_features_dict.items()
        #})
        fusion_out_dim = 64
        self.model_encoders = torch.nn.ModuleDict()
        self.model_encoders_normalizers = torch.nn.ModuleDict()
        for enc_name, enc_layers in encoder_nodes.items():
            self.model_encoders[enc_name] = tt.practical.MLPVanilla(
                int(enc_layers[0]), enc_layers[1:],int(enc_layers[0]), batch_norm, dropout)
            self.model_encoders_normalizers[enc_name] = torch.nn.BatchNorm1d(enc_layers[0])

        # The concatenated residual-enhanced encoder outputs will have dimension:
        #total_features = sum(in_features for in_features in in_features_dict.values())
        total_features = sum(
            int(enc_layers[0]) for enc_layers in encoder_nodes.values()
        )
        # Fused encoder: processes the concatenated outputs.
        # Its output dimension is defined by fused_nodes architecture.
        # (No additional residual connection at this stage since each modality already added its skip connection.)
        self.fused_net = tt.practical.MLPVanilla(
            total_features, fused_nodes,
            fused_nodes[-1] if isinstance(fused_nodes, list) else fused_nodes,
            batch_norm, dropout
        )

        # Risk-specific decoders: each one takes the fused network output.
        fused_out_features = fused_nodes[-1] if isinstance(fused_nodes, list) else fused_nodes
        self.risk_nets = torch.nn.ModuleList([
            tt.practical.MLPVanilla(
                fused_out_features, num_nodes_indiv, out_features,
                batch_norm, dropout
            )
            for _ in range(num_risks)
        ])

    def forward(self, x_dict):
        """
        Args:
            x_dict (dict): A dictionary mapping modality keys to input tensors.
                Each tensor is assumed to have shape (batch_size, features) where features
                match the corresponding in_features_dict entry.
        Returns:
            Tensor: Shape (batch_size, num_risks, out_features).
        """
        # If the input is a tuple (e.g. (x, target)), extract the dict x.
        #if isinstance(input, (tuple, list)):
        #    x_dict = input[0]
        #else:
        #    x_dict = input
            
        encoded_outputs = []
        for key, encoder in self.model_encoders.items():
            # For each modality, get the raw input.
            raw_input = x_dict[key]
            # Compute the encoder output.
            encoded = encoder(raw_input)
            encoded = self.model_encoders_normalizers[key](encoded)
            # Add the modality-specific skip connection: raw input added to encoder output.
            # This requires that encoder output has the same dimension as raw_input.
            enhanced = encoded + raw_input
            encoded_outputs.append(enhanced)
        # Concatenate all enhanced encoder outputs along the feature dimension.
        fused_input = torch.cat(encoded_outputs, dim=1)
        # Process the concatenated representation through the fused network.
        fused_out = self.fused_net(fused_input)
        # Compute risk-specific outputs.
        outputs = [net(fused_out) for net in self.risk_nets]
        # Stack outputs to have shape (batch_size, num_risks, out_features).
        return torch.stack(outputs, dim=1)


class DictToDevice:
    def __init__(self, x_dict):
        self.x_dict = x_dict
    def to(self, device):
        # Move each tensor to the device
        for k in self.x_dict:
            self.x_dict[k] = self.x_dict[k].to(device)
        return self
    def __getitem__(self, key):
        return self.x_dict[key]
    def keys(self):
        return self.x_dict.keys()

class DictDataset(torch.utils.data.Dataset):
    """
    Expects:
      x_dict: { 'modality1': np.array, 'modality2': np.array, ... }
              All arrays must have the same length (the number of samples).
      t: duration array
      e: event array
    """
    def __init__(self, x_dict, t, e, transform=None):
        self.x_dict = x_dict
        self.keys = list(x_dict.keys())
        self.t = t
        self.e = e
        self.transform = transform
        # All arrays must share length N:
        self.size = len(t)

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Grab each modality's row idx
        x_out = {}
        for k in self.keys:
            x_out[k] = torch.as_tensor(self.x_dict[k][idx], dtype=torch.float32)
        
        x_out = DictToDevice(x_out)
        t_ = torch.as_tensor(self.t[idx], dtype=torch.int64)
        e_ = torch.as_tensor(self.e[idx], dtype=torch.int64)
        #target = (t_, e_)
        #target = target.to_numpy()
        #from pycox.models.data import pair_rank_mat
        #rank_mat = pair_rank_mat(*target)
        #target = tt.tuplefy(*target, rank_mat).to_tensor()
        
        # Also fetch (time, event)
        return x_out, (t_, e_)

    #def __getitem__(self, idx):
    #    x1 = torch.as_tensor(self.x_dict["embeddings"].iloc[idx], dtype=torch.float32)
    #    x2 = torch.as_tensor(self.x_dict["radiomics"].iloc[idx], dtype=torch.float32)
    #    x3 = torch.as_tensor(self.x_dict["tabular"].iloc[idx], dtype=torch.float32)

    #    t_ = torch.as_tensor(self.t.iloc[idx], dtype=torch.float32)
    #    e_ = torch.as_tensor(self.e.iloc[idx], dtype=torch.int64)

        # Return a nested structure of Tensors only, e.g. ( (x1,x2,x3), (t_, e_) ).
    #    return (x1, x2, x3), (t_, e_)

    #def __getitem__(self, idx):
        # Grab each modality's row idx
    #    x_out = {}
    #    for k in self.keys:
    #        x_out[k] = torch.as_tensor(self.x_dict[k].iloc[idx], dtype=torch.float32)
        
    #    x_out = DictToDevice(x_out)
    #    # Also fetch (time, event)
    #    t_ = torch.as_tensor(self.t[idx], dtype=torch.float32)
    #    e_ = torch.as_tensor(self.e[idx], dtype=torch.int64)
    #    return x_out, (t_, e_)

def dict_to_device_collate(batch):
    """
    'batch' is a list of N items from Dataset.__getitem__.
    Each item is (DictToDevice(x_out), (t_, e_)).

    We want to combine them into a single batch:
       (
         DictToDevice(batched_x_out),  # stacked along dim=0
         (batched_t, batched_e)
       )
    """

    # Separate out the (x_out, (t,e)) pairs
    x_list = []
    t_list = []
    e_list = []
    for x_out, (t_, e_) in batch:
        x_list.append(x_out)  # DictToDevice
        t_list.append(t_)
        e_list.append(e_)

    # We assume x_list[0].x_dict has keys = all modalities
    keys = x_list[0].x_dict.keys()

    # Create a new dictionary to hold stacked Tensors
    merged_x_dict = {}
    for key in keys:
        # get a list of Tensors for this key
        stack_list = [xd.x_dict[key] for xd in x_list]
        merged_x_dict[key] = torch.stack(stack_list, dim=0)

    # Wrap in a new DictToDevice
    batched_x = DictToDevice(merged_x_dict)

    # Stack times/events
    t_batch = torch.stack(t_list, dim=0)
    e_batch = torch.stack(e_list, dim=0)

    return batched_x, (t_batch, e_batch)



def deephit_collate(batch):
    """
    batch: list of length B, each element: (dict_of_tensors, (t, e))
    
    We want to produce:
      (batched_dict_of_tensors, (t_batch, e_batch, rank_mat))
    where rank_mat is shape [B, B].
    """
    from pycox.models.data import pair_rank_mat
    x_list = []
    t_list = []
    e_list = []
    
    for x_out, (t_, e_) in batch:
        x_list.append(x_out)  # dictionary of Tensors
        t_list.append(t_)
        e_list.append(e_)

    # 1) Merge x_list into one dictionary of shape [B, features]
    merged_x_dict = {}
    for key in x_list[0].keys():
        stack_list = [x_dict[key] for x_dict in x_list]
        merged_x_dict[key] = torch.stack(stack_list, dim=0)
    
    merged_x = DictToDevice(merged_x_dict)

    # 2) Stack times and events
    t_batch = torch.stack(t_list, dim=0)
    e_batch = torch.stack(e_list, dim=0)
    
    # 3) Build NxN rank_mat
    # pair_rank_mat needs numpy arrays
    t_np = t_batch.numpy()
    e_np = e_batch.numpy()
    target = (t_np, e_np)
    rank_mat = pair_rank_mat(*target)
    # Convert to a torch tensor
    #rank_mat = torch.as_tensor(rank_mat, dtype=torch.float32)
    
    target = tt.tuplefy(*target, rank_mat).to_tensor()

    # 4) Return final structure
    # This is what PyCox expects if alpha>0: (inputs, (times, events, rank_mat))
    return tt.tuplefy(merged_x, target)

def deephit_collate_predict(batch):
    """
    batch: list of length B, each element: (dict_of_tensors, (t, e))
    
    We want to produce:
      (batched_dict_of_tensors, (t_batch, e_batch, rank_mat))
    where rank_mat is shape [B, B].
    """
    from pycox.models.data import pair_rank_mat
    x_list = []
    t_list = []
    e_list = []
    
    for x_out, (t_, e_) in batch:
        x_list.append(x_out)  # dictionary of Tensors
        t_list.append(t_)
        e_list.append(e_)

    # 1) Merge x_list into one dictionary of shape [B, features]
    merged_x_dict = {}
    for key in x_list[0].keys():
        stack_list = [x_dict[key] for x_dict in x_list]
        merged_x_dict[key] = torch.stack(stack_list, dim=0)
    
    merged_x = DictToDevice(merged_x_dict)

    # 2) Stack times and events
    t_batch = torch.stack(t_list, dim=0)
    e_batch = torch.stack(e_list, dim=0)
    
    # 3) Build NxN rank_mat
    # pair_rank_mat needs numpy arrays
    t_np = t_batch.numpy()
    e_np = e_batch.numpy()
    target = (t_np, e_np)
    #rank_mat = pair_rank_mat(*target)
    # Convert to a torch tensor
    #rank_mat = torch.as_tensor(rank_mat, dtype=torch.float32)
    
    target = tt.tuplefy(*target).to_tensor()
    return merged_x

    # 4) Return final structure
    # This is what PyCox expects if alpha>0: (inputs, (times, events, rank_mat))
    #return tt.tuplefy(merged_x, target)

#def make_dict_dataloader(x_dict, t, e, batch_size=256, shuffle=True, num_workers=0):
def make_dict_dataloader(input, batch_size=256, shuffle=True, num_workers=0):
    x_dict, target = input
    t, e = target
    ds = DictDataset(x_dict, t, e)
    return torch.utils.data.DataLoader(ds, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle, 
                                       num_workers=num_workers,
                                       collate_fn=deephit_collate)
    
def make_dict_dataloader_predict(input, batch_size=256, shuffle=True, num_workers=0):
    x_dict, target = input
    t, e = target
    ds = DictDataset(x_dict, t, e)
    return torch.utils.data.DataLoader(ds, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle, 
                                       num_workers=num_workers,
                                       collate_fn=deephit_collate_predict)


     
     

def get_continuous_columns_tabular():
    continuous_columns = ['Time spent watching television (TV)',
       'Sleep duration', 'Salad raw vegetable intake', 'Fresh fruit intake',
       'BMI', 'Age', 'Summed MET minutes per week for all activity']
    continuous_columns_bp = ["diastolic_bp_mean", "systolic_bp_mean", "pulse_mean"]
    continuous_columns_spirio = ["fvc_best", "fev1_best", "pef_best"]
    continuous_columns = continuous_columns + continuous_columns_bp + continuous_columns_spirio
    return continuous_columns

def get_continuous_columns_embeddings():
    return [r"feature_{}".format(i) for i in range(0, 1026)]

def get_continuous_columns_wb_embeddings():
    return [r"feature_wb_{}".format(i) for i in range(0, 1026)]


def get_continuous_columns_radiomics():
    organs = {'bone',
            'digestive',
            'eid',
            'endocrine',
            'fat',
            'heart',
            'kidney',
            'liver',
            'muscle',
            'pancreas',
            'respiratory',
            'spine',
            'spleen',
            'urinary',
            'vascular'
    }
    rad_cols = []
    for organ in organs:
        for i in range(1, 11):
            rad_cols.append(organ + "_" + str(i))
    return rad_cols

def get_cardiac_radiomics():
    rad_cols = ['LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)', 'LVCO (L/min)',
       'LVM (g)', 'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)',
       'LAV max (mL)', 'LAV min (mL)', 'LASV (mL)', 'LAEF (%)', 'RAV max (mL)',
       'RAV min (mL)', 'RASV (mL)', 'RAEF (%)', 'AAo max area (mm²)',
       'AAo min area (mm²)', 'DAo max area (mm²)', 'DAo min area (mm²)',
       'WT_AHA_1 (mm)', 'WT_AHA_2 (mm)', 'WT_AHA_3 (mm)', 'WT_AHA_4 (mm)',
       'WT_AHA_5 (mm)', 'WT_AHA_6 (mm)', 'WT_AHA_7 (mm)', 'WT_AHA_8 (mm)',
       'WT_AHA_9 (mm)', 'WT_AHA_10 (mm)', 'WT_AHA_11 (mm)', 'WT_AHA_12 (mm)',
       'WT_AHA_13 (mm)', 'WT_AHA_14 (mm)', 'WT_AHA_15 (mm)', 'WT_AHA_16 (mm)',
       'WT_Global (mm)', 'LV circumferential strain AHA 1',
       'LV circumferential strain AHA 2', 'LV circumferential strain AHA 3',
       'LV circumferential strain AHA 4', 'LV circumferential strain AHA 5',
       'LV circumferential strain AHA 6', 'LV circumferential strain AHA 7',
       'LV circumferential strain AHA 8', 'LV circumferential strain AHA 9',
       'LV circumferential strain AHA 10', 'LV circumferential strain AHA 11',
       'LV circumferential strain AHA 12', 'LV circumferential strain AHA 13',
       'LV circumferential strain AHA 14', 'LV circumferential strain AHA 15',
       'LV circumferential strain AHA 16', 'LV circumferential strain global',
       'LV radial strain AHA 1', 'LV radial strain AHA 2',
       'LV radial strain AHA 3', 'LV radial strain AHA 4',
       'LV radial strain AHA 5', 'LV radial strain AHA 6',
       'LV radial strain AHA 7', 'LV radial strain AHA 8',
       'LV radial strain AHA 9', 'LV radial strain AHA 10',
       'LV radial strain AHA 11', 'LV radial strain AHA 12',
       'LV radial strain AHA 13', 'LV radial strain AHA 14',
       'LV radial strain AHA 15', 'LV radial strain AHA 16',
       'LV radial strain global', 'LV longitudinal strain Segment 1',
       'LV longitudinal strain Segment 2', 'LV longitudinal strain Segment 3',
       'LV longitudinal strain Segment 4', 'LV longitudinal strain Segment 5',
       'LV longitudinal strain Segment 6', 'LV longitudinal strain global']
    return rad_cols       

from sklearn.preprocessing import StandardScaler
import pandas as pd

class DataScaler:
    def __init__(self, scaler=None):
        """
        Optionally pass your own scaler (e.g., MinMaxScaler()). 
        Defaults to StandardScaler().
        """
        self.scaler = scaler if scaler is not None else StandardScaler()
        
        self.candidate_cols = (
            get_continuous_columns_tabular() 
            + get_continuous_columns_embeddings() 
            + get_continuous_columns_radiomics() 
            + get_cardiac_radiomics()
            + get_continuous_columns_wb_embeddings()
        )
        
        # Store where each actual column is found: {col_name: data_key}
        self.actual_cols_map = {}

    def fit(self, X):
        """
        X can be a DataFrame or a dict of DataFrames.
        """
        self.actual_cols_map = {}

        if isinstance(X, dict):
            for key, df in X.items():
                for col in self.candidate_cols:
                    if col in df.columns:
                        self.actual_cols_map[col] = key
            # Collect all columns across all dfs to fit together
            fit_data = pd.concat([
                X[key][[col]] for col, key in self.actual_cols_map.items()
            ], axis=1)
        else:
            for col in self.candidate_cols:
                if col in X.columns:
                    self.actual_cols_map[col] = None  # all in one df
            fit_data = X[list(self.actual_cols_map.keys())]

        self.scaler.fit(fit_data)
        return self

    def transform(self, X):
        """
        X can be a DataFrame or a dict of DataFrames.
        """
        if isinstance(X, dict):
            # Copy the whole dict and each dataframe inside
            X_scaled = {key: df.copy() for key, df in X.items()}
            data_to_transform = pd.concat([
                X[key][[col]] for col, key in self.actual_cols_map.items()
            ], axis=1)

            scaled_values = self.scaler.transform(data_to_transform)
            scaled_df = pd.DataFrame(scaled_values, columns=self.actual_cols_map.keys(), index=data_to_transform.index)

            for col, key in self.actual_cols_map.items():
                X_scaled[key][col] = scaled_df[col]
                
            # convert to numpy 
            for key in X_scaled:
                X_scaled[key] = X_scaled[key].to_numpy()

            return X_scaled
        else:
            X_scaled = X.copy()
            cols = list(self.actual_cols_map.keys())
            X_scaled[cols] = self.scaler.transform(X[cols])
            return X_scaled.to_numpy()

    def fit_transform(self, X):
        return self.fit(X).transform(X)
