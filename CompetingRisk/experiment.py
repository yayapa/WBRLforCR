from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold, ShuffleSplit, ParameterSampler, train_test_split
import pandas as pd
import numpy as np
import pickle
import torch
import time
import os
import io

from deephit.utils import make_dict_dataloader_predict
from deephit.utils import DataScaler
from DeepSurvivalMachines.dsm.dsm_api import DeepSurvivalMachinesMultiEncoder



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location = 'cpu')
        else: 
            return super().find_class(module, name)

class ToyExperiment():

    def train(self, *args, cause_specific = False):
        print("Toy Experiment - Results already saved")

class Experiment():

    def __init__(self, hyper_grid = None, n_iter = 100, fold = None,
                k = 5, random_seed = 0, path = 'results', save = True, delete_log = False, times = 100):
        """
        Args:
            hyper_grid (Dict, optional): Dictionary of parameters to explore.
            n_iter (int, optional): Number of random grid search to perform. Defaults to 100.
            fold (int, optional): Fold to compute (this allows to parallelise computation). If None, starts from 0.
            k (int, optional): Number of split to use for the cross-validation. Defaults to 5.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 0.
            path (str, optional): Path to save results and log. Defaults to 'results'.
            save (bool, optional): Should we save result and log. Defaults to True.
            delete_log (bool, optional): Should we delete the log after all training. Defaults to False.
            times (int, optional): Number of time points where to evaluates. Defaults to 100.
        """
        self.hyper_grid = list(ParameterSampler(hyper_grid, n_iter = n_iter, random_state = random_seed) if hyper_grid is not None else [{}])
        self.random_seed = random_seed
        self.k = k
        
        # Allows to reload a previous model
        self.all_fold = fold
        self.iter, self.fold = 0, 0 
        self.best_hyper = {}
        self.best_model = {}
        self.best_nll = None

        self.times = times

        self.path = path
        self.tosave = save
        self.delete_log = delete_log
        self.running_time = 0

    @classmethod
    def create(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
                random_seed = 0, path = 'results', force = False, save = True, delete_log = False):
        if not(force):
            path = path if fold is None else path + '_{}'.format(fold)
            if os.path.isfile(path + '.csv'):
                return ToyExperiment()
            elif os.path.isfile(path + '.pickle'):
                print('Loading previous copy')
                try:
                    return cls.load(path+ '.pickle')
                except Exception as e:
                    print('ERROR: Reinitalizing object')
                    os.remove(path + '.pickle')
                    pass    
        return cls(hyper_grid, n_iter, fold, k, random_seed, path, save, delete_log)

    @classmethod
    def load(cls, path):
        file = open(path, 'rb')
        if torch.cuda.is_available():
            return pickle.load(file)
        else:
            se = CPU_Unpickler(file).load()
            for i in se.best_model:
                if not isinstance(se.best_model[i], dict):
                    se.best_model[i].cuda = False
            return se

    @classmethod
    def merge(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
            random_seed = 0, path = 'results', force = False, save = True, delete_log = False):
        if os.path.isfile(path + '.csv'):
            return ToyExperiment()
        merged = cls(hyper_grid, n_iter, fold, k, random_seed, path, save, delete_log)
        for i in range(5):
            path_i = path + '_{}.pickle'.format(i)
            if os.path.isfile(path_i):
                model = cls.load(path_i)
                print(model.iter, model.fold)
                merged.best_model[i] = model.best_model[i]
            else:
                print('Fold {} has not been computed yet'.format(i))
        merged.fold = 5 # Nothing to run
        return merged

    @classmethod
    def save(cls, obj):
        # create path if not exist
        if not os.path.exists(obj.path):
            os.makedirs(obj.path)
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')
    

            
    def save_results(self, x):
        def select_rows(x, index):
            if isinstance(x, dict):
                return {mod: df.loc[index] for mod, df in x.items()}
            else:
                return x.iloc[index]
        print("Shape x", x)
        predictions = []
        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            model = self.best_model[i]
            x_fold = select_rows(x, index)
            x_fold = self.scalers[i].transform(x_fold)
            predictions.append(pd.concat([self._predict_(model, 
                                                         x_fold, 
                                        r, index) for r in self.risks], axis = 1))

        predictions = pd.concat(predictions, axis = 0).loc[self.fold_assignment.dropna().index]

        if self.tosave:
            fold_assignment = self.fold_assignment.copy().to_frame()
            fold_assignment.columns = pd.MultiIndex.from_product([['Use'], ['']])
            pd.concat([predictions, fold_assignment], axis = 1).to_csv(self.path + '.csv')

        if self.delete_log:
            os.remove(self.path + '.pickle')
        return predictions

    def _to_cpu(self, dsm):
        """Detach all CUDA tensors inside a DeepSurvivalMachines instance."""
        if hasattr(dsm, "cpu"):          # DSM >= 0.2.2 implements .cpu()
            dsm.cpu()
        elif hasattr(dsm, "torch_model"):  # older versions
            dsm.torch_model.cpu()
        # train_dsm saves full batches here; drop the references
        for attr in ("_data_cache", "_val_cache"):
            if hasattr(dsm, attr):
                delattr(dsm, attr)
    
    def train(self, x, t, e, cause_specific = False):
        """
            Cross validation model

            Args:
                x (Dataframe n * d): Observed covariates
                t (Dataframe n): Time of censoring or event
                e (Dataframe n): Event indicator

                cause_specific (bool): If model should be trained in cause specific setting

            Returns:
                (Dict, Dict): Dict of fitted model and Dict of observed performances
        """
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        
        #self.scaler = StandardScaler()
        #x = self.scaler.fit_transform(x)
        #x = x.to_numpy() if isinstance(x, pd.DataFrame) else x
        e = e.astype(int)

        self.risks = np.unique(e[e > 0])
        #self.fold_assignment = pd.Series(np.nan, index = range(len(x)))
        self.fold_assignment = pd.Series(np.nan, index = range(len(t))) # more stable since x can be a dict
        groups = None
        if isinstance(self.k, list):
            kf = GroupKFold()
            groups = self.k
        elif self.k == 1:
            kf = ShuffleSplit(n_splits = self.k, random_state = self.random_seed, test_size = 0.2)
        else:
            kf = StratifiedKFold(n_splits = self.k, random_state = self.random_seed, shuffle = True)

        # First initialization
        if self.best_nll is None:
            self.best_nll = np.inf
            
        self.scalers = {}
        #for i, (train_index, test_index) in enumerate(kf.split(x, e, groups = groups)):
        for i, (train_index, test_index) in enumerate(kf.split(t, e, groups = groups)): # more stable since x can be a dict
            self.fold_assignment[test_index] = i
            if i < self.fold: continue # When reload: start last point
            if not(self.all_fold is None) and (self.all_fold != i): continue
            print('Fold {}'.format(i))

            train_index, dev_index = train_test_split(train_index, test_size = 0.2, random_state = self.random_seed, stratify = e[train_index])
            dev_index, val_index   = train_test_split(dev_index,   test_size = 0.5, random_state = self.random_seed, stratify = e[dev_index])
            
            #save_split_dir = "/home/dmitrii/GitHub/NeuralFineGray/data/data_splits/cardiac/"
            # save train_index, dev_index, val_index
            #np.save(os.path.join(save_split_dir, f"train_index_fold_{i}.npy"), train_index)
            #np.save(os.path.join(save_split_dir, f"dev_index_fold_{i}.npy"), dev_index)
            #np.save(os.path.join(save_split_dir, f"val_index_fold_{i}.npy"), val_index)
            
            
            #x_train, x_dev, x_val = x[train_index], x[dev_index], x[val_index]
            
            # Split x based on whether it is a dict or a DataFrame.
            if isinstance(x, dict):
                x_train = {k: v.iloc[train_index] for k, v in x.items()}
                x_dev   = {k: v.iloc[dev_index]   for k, v in x.items()}
                x_val   = {k: v.iloc[val_index]   for k, v in x.items()}
            else:
                x_train, x_dev, x_val = x.iloc[train_index], x.iloc[dev_index], x.iloc[val_index]
                
            self.scalers[i] = DataScaler()
            x_train = self.scalers[i].fit_transform(x_train)
            x_dev = self.scalers[i].transform(x_dev)
            x_val = self.scalers[i].transform(x_val)
            
            t_train, t_dev, t_val = t[train_index], t[dev_index], t[val_index]
            e_train, e_dev, e_val = e[train_index], e[dev_index], e[val_index]

            # Train on subset one domain
            ## Grid search best params
            for j, hyper in enumerate(self.hyper_grid):
                
                if j < self.iter: continue # When reload: start last point
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)

                start_time = time.process_time()
                model = self._fit_(x_train, t_train, e_train, x_val, t_val, e_val, hyper.copy(), cause_specific = cause_specific)
                #torch.cuda.empty_cache() # Free memory
                
                self.running_time += time.process_time() - start_time
                
                    # ----------- NEW: dev loss without graph -----------------------
                #with torch.no_grad():
                nll = self._nll_(model, x_dev, t_dev, e_dev, e_train, t_train)
                # ---------------------------------------------------------------

                # ----------- NEW: keep only the best model on GPU --------------
                if nll < self.best_nll:
                    # off-load previous winner for this fold (if any)
                    if i in self.best_model:
                        self._to_cpu(self.best_model[i])
                    self.best_model[i]  = model          # stays on GPU
                    self.best_hyper[i]  = hyper
                    self.best_nll       = nll
                else:
                    # loser → send to CPU and forget
                    self._to_cpu(model)
                del model                                # drop last ref
                torch.cuda.empty_cache()                 # now memory is released
                    # optional: print a quick memory trace
                if j % 1 == 0:
                    alloc  = torch.cuda.memory_allocated() / 1e6
                    cache  = torch.cuda.memory_reserved()  / 1e6
                    print(f"[fold {i} iter {j:>3}] alloc={alloc:8.1f}  cache={cache:8.1f}")
                
                """
                nll = self._nll_(model, x_dev, t_dev, e_dev, e_train, t_train)
                if nll < self.best_nll:
                    self.best_hyper[i] = hyper
                    self.best_model[i] = model
                    self.best_nll = nll
                #del model
                #torch.cuda.empty_cache() # Free memory did not work
                """
                self.iter = j + 1
                self.save(self)
            self.fold, self.iter = i + 1, 0
            self.best_nll = np.inf
            self.save(self)

        if self.all_fold is None:
            return self.save_results(x)

    def _fit_(self, *params):
        raise NotImplementedError()

    def _nll_(self, *params):
        raise NotImplementedError()

    def likelihood(self, x, t, e):
        x = x.to_numpy() if isinstance(x, pd.DataFrame) else x
        x = self.scaler.transform(x)
        nll_fold = {}

        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            train = self.fold_assignment[self.fold_assignment != i].index
            model = self.best_model[i]
            if type(model) is dict:
                nll_fold[i] = np.mean([self._nll_(model[r], x[index], t[index], e[index] == r, e[train] == r, t[train]) for r in self.risks])
            else:
                nll_fold[i] = self._nll_(model, x[index], t[index], e[index], e[train], t[train])

        return nll_fold

class DSMExperiment(Experiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):  
        print("DSM Experiment")
        from dsm import DeepSurvivalMachines

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)
        
        cuda = torch.cuda.is_available()
        #cuda = False

        model = DeepSurvivalMachines(**hyperparameter, cuda = cuda)
        model.fit(x, t, e, iters = epochs, batch_size = batch,
                learning_rate = lr, val_data = (x_val, t_val, e_val))
        
        return model

    def _nll_(self, model, x, t, e, *train):
        return model.compute_nll(x, t, e)

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.times.tolist(), risk = r), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

class MultiEncoderDSMExperiment(DSMExperiment):
    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):
        from dsm import DeepSurvivalMachinesMultiEncoder

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)
        model = DeepSurvivalMachinesMultiEncoder(**hyperparameter, cuda = torch.cuda.is_available())
        model.fit(x, t, e, iters = epochs, batch_size = batch,
                learning_rate = lr, val_data = (x_val, t_val, e_val))
        
        return model

class DeepHitExperiment(Experiment):
    """
        This class require a slightly more involved saving scheme to avoid a lambda error with pickle
        The models are removed at each save and reloaded before saving results 
    """

    @classmethod
    def load(cls, path):
        from pycox.models import DeepHitSingle, DeepHit
        file = open(path, 'rb')
        if torch.cuda.is_available():
            exp = pickle.load(file)
            for i in exp.best_model:
                if isinstance(exp.best_model[i], tuple):
                    net, cuts = exp.best_model[i]
                    exp.best_model[i] = DeepHit(net, duration_index = cuts) if len(exp.risks) > 1 \
                                    else DeepHitSingle(net, duration_index = cuts)
            return exp
        else:
            se = CPU_Unpickler(file).load()
            for i in se.best_model:
                if isinstance(se.best_model[i], tuple):
                    net, cuts = se.best_model[i]
                    se.best_model[i] = DeepHit(net, duration_index = cuts) if len(se.risks) > 1 \
                                    else DeepHitSingle(net, duration_index = cuts)
                    se.best_model[i].cuda = False
            return se

    @classmethod
    def save(cls, obj):
        from pycox.models import DeepHitSingle, DeepHit
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                for i in obj.best_model:
                    # Split model and save components (error pickle otherwise)
                    if isinstance(obj.best_model[i], DeepHit) or isinstance(obj.best_model[i], DeepHitSingle):
                        obj.best_model[i] = (obj.best_model[i].net, obj.best_model[i].duration_index)
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')

    def save_results(self, x):
        from pycox.models import DeepHitSingle, DeepHit

        # Reload models in memory
        for i in self.best_model:
            if isinstance(self.best_model[i], tuple):
                # Reload model
                net, cuts = self.best_model[i]
                self.best_model[i] = DeepHit(net, duration_index = cuts) if len(self.risks) > 1 \
                                else DeepHitSingle(net, duration_index = cuts)
        return super().save_results(x)

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific): 
        print("DeepHit Experiment")
        from deephit.utils import CauseSpecificNet, tt, LabTransform
        from pycox.models import DeepHitSingle, DeepHit

        n = hyperparameter.pop('n', 15)
        nodes = hyperparameter.pop('nodes', [100])
        shared = hyperparameter.pop('shared', [100])
        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)
        

        self.eval_times = np.linspace(0, t.max(), n)
        callbacks = [tt.callbacks.EarlyStopping()]
        num_risks = len(np.unique(e))- 1
        if  num_risks > 1:
            self.labtrans = LabTransform(self.eval_times.tolist())
            net = CauseSpecificNet(x.shape[1], shared, nodes, num_risks, self.labtrans.out_features, False)
            model = DeepHit(net, tt.optim.Adam, duration_index = self.labtrans.cuts)
        else:
            self.labtrans = DeepHitSingle.label_transform(self.eval_times.tolist())
            net = tt.practical.MLPVanilla(x.shape[1], shared + nodes, self.labtrans.out_features, False)
            model = DeepHitSingle(net, tt.optim.Adam, duration_index = self.labtrans.cuts)
        model.optimizer.set_lr(lr)
        model.fit(x.astype('float32'), self.labtrans.transform(t, e), batch_size = batch, epochs = epochs, 
                    callbacks = callbacks, val_data = (x_val.astype('float32'), self.labtrans.transform(t_val, e_val)))
        return model

    def _nll_(self, model, x, t, e, *train):
        return model.score_in_batches(x.astype('float32'), self.labtrans.transform(t, e))['loss']

    def _predict_(self, model, x, r, index):
        if len(self.risks) == 1:
            survival = model.predict_surv_df(x.astype('float32')).values
        else:
            survival = 1 - model.predict_cif(x.astype('float32'))[r - 1]

        # Interpolate at the point of evaluation
        survival = pd.DataFrame(survival, columns = index, index = model.duration_index)
        predictions = pd.DataFrame(np.nan, columns = index, index = self.times)
        survival = pd.concat([survival, predictions]).sort_index(kind = 'stable').bfill().ffill()
        survival = survival[~survival.index.duplicated(keep='first')]
        return survival.loc[self.times].set_index(pd.MultiIndex.from_product([[r], self.times])).T
    
class MultiEncoderDeepHitExperiment(DeepHitExperiment):
    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):
        from deephit.utils import LabTransform, MultiEncoderCauseSpecificNet, tt, make_dict_dataloader
        from pycox.models import DeepHitSingle, DeepHit

        # Hyperparameters (with defaults if not specified)
        n = hyperparameter.pop('n', 15)
        # New hyperparameters for multi-encoder:
        encoder_nodes = hyperparameter.pop('encoder_nodes', [100])
        fused_nodes   = hyperparameter.pop('fused_nodes', [100])
        # For backward compatibility, keep 'shared' for single encoder:
        shared        = hyperparameter.pop('shared', [100])
        nodes         = hyperparameter.pop('nodes', [100])
        epochs        = hyperparameter.pop('epochs', 1000)
        batch         = hyperparameter.pop('batch', 250)
        lr            = hyperparameter.pop('learning_rate', 0.001)
        
        self.eval_times = np.linspace(0, t.max(), n)
        callbacks = [tt.callbacks.EarlyStopping()]
        num_risks = len(np.unique(e)) - 1

        # Choose the label transformer based on the number of risks
        if num_risks > 1:
            self.labtrans = LabTransform(self.eval_times.tolist())
        else:
            self.labtrans = DeepHitSingle.label_transform(self.eval_times.tolist())
        
        # Check if the input x is a dict (new multi-encoder setting) or not
        if isinstance(x, dict):
            # Build a dict with the number of features for each modality
            in_features_dict = {key: v.shape[1] for key, v in x.items()}
            net = MultiEncoderCauseSpecificNet(
                in_features_dict, encoder_nodes, fused_nodes, nodes,
                num_risks, self.labtrans.out_features, batch_norm=False
            )
        else:
            # Fall back to the original single-encoder network
            net = CauseSpecificNet(x.shape[1], shared, nodes, num_risks, self.labtrans.out_features, batch_norm=False)

        # Instantiate the appropriate DeepHit model
        if num_risks > 1:
            model = DeepHit(net, tt.optim.Adam, duration_index=self.labtrans.cuts)
        else:
            model = DeepHitSingle(net, tt.optim.Adam, duration_index=self.labtrans.cuts)
        
        
        model.make_dataloader = make_dict_dataloader # mokey patch    
        model.optimizer.set_lr(lr)
        
        dur_train, evt_train = self.labtrans.transform(t, e)
        dur_val, evt_val     = self.labtrans.transform(t_val, e_val)
        
        model.fit(x, (dur_train, evt_train), batch_size=batch, epochs=epochs, callbacks=callbacks, val_data=(x_val, (dur_val, evt_val)))
        return model
    
    def _nll_(self, model, x, t, e, *train):
        #print("x", x)
        from deephit.utils import LabTransform, MultiEncoderCauseSpecificNet, tt, make_dict_dataloader
        
        #return model.score_in_bathes_dataloader
        return model.score_in_batches(x, self.labtrans.transform(t, e))['loss']
    
    def _predict_(self, model, x, r, index):
        if len(self.risks) == 1:
            survival = model.predict_surv_df(x).values
        else:
            survival = 1 - self.predict_cif(model, x, is_dataloader=True)[r - 1]

        # Interpolate at the point of evaluation
        survival = pd.DataFrame(survival, columns = index, index = model.duration_index)
        predictions = pd.DataFrame(np.nan, columns = index, index = self.times)
        survival = pd.concat([survival, predictions]).sort_index(kind = 'stable').bfill().ffill()
        survival = survival[~survival.index.duplicated(keep='first')]
        return survival.loc[self.times].set_index(pd.MultiIndex.from_product([[r], self.times])).T
    
    # since internal function do not have the is_dataloader parameter, we need to override the function
    
    def predict_cif(self, model, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0, is_dataloader=None):
        from deephit.utils import tt
        pmf = self.predict_pmf(model, input, batch_size, False, eval_, to_cpu, num_workers, is_dataloader=is_dataloader)
        cif = pmf.cumsum(1)
        return cif.cpu()
        #return tt.utils.array_or_tensor(cif, numpy, input)
    
    def predict_pmf(self, model, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0, is_dataloader=None):
        
        from deephit.utils import make_dict_dataloader_predict, tt
        from pycox.models.utils import pad_col

        #print("x", x["embeddings"].shape)
        t_dummy = np.zeros((input[list(input.keys())[0]].shape[0]), dtype=np.int64)
        e_dummy = np.zeros((input[list(input.keys())[0]].shape[0]), dtype=np.int64)
        dur_dummy, evt_dummy = t_dummy, e_dummy #self.labtrans.transform(t_dummy, e_dummy)
        #print("duration_dummy", duration_dummy.shape)
        #print("events_dummy", events_dummy.shape)
        input_dl = (input, (dur_dummy, evt_dummy))
        x_dl = make_dict_dataloader_predict(input_dl, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        batch = next(iter(x_dl))
        print("batch", batch)
        print(batch)
        preds = model.predict(x_dl, batch_size, False, eval_, False, to_cpu, num_workers, is_dataloader=is_dataloader)
        pmf = pad_col(preds.view(preds.size(0), -1)).softmax(1)[:, :-1]
        pmf = pmf.view(preds.shape).transpose(0, 1).transpose(1, 2)
        #print("pmf", pmf)
        return pmf
        #return tt.utils.array_or_tensor(pmf, numpy, input)
    



class NFGExperiment(DSMExperiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):
        print("NFG Experiment")  
        from nfg import NeuralFineGray, NeuralFineGrayJoint
        epochs = hyperparameter.pop('epochs', 1000)
        
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)
        patience_max = hyperparameter.pop('patience_max', 3)
        contrastive_weight = hyperparameter.pop('contrastive_weight', 0.0)
        if isinstance(x, dict):
            model = NeuralFineGrayJoint(**hyperparameter, cause_specific = cause_specific)
        else:
            model = NeuralFineGray(**hyperparameter, cause_specific = cause_specific)
        #print("model", model)
        model.fit(x, t, e, n_iter = epochs, bs = batch, patience_max = patience_max,
                lr = lr, val_data = (x_val, t_val, e_val), contrastive_weight = contrastive_weight)
        
        return model

    def _predict_(self, model, x, r, index):
        return pd.DataFrame(model.predict_survival(x, self.times.tolist(), r if model.torch_model.risks >= r else 1), columns = pd.MultiIndex.from_product([[r], self.times]), index = index)

class DeSurvExperiment(NFGExperiment):

    def _fit_(self, x, t, e, x_val, t_val, e_val, hyperparameter, cause_specific):  
        from desurv import DeSurv

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)
        patience_max = hyperparameter.pop('patience_max', 3)

        model = DeSurv(**hyperparameter)
        model.fit(x, t, e, n_iter = epochs, bs = batch, patience_max = patience_max,
                lr = lr, val_data = (x_val, t_val, e_val))
        
        return model