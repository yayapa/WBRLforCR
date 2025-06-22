from pyparsing import col
from dsm.datasets import load_dataset as load_dsm
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from pycox import datasets
import pandas as pd
import numpy as np

EPS = 1e-8

def load_dataset(dataset='SUPPORT', path = './', normalize = True, multi_encoder = False, **kwargs):
    abdomen_dataset_path = "/home/dmitrii/GitHub/NeuralFineGray/data/datasets/labels/cvd_diabetes_copd_ckd_3m_unspec/"
    if dataset == 'GBSG':
        df = datasets.gbsg.read_df()
    elif dataset == 'METABRIC':
        df = datasets.metabric.read_df()
        df = df.rename(columns = {'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2', 
                                  'x4': 'Hormone', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy', 'x7': 'ER-positive', 
                                  'x8': 'Age at diagnosis'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'SYNTHETIC':
        df = datasets.rr_nl_nhp.read_df()
        df = df.drop([c for c in df.columns if 'true' in c], axis = 'columns')
    elif dataset == 'SEER':
        df = pd.read_csv(path + 'data/export.csv')
        df = process_seer(df)
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == 'SYNTHETIC_COMPETING':
        df = pd.read_csv('https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/SYNTHETIC/synthetic_comprisk.csv')
        df = df.drop(columns = ['true_time', 'true_label']).rename(columns = {'label': 'event', 'time': 'duration'})
        df['duration'] += EPS # Avoid problem of the minimum value 0
    elif dataset == "ABDOMEN_RADIOMICS":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df = pd.read_csv(abdomen_dataset_path + "wb_radiomics_pca_10.csv")
        col_rad = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "ABDOMEN_EMBEDDINGS":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "ABDOMEN_EMBEDDINGS_PCA":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_cls_pca21.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
        
    elif dataset == "ABDOMEN_EMBEDDINGS_PCA+TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        df_emb = pd.read_csv(abdomen_dataset_path + "embeddings_cls_pca21.csv")
        
        col_emb = list(set(list(df_emb.columns)) - set(['eid']))
        col_tabular = list(set(list(df_lifestyle.columns)) - set(['eid']))
        col_rad = []
        
        df = df_emb.merge(labels, on = 'eid', how = 'right')
        
        df = df.merge(df_lifestyle, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
        
    elif dataset == "ABDOMEN_EMBEDDINGS_BLOCKWISE":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_cls_blockwise.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "ABDOMEN_EMBEDDINGS_WAT":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_cls_wat.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "ABDOMEN_EMBEDDINGS+RADIOMICS":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df = pd.read_csv(abdomen_dataset_path + "wb_radiomics_pca_10.csv")
        df = df.merge(labels, on = 'eid', how = 'right')
        df_rad = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        #df_rad = df_rad.merge(labels, on = 'eid', how = 'right')
        df = df.merge(df_rad, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "ABDOMEN_TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        #df_blood = pd.read_csv(abdomen_dataset_path + "blood.csv")
        #df_spirio = pd.read_csv(abdomen_dataset_path + "spirio.csv")
        df = df_lifestyle
        #df = df_lifestyle.merge(df_blood, on = 'eid')
        #df = df.merge(df_spirio, on = 'eid')
        df = df.merge(labels, on = 'eid', how = 'right')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
        
    elif dataset == "ABDOMEN_EMBEDDINGS+TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        df_emb = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        
        col_emb = list(set(list(df_emb.columns)) - set(['eid']))
        col_tabular = list(set(list(df_lifestyle.columns)) - set(['eid']))
        col_rad = []
        
        df = df_emb.merge(labels, on = 'eid', how = 'right')
        
        df = df.merge(df_lifestyle, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    
    elif dataset == "ABDOMEN_EMBEDDINGS+RADIOMICS+TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        df_blood = pd.read_csv(abdomen_dataset_path + "blood.csv")
        #df_spirio = pd.read_csv(abdomen_dataset_path + "spirio.csv")
        df_rad = pd.read_csv(abdomen_dataset_path + "wb_radiomics_pca_10.csv")
        df_emb = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        
        col_rad = list(set(list(df_rad.columns)) - set(['eid']))
        col_emb = list(set(list(df_emb.columns)) - set(['eid']))
        col_tabular = list(set(list(df_lifestyle.columns) + list(df_blood.columns)) - set(['eid']))
        
        df = df_emb.merge(labels, on = 'eid', how = 'right')

        df = df.merge(df_rad, on = 'eid')
        df = df.merge(df_lifestyle, on = 'eid')
        df = df.merge(df_blood, on = 'eid')
       # df = df.merge(df_spirio, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
        
    elif dataset == "CARDIAC_RADIOMICS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "cardiac_radiomics.csv")
        col_rad = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
        
    elif dataset == "CARDIAC_RADIOMICS_STRUCTURAL_ONLY":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "cardiac_radiomics_structural_only.csv")
        col_rad = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CARDIAC_EMBEDDINGS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "cardiac_embeddings.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CARDIAC_TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        #df_blood = pd.read_csv(abdomen_dataset_path + "blood.csv")
        #df_spirio = pd.read_csv(abdomen_dataset_path + "spirio.csv")
        #df = df_lifestyle.merge(df_blood, on = 'eid')
        #df = df.merge(df_spirio, on = 'eid')
        df = df_lifestyle
        df = df.merge(labels, on = 'eid', how = 'right')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "CARDIAC_EMBEDDINGS+RADIOMICS+TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        df_blood = pd.read_csv(abdomen_dataset_path + "blood.csv")
        #df_spirio = pd.read_csv(abdomen_dataset_path + "spirio.csv")
        df_rad = pd.read_csv(abdomen_dataset_path + "cardiac_radiomics.csv")
        df_emb = pd.read_csv(abdomen_dataset_path + "cardiac_embeddings.csv")
        
        col_rad = list(set(list(df_rad.columns)) - set(['eid']))
        col_emb = list(set(list(df_emb.columns)) - set(['eid']))
        col_tabular = list(set(list(df_lifestyle.columns) + list(df_blood.columns)) - set(['eid']))
        
        df = df_emb.merge(labels, on = 'eid', how = 'right')
        
        df = df.merge(df_rad, on = 'eid')
        df = df.merge(df_lifestyle, on = 'eid')
        df = df.merge(df_blood, on = 'eid')
        #df = df.merge(df_spirio, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "CARDIAC_RADIOMICS+TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        df_blood = pd.read_csv(abdomen_dataset_path + "blood.csv")
        #df_spirio = pd.read_csv(abdomen_dataset_path + "spirio.csv")
        df_rad = pd.read_csv(abdomen_dataset_path + "cardiac_radiomics.csv")
        #df_emb = pd.read_csv(abdomen_dataset_path + "cardiac_embeddings.csv")
        
        col_rad = list(set(list(df_rad.columns)) - set(['eid']))
        #col_emb = list(set(list(df_emb.columns)) - set(['eid']))
        col_tabular = list(set(list(df_lifestyle.columns) + list(df_blood.columns)) - set(['eid']))
        
        df = df_rad.merge(labels, on = 'eid', how = 'right')
        
        df = df.merge(df_lifestyle, on = 'eid')
        df = df.merge(df_blood, on = 'eid')
        #df = df.merge(df_spirio, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "CARDIAC_WB_RADIOMICS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "wb_radiomics_pca_10.csv")
        col_rad = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CARDIAC_WB_EMBEDDINGS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CARDIAC_WB_RADIOMICS+RADIOMICS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "wb_radiomics_pca_10.csv")
        df = df.merge(labels, on = 'eid', how = 'right')
        df_rad = pd.read_csv(abdomen_dataset_path + "cardiac_radiomics.csv")
        df = df.merge(df_rad, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "CARDIAC_WB_EMBEDDINGS+RADIOMICS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        df_rad = pd.read_csv(abdomen_dataset_path + "cardiac_radiomics.csv")
        col_rad = list(set(list(df_rad.columns)) - set(['eid']))
        df = df.merge(df_rad, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "CARDIAC_WB_EMBEDDINGS+RADIOMICS+TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        #df_blood = pd.read_csv(abdomen_dataset_path + "blood.csv")
        #df_spirio = pd.read_csv(abdomen_dataset_path + "spirio.csv")
        df_emb = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        df_rad = pd.read_csv(abdomen_dataset_path + "cardiac_radiomics.csv")
        col_rad = list(set(list(df_rad.columns)) - set(['eid']))
        col_emb = list(set(list(df_emb.columns)) - set(['eid']))
        col_tabular = list(set(list(df_lifestyle.columns)) - set(['eid']))
        df = df_emb.merge(labels, on = 'eid', how = 'right')
        df = df.merge(df_rad, on = 'eid')
        df = df.merge(df_lifestyle, on = 'eid')
        #df = df.merge(df_blood, on = 'eid')
        #df = df.merge(df_spirio, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
        
    elif dataset == "CARDIAC_RADIOMICS_REDUCED_CORR":
        df = pd.read_csv("/home/dmitrii/GitHub/NeuralFineGray/data/datasets/labels/cvd_diabetes_copd_ckd_3m_unspec/cardiac_radiomics_wo_corr0.95.csv")
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = df.merge(labels, on = 'eid', how = 'right')
        drop_cols =  ["eid"]
        #col_rad = list(set(list(df.columns)) - set(drop_cols)) 
        df = df.drop(columns=drop_cols)
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "ABDOMEN_EMBEDDINGS_REDUCED_CORR":
        df = pd.read_csv("/home/dmitrii/GitHub/NeuralFineGray/data/datasets/labels/cvd_diabetes_copd_ckd_3m_unspec/embeddings_wo_corr0.95.csv")
        labels = pd.read_csv(abdomen_dataset_path + "labels.csv")
        df = df.merge(labels, on = 'eid', how = 'right')
        drop_cols =  ["eid"]
        #col_rad = list(set(list(df.columns)) - set(drop_cols))
        df = df.drop(columns=drop_cols)
        # rename column time_to_event to duration   
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "CARDIAC_WB_EMBEDDINGS+EMBEDDINGS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        # rename all feature_0, feature_1,... to feature_wb_0, feature_wb_1,...
        df = df.rename(columns=lambda x: x.replace('feature_', 'feature_wb_') if 'feature_' in x else x)
        df_cardiac_emb = pd.read_csv(abdomen_dataset_path + "cardiac_embeddings.csv")
        col_rad = list(set(list(df_cardiac_emb.columns)) - set(['eid']))
        df = df.merge(df_cardiac_emb, on = 'eid', how = 'left')

        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CARDIAC_WB_EMBEDDINGS+EMBEDDINGS+TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        #df_blood = pd.read_csv(abdomen_dataset_path + "blood.csv")
        #df_spirio = pd.read_csv(abdomen_dataset_path + "spirio.csv")
        df_emb = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        df_cardiac_emb = pd.read_csv(abdomen_dataset_path + "cardiac_embeddings.csv")
        col_emb = list(set(list(df_emb.columns)) - set(['eid']))
        col_tabular = list(set(list(df_lifestyle.columns)) - set(['eid']))
        col_rad = list(set(list(df_cardiac_emb.columns)) - set(['eid']))
        df = df_emb.merge(df_cardiac_emb, on = 'eid', how = 'left')
        df = df.merge(df_lifestyle, on = 'eid')
        df = df.merge(labels, on = 'eid', how = 'right')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "CARDIAC_ECG":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_ecg_gp.csv")
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CARDIAC_ECG_DOMAINSPECIFIC":
        labels = pd.read_csv(abdomen_dataset_path + "labels_ihd_hd_stroke.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_ecg_gp_domainSpecific.csv")
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CVD_ECG_DOMAINSPECIFIC":
        labels = pd.read_csv(abdomen_dataset_path + "labels_cvd.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_ecg_gp_domainSpecific.csv")
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CVD_TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels_cvd.csv")
        df = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        col_tabular = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CVD_WB_EMBEDDINGS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_cvd.csv")
        df = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CVD_EMBEDDINGS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_cvd.csv")
        df = pd.read_csv(abdomen_dataset_path + "cardiac_embeddings.csv")
        col_emb = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CVD_WB_RADIOMICS":
        labels = pd.read_csv(abdomen_dataset_path + "labels_cvd.csv")
        df = pd.read_csv(abdomen_dataset_path + "wb_radiomics_pca_10.csv")
        col_rad = list(set(list(df.columns)) - set(['eid']))
        df = df.merge(labels, on = 'eid', how = 'right')
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df = df.drop(columns = ['eid'])
        df['duration'] += EPS
    elif dataset == "CVD_WB_EMBEDDINGS+TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels_cvd.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        #df_blood = pd.read_csv(abdomen_dataset_path + "blood.csv")
        #df_spirio = pd.read_csv(abdomen_dataset_path + "spirio.csv")
        df_emb = pd.read_csv(abdomen_dataset_path + "embeddings_cls.csv")
        col_emb = list(set(list(df_emb.columns)) - set(['eid']))
        col_tabular = list(set(list(df_lifestyle.columns)) - set(['eid']))
        col_rad = []
        df = df_emb.merge(labels, on = 'eid', how = 'right')
        df = df.merge(df_lifestyle, on = 'eid')
        #df = df.merge(df_blood, on = 'eid')
        #df = df.merge(df_spirio, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    elif dataset == "CVD_WB_RADIOMICS+TABULAR":
        labels = pd.read_csv(abdomen_dataset_path + "labels_cvd.csv")
        df_lifestyle = pd.read_csv(abdomen_dataset_path + "lifestyle.csv")
        #df_blood = pd.read_csv(abdomen_dataset_path + "blood.csv")
        #df_spirio = pd.read_csv(abdomen_dataset_path + "spirio.csv")
        df_rad = pd.read_csv(abdomen_dataset_path + "wb_radiomics_pca_10.csv")
        col_rad = list(set(list(df_rad.columns)) - set(['eid']))
        col_tabular = list(set(list(df_lifestyle.columns)) - set(['eid']))
        df = df_rad.merge(labels, on = 'eid', how = 'right')
        df = df.merge(df_lifestyle, on = 'eid')
        #df = df.merge(df_blood, on = 'eid')
        #df = df.merge(df_spirio, on = 'eid')
        df = df.drop(columns = ['eid'])
        # rename column time_to_event to duration
        df = df.rename(columns={"time_to_event": "duration"})
        df['duration'] += EPS
    if "53-2.0" in df.columns:
        df = df.drop(columns = ["53-2.0"])
        
    # replace nan with mean of the column
    print("Number of rows with NaN:", len(df[df.isna().any(axis=1)]))
    df = df.fillna(df.mean(numeric_only=True))
    print("Number of rows with NaN after filling:", df.isna().sum().sum())
    
    covariates = df.drop(['duration', 'event'], axis = 'columns')

    if normalize:
        if "TABULAR" in dataset: 
            continuous_columns = ['Time spent watching television (TV)',
        'Sleep duration', 'Salad raw vegetable intake', 'Fresh fruit intake',
        'BMI', 'Age', 'Summed MET minutes per week for all activity']
            continuous_columns_bp = ["diastolic_bp_mean", "systolic_bp_mean", "pulse_mean"]
            continuous_columns_spirio = ["fvc_best", "fev1_best", "pef_best"]
            continuous_columns = continuous_columns + continuous_columns_bp + continuous_columns_spirio
            # scale continious columns
            covariates[continuous_columns] = StandardScaler().fit_transform(
                covariates[continuous_columns].values).astype(float) if normalize else covariates[continuous_columns].values.astype(float
            )
        if "EMBEDDINGS" in dataset:
            covariates[col_emb] = StandardScaler().fit_transform(
                covariates[col_emb].values).astype(float) if normalize else covariates[col_emb].values.astype(float
            )
            
        if "RADIOMICS" in dataset:
            covariates[col_rad] = StandardScaler().fit_transform(
                covariates[col_rad].values).astype(float) if normalize else covariates[col_rad].values.astype(float
            )
        
    #else:
    #    covariates = StandardScaler().fit_transform(covariates.values).astype(float) if normalize else covariates.values.astype(float)
    
    if multi_encoder:
        multi_covariates = {}
        cols_to_return = []
        if "EMBEDDINGS" in dataset:
            if dataset.count("EMBEDDINGS") > 1:
                multi_covariates['embeddings'] = covariates[col_emb].astype(float)
                multi_covariates['radiomics'] = covariates[col_rad].astype(float)
            else:
                multi_covariates['embeddings'] = covariates[col_emb]
            cols_to_return += col_emb
        if "RADIOMICS" in dataset:
            multi_covariates['radiomics'] = covariates[col_rad]
            cols_to_return += col_rad
        if "TABULAR" in dataset:
            multi_covariates['tabular'] = covariates[col_tabular]
            cols_to_return += col_tabular
        return multi_covariates,\
           df['duration'].values.astype(float),\
           df['event'].values.astype(int),\
           cols_to_return

    print("covariate shape", covariates.shape)
    print("duration shape", df['duration'].values.shape)
    print("event shape", df['event'].values.shape)
    return covariates,\
           df['duration'].values.astype(float),\
           df['event'].values.astype(int),\
           covariates.columns

def process_seer(df):
    # Remove multiple visits
    df = df.groupby('Patient ID').first().drop(columns= ['Site recode ICD-O-3/WHO 2008'])

    # Encode using dictionary to remove missing data
    df["RX Summ--Surg Prim Site (1998+)"].replace('126', np.nan, inplace = True)
    df["Sequence number"].replace(['88', '99'], np.nan, inplace = True)
    df["Regional nodes positive (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
    df["Regional nodes examined (1988+)"].replace(['95', '96', '97', '98', '99', '126'], np.nan, inplace = True)
    df = df.replace(['Blank(s)', 'Unknown'], np.nan).rename(columns = {"Survival months": "duration"})

    # Remove patients without survival time
    df = df[~df.duration.isna()]

    # Outcome 
    df['duration'] = df['duration'].astype(float)
    df['event'] = df["SEER cause-specific death classification"] == "Dead (attributable to this cancer dx)" # Death 
    df['event'].loc[(df["COD to site recode"] == "Diseases of Heart") & (df["SEER cause-specific death classification"] == "Alive or dead of other cause")] = 2 # CVD 

    df = df.drop(columns = ["COD to site recode"])

    # Imput and encode categorical
    ## Categorical
    categorical_col = ["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)", "Laterality", 
        "Diagnostic Confirmation", "Histology recode - broad groupings", "Chemotherapy recode (yes, no/unk)",
        "Radiation recode", "ER Status Recode Breast Cancer (1990+)", "PR Status Recode Breast Cancer (1990+)",
        "Histologic Type ICD-O-3", "ICD-O-3 Hist/behav, malignant", "Sequence number", "RX Summ--Surg Prim Site (1998+)",
        "CS extension (2004-2015)", "CS lymph nodes (2004-2015)", "CS mets at dx (2004-2015)", "Origin recode NHIA (Hispanic, Non-Hisp)"]
    ordinal_col = ["Age recode with <1 year olds", "Grade", "Year of diagnosis"]

    imputer = SimpleImputer(strategy='most_frequent')
    enc = OrdinalEncoder()
    df_cat = pd.DataFrame(enc.fit_transform(imputer.fit_transform(df[categorical_col])), columns = categorical_col, index = df.index)
    
    df_ord = pd.DataFrame(imputer.fit_transform(df[ordinal_col]), columns = ordinal_col, index = df.index)
    df_ord = df_ord.replace(
      {age: number
        for number, age in enumerate(['01-04 years', '05-09 years', '10-14 years', '15-19 years', '20-24 years', '25-29 years',
        '30-34 years', '35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years', 
        '60-64 years', '65-69 years', '70-74 years', '75-79 years', '80-84 years', '85+ years'])
      }).replace({
        grade: number
        for number, grade in enumerate(['Well differentiated; Grade I', 'Moderately differentiated; Grade II',
       'Poorly differentiated; Grade III', 'Undifferentiated; anaplastic; Grade IV'])
      })

    ## Numerical
    numerical_col = ["Total number of in situ/malignant tumors for patient", "Total number of benign/borderline tumors for patient",
          "CS tumor size (2004-2015)", "Regional nodes examined (1988+)", "Regional nodes positive (1988+)"]
    imputer = SimpleImputer(strategy='mean')
    df_num = pd.DataFrame(imputer.fit_transform(df[numerical_col].astype(float)), columns = numerical_col, index = df.index)

    return pd.concat([df_cat, df_num, df_ord, df[['duration', 'event']]], axis = 1)
    
    