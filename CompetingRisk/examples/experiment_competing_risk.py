
# Comparsion models for competing risks
# In this script we train the different models for competing risks
from operator import mul
import sys
from nfg import datasets
from experiment import *
from pycox.models.data import pair_rank_mat

random_seed = 0

# Open dataset
#dataset = "ABDOMEN_TABULAR" #sys.argv[1] # FRAMINGHAM, SYNTHETIC_COMPETING, PBC, SEER
dataset = sys.argv[1] # FRAMINGHAM, SYNTHETIC_COMPETING, PBC, SEER
#num_encoders = 3 
num_encoders = 1
multi_encoder = num_encoders > 1


# Specific fold selection
fold = None
if len(sys.argv) == 3:
    fold = int(sys.argv[2])

print("Script running experiments on ", dataset)
x, t, e, covariates = datasets.load_dataset(dataset, competing = True, normalize=False, multi_encoder=multi_encoder) 

# Hyperparameters
max_epochs = 1000
grid_search = 100
#layers = [[i] * (j + 1) for i in [25, 50] for j in range(4)]
#layers_large = [[i] * (j + 1) for i in [25, 50] for j in range(8)]
#layers_large_extra = [[25, 50], [50, 50], [50, 50, 50, 50], [128, 128], [128, 128, 128], [256, 256, 256], [128, 256, 512], [512, 256, 128], [512, 512, 512], [128, 1024, 128], [128, 1024, 1024, 128]]
layers = [[i] * (j + 1) for i in [32, 64] for j in range(4)]
layers_large = [[i] * (j + 1) for i in [32, 64] for j in range(8)]
layers_custom = [[i] * (j + 1) for i in [32, 64, 128, 512, 1024] for j in range(4)]
layers_custom_large = [[i] * (j + 1) for i in [32, 64, 128, 512, 1024] for j in range(8)]

batch = [100, 250] if dataset != 'SEER' else [1000, 5000]
batch = [1000, 3200]
from itertools import product

# DSM
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'k' : [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'distribution' : ['LogNormal', 'Weibull'],
    'layers' : layers_custom,
    #'fusion_outdim': [32, 64, 128, 256, 512],
    
    #'layers': [{"embeddings": emb, "tabular": tab} for emb, tab in product([[x["embeddings"].shape[1]] + l for l in layers_custom], [[x["tabular"].shape[1]] + l for l in layers])]
    #'layers': [{"embeddings": emb, "radiomics": tab} for emb, tab in product([[x["embeddings"].shape[1]] + l for l in layers_custom], [[x["radiomics"].shape[1]] + l for l in layers_custom])]
    #'layers': [{"embeddings": emb, "tabular": tab, "radiomics": rad} for emb, tab, rad in product([[x["embeddings"].shape[1]] + l for l in layers_custom], [[x["tabular"].shape[1]] + l for l in layers], [[x["radiomics"].shape[1]] + l for l in layers_custom])],
}


#MultiEncoderDSMExperiment.create(param_grid, n_iter = grid_search, path = 'Results_04.06/{}_dsm_multi_encoders'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
DSMExperiment.create(param_grid, n_iter = grid_search, path = 'Results_13.06/{}_dsm'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
#DSMExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsmnc'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e == 1)

# NFG Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-2, 1e-3, 1e-4], # 1e-4
    'batch': batch,
    
    'dropout': [0., 0.25, 0.5, 0.75],

    'layers_surv': layers_custom,
    #'fusion_outdim': [32, 64, 128, 256, 512],
    #'layers': [{"embeddings": emb, "tabular": tab} for emb, tab in product([[x["embeddings"].shape[1]] + l for l in layers_custom], [[x["tabular"].shape[1]] + l for l in layers])],
    #'layers': [{"embeddings": emb, "radiomics": tab} for emb, tab in product([[x["embeddings"].shape[1]] + l for l in layers_custom], [[x["radiomics"].shape[1]] + l for l in layers_custom])],
    
    #'layers': [{"embeddings": emb, "tabular": tab, "radiomics": rad} for emb, tab, rad in product([[x["embeddings"].shape[1]] + l for l in layers_custom], [[x["tabular"].shape[1]] + l for l in layers], [[x["radiomics"].shape[1]] + l for l in layers_custom])],
    'layers' : layers_custom,
    'act': ['Tanh'],
    #'contrastive_weight': [0.01]
}
#NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results_test_split/{}_nfg'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results_13.06/{}_nfg'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
#NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgnc'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e == 1)
#NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgcs'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e, cause_specific = True)

#NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results_04.06/{}_nfg_multi_encoders'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e, cause_specific = False)
#NFGExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_nfgmono'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)

# Desurv
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'embedding': [True],
    'layers_surv': layers_custom,
    'layers': layers_custom,
    'act': ['Tanh'],
}
#DeSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_ds'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
#DeSurvExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dsnc'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e == 1)

layers_deephit = [256, 256]
layers_deephit_fused = [256, 256]
# DeepHit Competing risk
param_grid = {
    #'n': [15, 30, 50, 100, 200, 300], 
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,

    'nodes' : layers_custom,
    'shared' : layers_custom,
    
    
    #'encoder_nodes': [{"embeddings": emb, "radiomics": tab} for emb, tab in product([[x["embeddings"].shape[1]] + l for l in layers_custom], [[x["radiomics"].shape[1]] + l for l in layers_custom])],
    
    'encoder_nodes' : layers_custom,
    #'encoder_nodes': [{"embeddings": emb, "tabular": tab} for emb, tab in product([[x["embeddings"].shape[1]] + l for l in layers_custom], [[x["tabular"].shape[1]] + l for l in layers])],
    #'encoder_nodes': [{"embeddings": emb, "tabular": tab, "radiomics": rad} for emb, tab, rad in product([[x["embeddings"].shape[1]] + l for l in layers_custom], [[x["tabular"].shape[1]] + l for l in layers], [[x["radiomics"].shape[1]] + l for l in layers_custom])],
    #'fused_nodes' : layers_custom,
}

#MultiEncoderDeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results_13.06/{}_dh_multi_encoders'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)

DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results_13.06/{}_dh'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e)
#DeepHitExperiment.create(param_grid, n_iter = grid_search, path = 'Results/{}_dhnc'.format(dataset), random_seed = random_seed, fold = fold).train(x, t, e == 1)