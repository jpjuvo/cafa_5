CFG = {
    'epochs' : 57,
    'n_labels' : 1500,
    'emb_type' : 'all',
    'emb_dict' : {
        'emb_t5_p': 0.79997, 
        'emb_esm2_p': 0.278908, 
        'emb_protbert_p': 0
    },
    'batch_size' : 2048,
    'train_folds' : [0, 1, 2, 3, 4],
    'lr' : 0.00013752313600904308,
    'optimizer' : 'adam',
    'schedule' : 'cosine',
    'model_fn' : 'embeddingmodel_v1',
    'model_kwargs' : { 
        'n_hidden' : 4092,
        'dropout1_p': 0.22818862455835717, 
        'use_norm': True,
        'use_residual': True
    }
}