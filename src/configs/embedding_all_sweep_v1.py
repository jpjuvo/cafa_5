CFG = {
    'epochs' : 36,
    'n_labels' : 1500,
    'emb_type' : 'all',
    'emb_dict' : {
        'emb_t5_p': 0.9891911405115349, 
        'emb_esm2_p': 0.43094809163443176, 
        'emb_protbert_p': 0.02375221595560384
    },
    'batch_size' : 2048,
    'train_folds' : [0, 1, 2, 3, 4],
    'lr' : 0.0008702328970615781,
    'optimizer' : 'adam',
    'schedule' : 'cosine',
    'model_fn' : 'embeddingmodel_v1',
    'model_kwargs' : { 
        'n_hidden' : 2048,
        'dropout1_p': 0.42815632694378986, 
        'use_norm': True,
        'use_residual': True
    }
}