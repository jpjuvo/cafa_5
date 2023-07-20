CFG = {
    'epochs' : 50,
    'n_labels' : 1500,
    'emb_type' : 'esm2_3b',
    'batch_size' : 2048,
    'train_folds' : [0, 1, 2, 3, 4],
    'input_shape' : 2560,
    'lr' : 0.001,
    'optimizer' : 'adam',
    'schedule' : 'cosine',
    'model_fn' : 'embeddingmodel_v1',
    'model_kwargs' : { 
        'n_hidden' : 1024
    }
}