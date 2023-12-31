CFG = {
    'epochs' : 15,
    'n_labels' : 1500,
    'emb_type' : 'protbert',
    'batch_size' : 2048,
    'train_folds' : [0, 1, 2, 3, 4],
    'input_shape' : 1024,
    'lr' : 0.001,
    'optimizer' : 'adam',
    'schedule' : 'cosine',
    'model_fn' : 'embeddingmodel_v1',
    'model_kwargs' : { 
        'n_hidden' : 1536
    }
}