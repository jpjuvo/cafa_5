CFG = {
    'epochs' : 10,
    'n_labels' : 1500,
    'emb_type' : 'umap512',
    'batch_size' : 2048,
    'train_folds' : [0, 1, 2, 3, 4],
    'input_shape' : 512,
    'lr' : 0.01,
    'optimizer' : 'adam',
    'schedule' : 'cosine',
    'model_fn' : 'embeddingmodel_v1',
    'model_kwargs' : { 
        'n_hidden' : 1024
    }
}