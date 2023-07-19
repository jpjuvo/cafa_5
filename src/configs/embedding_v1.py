CFG = {
    'epochs' : 20,
    'batch_size' : 2048,
    'train_folds' : [0, 1, 2, 3, 4],
    'input_shape' : 1024,
    'lr' : 0.001,
    'optimizer' : 'adam',
    'schedule' : 'none',
    'model_fn' : 'embeddingmodel_v1',
    'model_kwargs' : { 
        'n_hidden' : 512
    }
}