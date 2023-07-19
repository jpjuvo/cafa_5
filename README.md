![](https://storage.googleapis.com/kaggle-competitions/kaggle/41875/logos/header.png?t=2023-02-28-14-27-02)

# CAFA 5 Protein Function Prediction

Predict the biological function of a protein. 

A solution codebase to the [CAFA5 Kaggle competition](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/overview).

## Python environment & data setup

Check [INSTALL.md](./INSTALL.md).

## Train MLP with T5 embeddings

This creates five CV fold models. The folds are split with protein sequence similarity clustering so that similar proteins end up in train and test splits. This is to mimic the competition train & test data where test proteins are very similar to train proteins. 

To train with a config in `src/configs/embedding_v1.py`, run:

```bash
python src/train.py -c embedding_v1
```

*Flags*
- `-c`  or  `--config`,     default="embedding_v1" -    config name without .py extension
- `-d`  or  `--device`,     default="cuda" -    cuda or cpu
- `-e`  or  `--eval_every`,     default=2 - how often to evaluate between epochs
- `-m`  or   `--metric_every`,  default=50 -    how often to evaluate competition metric between epochs

Sample **embedding_v1 config.py** contents:
```
CFG = {
    'epochs' : 20,
    'batch_size' : 2048,
    'train_folds' : [0, 1, 2, 3, 4],        # what folds out of 0-4 to train 
    'input_shape' : 1024,                   # T5 embedding shape
    'lr' : 0.001,
    'optimizer' : 'adam',                   # adam and adamw implemented
    'schedule' : 'none',                    # if not 'none', Uses CosineAnnealing 
    'model_fn' : 'embeddingmodel_v1',       # model file name without .py in models dir
    'model_kwargs' : {                      # kwargs passed for model init
        'n_hidden' : 1024
    }
}
```

Running `train.py` creates timestamped run output folder under `./output/` with `fold-n` subfolders. Models and out-of-fold predictions are saved under each fold dir and metrics & losses are logged to Weights and Biases. Sample logs below:

![logs](./media/logs.png)

## Calculating a DIAMOND submission

This takes a ~day to generate.
See [DIAMOND docs](https://github.com/bbuchfink/diamond)

Follow instructions in `notebooks/diamond/diamond.ipynb` and `notebooks/diamond/diamond_add_data_NetGo2.0.ipynb`.
