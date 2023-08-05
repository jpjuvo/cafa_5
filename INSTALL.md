# Install

Install in conda env.

```bash
conda create -n cafa5 --yes python=3.9 jupyter
conda activate cafa5

conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

conda deactivate
conda activate cafa5

pip install --upgrade pip

pip install -r requirements.txt
conda install -c conda-forge umap-learn

```

## Competition metric

The metric used in this competition is weighted F-score for three ontologies. [CAFA-evaluator](https://github.com/BioComputingUP/CAFA-evaluator) implements this metric so we can use it as a submodule. To install, run:

```
git submodule update --init --recursive --remote --merge
```

## Weights and Biases logging

[Login to your W&B account](https://docs.wandb.ai/quickstart)

## Download data

Place these datasets into `input` folder according to the folder structure shown below.  

- [competition data](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data) 
- [T5 embeddings](https://www.kaggle.com/datasets/sergeifironov/t5embeds)
- [ESM2 3B embeddings](https://www.kaggle.com/datasets/andreylalaley/4637427)
- [ProtBERT embeddings](https://www.kaggle.com/datasets/henriupton/protbert-embeddings-for-cafa5)


### Folder & file structure
```
root
|_input                  (datasets - gitignored)
|  |_cafa-5-protein-function-prediction
|  | |_...
|  |_t5embeds
|  | |_...
|  |_esm23b
|  | |_...
|  |_protbert
|    |_...
|_notebooks              (jupyter notebooks)
|  |_...
|_src                    (python files)
|  |_...
|_...
```