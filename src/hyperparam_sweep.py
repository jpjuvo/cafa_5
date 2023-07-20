import os
import sys
import gc
import numpy as np
import pandas as pd
import json
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold
import optuna

from configs import get_config
from models import get_model
from cafa_utils import *
from data_utils import *
from metric import *
from eval import calculate_metric, evaluate_all_folds
from binarymetrics import BinaryMetrics

sys.path.append('.')
sys.path.append('src')

# constants
N_SPLITS = 5
RND_SEED = 2023

# global vars late init
train_terms_updated = None
train_protein_ids = None
test_protein_ids = None
train_df, test_df = None, None
num_of_labels = 1500
labels_to_consider = None
labels_df = None


def train_one_ep(fold_model, dl, optimizer, criterion, scaler, binarymetrics, logs, device, log_metrics=True):
    """ Train for one epoch """
    fold_model.train()
    running_loss_trn = 0
        
    tk0 = tqdm(dl, total=int(len(dl)), leave=False, position=2)
    for cpu_data in tk0:
        data = (cpu_data[0].to(device), cpu_data[1].to(device))
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = fold_model(data[0])
            loss = criterion(logits, data[1])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # calculate score
        if log_metrics:
            binarymetrics.accumulate_batch(F.sigmoid(logits).detach(), data[1])

        running_loss_trn += loss.item()
    
    logs['train_loss'] = running_loss_trn / len(dl)

    if log_metrics:
        binarymetrics.compute(logs=logs, prefix='train_')


def evaluate(fold_model, dl_val, criterion, binarymetrics, logs, device):
    """ Evaluate dl_val and calculate loss and metrics """
    fold_model.eval()
    loss_val  = 0
    val_probs = []
    val_trues = []

    tk1 = tqdm(
        dl_val, total=int(len(dl_val)), 
        desc="validating", ascii=True, leave=False, position=2)

    for cpu_data in tk1:
        data = (cpu_data[0].to(device), cpu_data[1].to(device))
        with torch.no_grad() and torch.cuda.amp.autocast():
            logits = fold_model(data[0])
            loss_val += criterion(logits, data[1]).item()
            probs = F.sigmoid(logits).detach()

            val_trues.append(cpu_data[1].numpy())
            val_probs.append(probs.cpu().numpy())

            binarymetrics.accumulate_batch(probs, data[1])
    
    logs['val_loss'] = loss_val / len(dl_val)
    binarymetrics.compute(logs=logs, prefix='val_')
    
    return np.concatenate(val_probs, 0), np.concatenate(val_trues, 0)


def train_cv(config, sweep_params:dict, device:str='cuda', metric_to_monitor='val_f1'):
    global num_of_labels, train_terms_updated, train_protein_ids, test_protein_ids, \
        train_df, test_df, labels_to_consider, labels_df

    CFG = get_config(config)
    CFG.update(sweep_params)
    num_of_labels = CFG["n_labels"] if 'n_labels' in CFG else 1500

    (train_terms_updated, train_protein_ids, test_protein_ids, 
     train_df, test_df, labels_to_consider, labels_df) = prepare_dataframes(
        n_labels=num_of_labels,
        emb_type=CFG["emb_type"] if 'emb_type' in CFG else 't5',
        verbose=0
        )
    binarymetrics = BinaryMetrics(device=device)

    # Create a KFold object
    if train_sequence_clusters_df is None:
        skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RND_SEED)
        tkfold = tqdm(enumerate(skf.split(train_df)), desc="Fold", leave=False, position=0)
    else:
        # include similar proteins in test sets
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND_SEED)
        tkfold = tqdm(enumerate(skf.split(train_df, train_sequence_clusters_df['cluster_id'].values)), desc="Fold", leave=True, position=0)
    
    fold_metrics = []

    for fold, (train_index, test_index) in tkfold:
        if fold not in CFG['train_folds']: continue 

        fold_cfg = {'fold':fold}
        fold_cfg.update(CFG)

        dl, dl_val = get_dataloaders(
            train_index, test_index, 
            train_df, labels_df,
            batch_size=CFG['batch_size'])

        criterion = nn.BCEWithLogitsLoss().to(device)
        fold_model = get_model(
            model_fn=CFG["model_fn"], 
            input_shape=CFG["input_shape"],
            num_of_labels=num_of_labels,
            **CFG["model_kwargs"])
        fold_model.to(device)

        if CFG['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(fold_model.parameters(), lr=CFG['lr']) 
        elif CFG['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(fold_model.parameters(), lr=CFG['lr'])
        else:
            raise Exception(f'optimizer {CFG["optimizer"]} not implemented')
            
        scheduler = None if CFG['schedule'] == 'none' else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG["epochs"], eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler()

        tkep = tqdm(range(1, CFG["epochs"] + 1), desc="epoch", leave=False, position=1)
        for epoch in tkep:
            logs = {}
            
            train_one_ep(fold_model, dl, optimizer, 
                         criterion, scaler, binarymetrics, logs, device,
                         log_metrics=False)
            
            if scheduler is not None:
                scheduler.step()
                
            #release GPU memory cache
            torch.cuda.empty_cache()
            gc.collect()
        
            #eval
            val_probs, val_trues = evaluate(fold_model, dl_val, criterion, binarymetrics, logs, device)

            #release GPU memory cache
            del val_probs, val_trues
            torch.cuda.empty_cache()
            gc.collect()

            if epoch == CFG['epochs']:
                fold_metrics.append(logs[metric_to_monitor])

    return np.mean(fold_metrics)

def objective(trial):
    config = 'embedding_esm2_3b_v1'
    ep = trial.suggest_int('epochs', 5, 100)
    n_hidden = trial.suggest_categorical('n_hidden', [512, 1024, 2048, 4096])
    dropout1_p = trial.suggest_float('dropout1_p', 0, 0.8)
    use_norm = trial.suggest_categorical('use_norm', [True, False])
    
    sweep_params = {'epochs' : ep, 'model_kwargs' : 
                    {'n_hidden' : n_hidden, 'dropout1_p': dropout1_p, 'use_norm':use_norm}}
    metric = train_cv(config=config, sweep_params=sweep_params)
    return metric

if __name__ == "__main__":

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, gc_after_trial=True)

    print('')
    print(study.best_params)