import os
import sys
import numpy as np
import pandas as pd
import wandb
import json
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, KFold

from configs import get_config
from models import get_model
from cafa_utils import *
from data_utils import *
from metric import *
from eval import calculate_metric
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

sys.path.append('.')
sys.path.append('src')

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config", default="embedding_v1", help="config name without .py extension")
argParser.add_argument("-d", "--device", default="cuda", help="cuda or cpu")
argParser.add_argument("-e", "--eval_every", default=2, type=int, help="how often to evaluate between epochs")
argParser.add_argument("-m", "--metric_every", default=50, type=int, help="how often to evaluate metric between epochs")

# constants
N_SPLITS = 5
RND_SEED = 2023


def train_one_ep(fold_model, dl, optimizer, criterion, scaler, logs):
    """ Train for one epoch """
    fold_model.train()
    running_loss_trn = 0
    train_probs = []
    train_trues = []
        
    tk0 = tqdm(dl, total=int(len(dl)), leave=False, position=2)
    for i,cpu_data in enumerate(tk0):
        data = (cpu_data[0].to(device), cpu_data[1].to(device))
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            y_true = data[1]
            logits = fold_model(data[0])
            loss = criterion(logits, y_true)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # calculate score
        probs = F.sigmoid(logits)
        train_trues.append(y_true.cpu().numpy())
        train_probs.append(probs.cpu().detach().numpy())

        running_loss_trn += loss.item()
        tk0.set_postfix(loss=(running_loss_trn / (i + 1)))
        
    train_probs = np.concatenate(train_probs, 0)
    train_trues = np.concatenate(train_trues, 0)
        
    epoch_loss_trn = running_loss_trn / len(dl)
    logs['train_loss'] = epoch_loss_trn
    logs['train_auc'] = roc_auc_score(train_trues, train_probs)
    
    int_trues = train_trues.astype(np.int32).ravel()
    int_preds = (train_probs > 0.5).astype(np.int32).ravel()
    logs['train_f1'] = f1_score(int_trues, int_preds)
    logs['train_precision'] = precision_score(int_trues, int_preds)
    logs['train_recall'] = recall_score(int_trues, int_preds)


def evaluate(fold_model, dl_val, criterion, logs):
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
            loss = criterion(logits, data[1])
            probs = F.sigmoid(logits).detach().cpu().numpy()
            y_true = data[1]
            loss_val += loss.item()
            val_trues.append(y_true.cpu().numpy())
            val_probs.append(probs)
    
    val_probs = np.concatenate(val_probs, 0)
    val_trues = np.concatenate(val_trues, 0)
    loss_val  /= len(dl_val)
    
    logs['val_loss'] = loss_val
    logs['val_auc'] = roc_auc_score(val_trues, val_probs)
    
    int_trues = val_trues.astype(np.int32).ravel()
    int_preds = (val_probs > 0.5).astype(np.int32).ravel()
    logs['val_f1'] = f1_score(int_trues, int_preds)
    logs['val_precision'] = precision_score(int_trues, int_preds)
    logs['val_recall'] = recall_score(int_trues, int_preds)
    
    return val_probs, val_trues


def main(config, device:str, eval_every:int, metric_every:int):
    CFG = get_config(config)
    out_dir = create_out_dir()
    group_id = out_dir.split('/')[-1]
    

    # Create a KFold object
    if train_sequence_clusters_df is None:
        skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RND_SEED)
        tkfold = tqdm(enumerate(skf.split(train_df)), desc="Fold", leave=True, position=0)
    else:
        # include similar proteins in test sets
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND_SEED)
        tkfold = tqdm(enumerate(skf.split(train_df, train_sequence_clusters_df['cluster_id'].values)), desc="Fold", leave=True, position=0)
    
    for fold, (train_index, test_index) in tkfold:
        if fold not in CFG['train_folds']: continue 
        fold_out_dir = os.path.join(out_dir, f'fold-{fold}')
        if not os.path.exists(fold_out_dir): os.mkdir(fold_out_dir)

        fold_cfg = {'fold':fold}
        fold_cfg.update(CFG)
        wandb.init(project='cafa-5', group=group_id, dir=fold_out_dir, config=fold_cfg)

        dl, dl_val = get_dataloaders(
            train_index, test_index, 
            train_df, labels_df,
            batch_size=CFG['batch_size'])

        # metric
        fold_terms = train_terms_updated.loc[train_terms_updated.EntryID.isin(train_protein_ids[test_index])]
        fold_terms.to_csv(os.path.join(fold_out_dir, f'terms.tsv'), sep='\t', index=False)
        vec_test_protein_ids = fold_terms['EntryID'].values
        metric_gt = gt_parser(os.path.join(fold_out_dir, f'terms.tsv'), ontologies)

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

        tkep = tqdm(range(1, CFG["epochs"] + 1), desc="epoch", leave=True, position=1)
        for epoch in tkep:
            logs = {}
            
            train_one_ep(fold_model, dl, optimizer, criterion, scaler, logs)
            
            if scheduler is not None:
                scheduler.step()
                
            #release GPU memory cache
            torch.cuda.empty_cache()
            gc.collect()
        
            #eval
            if (epoch % eval_every == 0) or epoch == CFG["epochs"]:
                val_probs, val_trues = evaluate(fold_model, dl_val, criterion, logs)
                
                # competition metric - slow
                if (epoch % metric_every == 0) or epoch == CFG["epochs"]:
                    calculate_metric(val_probs, vec_test_protein_ids, metric_gt, labels_to_consider, logs)
                    
                    # plot also prediction histogram at the same intervals # subsample equally
                    table_data = val_probs_to_ontology_columns(val_probs)[::5000]
                    for col_i, col in enumerate(['BPO', 'CCO', 'MFO']):
                        table = wandb.Table(
                            data=table_data[:,col_i], 
                            columns=[col])
                        wandb.log({
                            'pred_histogram' : wandb.plot.histogram(
                                table[:,0],
                                col,
                                title=f"{col} Confidence Scores")})
                
                # Save out-of-fold predictions if this is the last epoch
                if epoch == CFG["epochs"]:
                    np.save(os.path.join(fold_out_dir, f'oof_probs.npy'), val_probs)
                    np.save(os.path.join(fold_out_dir, f'oof_trues.npy'), val_trues)
                    np.save(os.path.join(fold_out_dir,f'oof_ids.npy'), vec_test_protein_ids)

                #release GPU memory cache
                del val_probs, val_trues
                torch.cuda.empty_cache()
                gc.collect()
                
                tkep.set_postfix(
                    train_loss=logs['train_loss'], train_f1=logs['train_f1'], 
                    val_loss=logs['val_loss'],val_f1=logs['val_f1'])
            else:
                tkep.set_postfix(train_loss=logs['train_loss'], train_f1=logs['train_f1'])

            wandb.log(logs, step=epoch)
        
        # Save final fold model
        torch.save(
            fold_model.state_dict(), 
            os.path.join(fold_out_dir, f'model.pth'))
        
        wandb.finish()


if __name__ == "__main__":
    main(**{n:v for n,v in vars(argParser.parse_args()).items()})