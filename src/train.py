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
from eval import calculate_metric, evaluate_all_folds
from binarymetrics import BinaryMetrics

sys.path.append('.')
sys.path.append('src')

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--config", default="embedding_v1", help="config name without .py extension")
argParser.add_argument("-d", "--device", default="cuda", help="cuda or cpu")
argParser.add_argument("-e", "--eval_every", default=1, type=int, help="how often to evaluate between epochs")
argParser.add_argument("-m", "--metric_every", default=100, type=int, help="how often to evaluate metric between epochs")

# constants
N_SPLITS = 5
RND_SEED = 2023


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


def save_test_preds(fold_model, dl_test, fold_out_dir, device):
    """ Save test set predictions to fold dir """
    fold_model.eval()
    test_probs = []

    tk1 = tqdm(
        dl_test, total=int(len(dl_test)), 
        desc="predicting test set", ascii=True, leave=False, position=2)
    for cpu_data in tk1:
        data = cpu_data[0].to(device)
        with torch.no_grad() and torch.cuda.amp.autocast():
            logits = fold_model(data)
            probs = F.sigmoid(logits).detach()
            test_probs.append(probs.cpu().numpy())

    test_probs = np.concatenate(test_probs, 0)
    np.save(os.path.join(fold_out_dir, 'test_preds.npy'), test_probs)
    with open(os.path.join(fold_out_dir, 'labels_to_consider.txt'), 'w') as fp:
        for lbl in labels_to_consider:
            fp.write("%s\n" % lbl)


def main(config, device:str, eval_every:int, metric_every:int):
    CFG = get_config(config)
    prepare_dataset(
        n_labels=CFG["n_labels"] if 'n_labels' in CFG else 1500,
        emb_type=CFG["emb_type"] if 'emb_type' in CFG else 't5'
        )
    out_dir = create_out_dir()
    group_id = out_dir.split('/')[-1]
    binarymetrics = BinaryMetrics(device=device)
    dl_test = get_test_dl(test_df=test_df, batch_size=CFG['batch_size'])

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
            
            train_one_ep(fold_model, dl, optimizer, 
                         criterion, scaler, binarymetrics, logs, device,
                         log_metrics=(epoch % eval_every == 0) or epoch == CFG["epochs"])
            
            if scheduler is not None:
                scheduler.step()
                
            #release GPU memory cache
            torch.cuda.empty_cache()
            gc.collect()
        
            #eval
            if (epoch % eval_every == 0) or epoch == CFG["epochs"]:
                val_probs, val_trues = evaluate(fold_model, dl_val, criterion, binarymetrics, logs, device)
                
                # competition metric - slow
                if (epoch % metric_every == 0):
                    calculate_metric(val_probs, vec_test_protein_ids, metric_gt, labels_to_consider, logs)
                    
                    # plot also prediction histogram at the same intervals # subsample equally
                    table_data = val_probs_to_ontology_columns(val_probs)[::5000]
                    for col_i, col in enumerate(['BPO', 'CCO', 'MFO']):
                        table = wandb.Table(
                            data=table_data[:,col_i:col_i+1], 
                            columns=[col])
                        wandb.log({
                            'pred_histogram' : wandb.plot.histogram(
                                table,
                                col,
                                title=f"{col} Confidence Scores")})
                
                # Save out-of-fold and test set predictions if this is the last epoch
                if epoch == CFG["epochs"]:
                    np.save(os.path.join(fold_out_dir, f'oof_probs.npy'), val_probs)
                    np.save(os.path.join(fold_out_dir, f'oof_trues.npy'), val_trues)
                    np.save(os.path.join(fold_out_dir,f'oof_ids.npy'), vec_test_protein_ids)
                    save_test_preds(fold_model, dl_test, fold_out_dir, device)

                #release GPU memory cache
                del val_probs, val_trues
                torch.cuda.empty_cache()
                gc.collect()
                
                tkep.set_postfix(
                    train_loss=logs['train_loss'], train_f1=logs['train_f1'], 
                    val_loss=logs['val_loss'],val_f1=logs['val_f1'])
            else:
                tkep.set_postfix(train_loss=logs['train_loss'])

            wandb.log(logs, step=epoch)
        
        # Save final fold model
        torch.save(
            fold_model.state_dict(), 
            os.path.join(fold_out_dir, f'model.pth'))
        
        wandb.finish()

    # create submission file
    fold_ensemble_submission(out_dir)

    # extremely slow
    evaluate_all_folds(
        out_dir=out_dir, 
        train_terms=train_terms, 
        labels_to_consider=labels_to_consider)


if __name__ == "__main__":
    main(**{n:v for n,v in vars(argParser.parse_args()).items()})