import os
import numpy as np
import pandas as pd
import wandb
import json
from tqdm.auto import tqdm
import subprocess

from cafa_utils import train_terms
from metric import ontologies, gt_parser, pred_parser, evaluate_prediction, tau_arr

def evaluate_all_folds(out_dir, train_terms, labels_to_consider, CFG=None):
    """ Evaluates metric for output folder containing five fold dirs and oof probs """
    train_terms.to_csv(os.path.join(out_dir, f'terms.tsv'), sep='\t', index=False)
    metric_gt = gt_parser(os.path.join(out_dir, f'terms.tsv'), ontologies)
    metric_preds = []
    logs = {}
    
    n_folds = np.sum([1 for d in os.listdir(out_dir) if str(d).startswith('fold-')])
    
    for fold in range(n_folds):
        val_probs = np.load(os.path.join(out_dir, f'fold-{fold}', f'oof_probs.npy')).astype(np.float16)
        vec_test_protein_ids = np.load(os.path.join(out_dir, f'fold-{fold}',f'oof_ids.npy'),
                                      allow_pickle=True).astype(str)
        
        indices = np.where(val_probs > 0.01)
        
        tk2 = tqdm(
            zip(*indices), total=len(indices[0]), 
            desc=f"collecting metric - fold-{fold}", leave=False, position=0)
        
        metric_preds += [
            (
                vec_test_protein_ids[i],
                labels_to_consider[j],
                val_probs[i, j]
            ) for i, j in tk2
        ]
        
    metric_preds = pred_parser(metric_preds, ontologies, metric_gt, prop_mode='fill', max_terms=500)
    df_metrics = evaluate_prediction(
        metric_preds, metric_gt, 
        ontologies, tau_arr, n_cpu=4)
    
    df_metrics.to_csv(os.path.join(out_dir, "df_metric.csv"), index=False)
    metric_dict = df_metrics.groupby('ns').agg({'wf':'max'}).to_dict()['wf']
    metric_bio = metric_dict['biological_process'] if 'biological_process' in metric_dict else 0
    metric_cell = metric_dict['cellular_component'] if 'cellular_component' in metric_dict else 0
    metric_mol = metric_dict['molecular_function'] if 'molecular_function' in metric_dict else 0
    metric_mean = np.mean([metric_bio, metric_cell, metric_mol])
    
    logs['biological_process_CV-OOF'] = metric_bio
    logs['cellular_component_CV-OOF'] = metric_cell
    logs['molecular_function_CV-OOF'] = metric_mol
    logs['ontology_avg_CV-OOF'] = metric_mean
    
    # log w&b
    group_id = out_dir.split('/')[-1]
    wandb.init(project='cafa-5', group=group_id, dir=out_dir, config=CFG)
    wandb.log(logs)
    wandb.finish()
    
    # write to output dir
    with open(os.path.join(out_dir, "metric.json"), "w") as outfile:
        json.dump(logs, outfile)
    
    return logs


def calculate_metric(probs, vec_test_protein_ids, metric_gt, labels_to_consider, logs):
    """ Calculates the competition metric scores """
    
    indices = np.where(probs > 0.01)
        
    tk2 = tqdm(
        zip(*indices), total=len(indices[0]), 
        desc=f"collecting metric", leave=False, position=0)
    
    metric_preds = [
        (
            vec_test_protein_ids[i],
            labels_to_consider[j],
            probs[i, j]
        ) for i, j in tk2
    ]

    metric_preds = pred_parser(metric_preds, ontologies, metric_gt, prop_mode='fill', max_terms=500)
    df_metrics = evaluate_prediction(
        metric_preds, metric_gt, 
        ontologies, tau_arr, n_cpu=4)
    metric_dict = df_metrics.groupby('ns').agg({'wf':'max'}).to_dict()['wf']
    metric_bio = metric_dict['biological_process'] if 'biological_process' in metric_dict else 0
    metric_cell = metric_dict['cellular_component'] if 'cellular_component' in metric_dict else 0
    metric_mol = metric_dict['molecular_function'] if 'molecular_function' in metric_dict else 0
    metric_mean = np.mean([metric_bio, metric_cell, metric_mol])
    
    logs['biological_process'] = metric_bio
    logs['cellular_component'] = metric_cell
    logs['molecular_function'] = metric_mol
    logs['ontology_avg'] = metric_mean
    

def evaluate_out_dir(out_dir):
    """ Evaluates metric for output folder containing five fold dirs and oof probs """
    gt_file = os.path.join(out_dir, f'terms.tsv')
    if not os.path.exists(gt_file):
        train_terms.to_csv(gt_file, sep='\t', index=False)
    
    pred_dir = os.path.join(out_dir, 'oof_preds')
    if not os.path.exists(pred_dir): os.mkdir(pred_dir)

    res_dir = os.path.join(out_dir, 'metric')
    if not os.path.exists(res_dir): os.mkdir(res_dir)

    pred_fn = os.path.join(pred_dir, 'predictions.txt')
    if not os.path.exists(pred_fn):

        labels_to_consider = []
        
        n_folds = np.sum([1 for d in os.listdir(out_dir) if str(d).startswith('fold-')])
        with open(pred_fn, 'w') as pred_f:
            for fold in range(n_folds):
                val_probs = np.load(os.path.join(out_dir, f'fold-{fold}', f'oof_probs.npy')).astype(np.float16)
                labels_fn = os.path.join(out_dir, f'fold-{fold}', f'labels_to_consider.txt')
                if len(labels_to_consider) == 0 and os.path.exists(labels_fn):
                        with open(labels_fn, "r") as f:
                            for line in f:
                                labels_to_consider.append(line.strip())
                
                vec_test_protein_ids = np.load(os.path.join(out_dir, f'fold-{fold}',f'oof_ids.npy'),
                                            allow_pickle=True).astype(str)
                
                indices = np.where(val_probs > 0.01)
                
                tk2 = tqdm(
                    zip(*indices), total=len(indices[0]), 
                    desc=f"collecting metric - fold-{fold}", leave=False, position=0)
                
                for i,j in tk2:
                    pred_f.write(f'{vec_test_protein_ids[i]} {labels_to_consider[j]} {val_probs[i, j]}\n')
        
    print('Starting CAFA evaluator')
    # python3 main.py go-basic.obo predition_dir/ test_terms.tsv -out_dir results/ -ia IA.txt -prop fill -norm cafa -th_step 0.001 -max_terms 500
    PYTHON_PATH = subprocess.check_output("which python", shell=True).strip().decode('utf-8')
    OBO_PATH = 'input/cafa-5-protein-function-prediction/Train/go-basic.obo'
    IA_PATH = 'input/cafa-5-protein-function-prediction/IA.txt'
    cmd = [PYTHON_PATH, 
           "main.py", 
           os.path.abspath(OBO_PATH), 
           os.path.abspath(pred_dir),
           os.path.abspath(gt_file),
           '-out_dir', os.path.abspath(res_dir),
           "-ia", os.path.abspath(IA_PATH),
           '-prop', 'fill',
           '-norm', 'cafa', 
           '-th_step', '0.001', 
           '-max_terms', '500'
    ]
    process = subprocess.Popen(cmd, cwd="CAFA-evaluator/src/")
    _, _ = process.communicate() # this blocks until process completes

if __name__ == "__main__":
    #pass
    evaluate_out_dir(
        out_dir='./output/20230720-171629/')