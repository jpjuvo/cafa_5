import os
import sys
import gc
import numpy as np
import pandas as pd 
import argparse
from tqdm.auto import tqdm

import torch as th
from torch import nn
from torch.nn import functional as F

sys.path.append('.')
sys.path.append('src')

from dgz_utils import *
from data_utils import FastTensorDataLoader

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--device", default="cuda", help="cuda or cpu")
argParser.add_argument("-o", "--ont", default='bp', type=str, help="ontology - one of mf | bp | cc")
argParser.add_argument("-b", "--batch_size", default=256, type=int, help="batch size for inference")
argParser.add_argument("-m", "--terms_mode", default=1, type=int, help="0=deepgozero_zero_10, 1=deepgozero_zero, 2=deepgozero")
argParser.add_argument("-t", "--threshold", default=0.1, type=float, help="inference th")


# global vars
data_root='./input/deepgozero-data/data'
go_file = f'{data_root}/go.norm'
model_file = None
terms_file = None
sub_file = None
out_file = "./output/DGZ/predictions_deepgozero_zero_10.pkl"
test_df = pd.read_csv('./input/cafa-fasta-4/test_df.csv')

def main(device:str, ont:str, batch_size:int, terms_mode:int, threshold:float):
    global model_file, terms_file, sub_file
    
    if terms_mode == 0:
        model_file = f'{data_root}/{ont}/deepgozero_zero_10.th' # Model file
        terms_file = f'{data_root}/{ont}/terms_zero_10.pkl'
    elif terms_mode == 1:
        model_file = f'{data_root}/{ont}/deepgozero_zero.th' # Model file
        terms_file = f'{data_root}/{ont}/terms_zero.pkl'
    elif terms_mode == 2:
        model_file = f'{data_root}/{ont}/deepgozero.th' # Model file
        terms_file = f'{data_root}/{ont}/terms.pkl'

    sub_file = f'./output/DGZ/{ont}_th-{threshold}_{os.path.basename(model_file)[:-3]}_submission.tsv'

    # Prep dictionary terms
    iprs_dict, terms_dict = load_data(data_root, ont, terms_file)
    n_terms = len(terms_dict)
    n_iprs = len(iprs_dict)
        
    nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(go_file, terms_dict)
    n_rels = len(relations)
    n_zeros = len(zero_classes)

    # model
    net = DGELModel(n_iprs, n_terms, n_zeros, n_rels, device).to(device)
    net.load_state_dict(th.load(model_file, map_location=device))
    net.eval()

    gc.collect()

    # With default terms, test_data is tensor of torch.Size([138417, 26406]), which  is equal to len(test_df) x len(interpro embeds)
    test_data, prot_ids = get_data_test(test_df, iprs_dict,terms_dict)
    test_dl = FastTensorDataLoader(test_data, batch_size=batch_size, shuffle=False)
    go = Ontology(f'{data_root}/go.obo', with_rels=True)

    preds = []
    with th.no_grad():
        for cpu_data in test_dl:
            batch_features = cpu_data[0].to(device)
            logits = net(batch_features)   
            preds = np.append(preds, logits.detach().cpu().numpy())
        
    preds = list(preds.reshape(-1, n_terms))
    with open(sub_file, 'wt') as w:
        # Propagate scores using ontology structure
        # Iterates over each of the score vectors in preds
        # (len(scores)) with default terms and model setup 10101
        for i, scores in tqdm(enumerate(preds), total=len(preds)):
            # Use the index in preds to fetch protein id
            prot_id = prot_ids[i]
            prop_annots = {}
            for go_id, j in terms_dict.items():
                score = scores[j]
                # iterates over the ancestors of a given term in the ontology.
                # If an ancestor term already has a score in prop_annots, 
                # it is updated with the maximum of the current score and the new score. 
                # If it does not have a score, it is assigned the new score.
                for sup_go in go.get_anchestors(go_id):
                    if sup_go in prop_annots:
                        prop_annots[sup_go] = max(prop_annots[sup_go], score)
                    else:
                        prop_annots[sup_go] = score
            # loop over prop_annots.items() 
            # updates the scores in the original scores vector based on the propagated scores.
            for go_id, score in prop_annots.items():
                if go_id in terms_dict:
                    scores[terms_dict[go_id]] = score
            #sort them and go over them, looking at thresholds to write to submission file
            # For default terms_zero_10, len of prop_annots and sannots is 10490. This is a bit more than 10101 score length, probably due to GO hierarchy.
            
            sannots = sorted(prop_annots.items(), key=lambda x: x[1], reverse=True)
            for go_id, score in sannots:
                    if score >= threshold:
                        w.write(prot_id + '\t' + go_id + '\t%.3f\n' % score)
            w.write('\n')


if __name__ == "__main__":
    main(**{n:v for n,v in vars(argParser.parse_args()).items()})