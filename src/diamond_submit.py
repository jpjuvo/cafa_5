import numpy as np
import pandas as pd
import os
import json
from tqdm.auto import tqdm

diamond_dir = './output/diamond'
test_protein_ids = np.load('./input/t5embeds/test_ids.npy')

diamond_netgo_df = pd.read_csv(diamond_dir + '/test_diamond_netqo.res', sep='\t', names=['qsegid', 'sseqid', 'bitscore', 'pident', 'evalue'])
diamond_netgo_df['testid'] = diamond_netgo_df['qsegid'].apply(lambda x: str(x).split('\\t')[0])
diamond_netgo_df = diamond_netgo_df.drop(columns='qsegid')
diamond_netgo_df = diamond_netgo_df[diamond_netgo_df.evalue < 0.001]

diamond_df = pd.read_csv(diamond_dir + '/test_diamond_2.res', sep='\t', names=['qsegid', 'sseqid', 'bitscore', 'pident', 'evalue'])
diamond_df['testid'] = diamond_df['qsegid'].apply(lambda x: str(x).split('\\t')[0])
diamond_df = diamond_df.drop(columns='qsegid')
diamond_df = diamond_df[diamond_df.evalue < 0.001]

diamond_df = pd.concat([diamond_df, diamond_netgo_df])
diamond_df['bitscore_max'] = diamond_df['bitscore'] * (1./((diamond_df['pident']/100)))

train_terms = pd.read_csv("./input/cafa-5-protein-function-prediction/Train/train_terms.tsv",sep="\t")
train_terms.set_index('EntryID', inplace=True)

unique_train_ids = train_terms.index.unique()
precalculated_terms = None
#precalc_fn = "./output/precalculated_terms.json"
precalc_fn = "./output/precalculated_terms_netgo.json"


def matches_for_query_id(query_id:str):
    """ Returns a list of array(seq_id, bitscore) matches """
    matches = diamond_df[diamond_df.testid == query_id]
    #matches = diamond_df.loc[query_id]

    if len(matches) == 0: return []
    
    return list(matches[['sseqid', 'bitscore', 'bitscore_max']].values.tolist())


def terms_for_train_id(train_id:str):
    #train_id_terms = train_terms[train_terms.EntryID == train_id]
    train_id_terms = train_terms.loc[train_id]
    return {
        'BPO' : train_id_terms[train_id_terms.aspect == 'BPO']['term'].values.tolist(),
        'CCO' : train_id_terms[train_id_terms.aspect == 'CCO']['term'].values.tolist(),
        'MFO' : train_id_terms[train_id_terms.aspect == 'MFO']['term'].values.tolist(),
    }


def calculate_diamond_scores(query_id:str):
    matches = matches_for_query_id(query_id)
    bitscores = [float(bitscore) for _,bitscore,_ in matches]
    bitscore_maxes = [float(bitscore_max) for _,_,bitscore_max in matches]
    terms_dicts = [precalculated_terms[train_id] for train_id,_,_ in matches]
    found_annos = []
    
    for ont in ['BPO', 'CCO', 'MFO']:
        new_terms = {}
        all_scores = 0
        for term_d, bitscore, bitscore_max in zip(terms_dicts, bitscores, bitscore_maxes):
            if len(term_d[ont]) > 0:
                all_scores += bitscore_max
            for term in term_d[ont]:
                pred_score = bitscore
                if term in new_terms:
                    pred_score += new_terms[term]
                
                new_terms[term] = pred_score
        
        # normalize by all matches that had this ontology annotated
        for term in list(new_terms.keys()):
            score = new_terms[term]/all_scores
            # metric doesn't consider smaller ths
            if score > 0.001:
                found_annos.append((term, score))
    
    return found_annos

def main():
    global precalculated_terms
    
    if os.path.exists(precalc_fn):
        # checkpoint
        with open(precalc_fn, "r") as outfile:
            precalculated_terms = json.load(outfile)
    else:
        precalculated_terms = {
            train_id : terms_for_train_id(train_id) for train_id in tqdm(unique_train_ids, total=len(unique_train_ids), desc='precalc.')
        }

        with open(precalc_fn, "w") as outfile:
            json.dump(precalculated_terms, outfile)
    

    submission_dicts = []

    for test_id in tqdm(test_protein_ids, total=len(test_protein_ids), desc='creating sub'):
        annos = calculate_diamond_scores(test_id)
        submission_dicts += [
            {
                'Protein Id' : test_id,
                'GO Term Id' : anno[0],
                'Prediction' : anno[1]
            } for anno in annos
        ]

    df_submission = pd.DataFrame(submission_dicts)
    df_submission.to_csv("./output/diamond_submission_netgo.tsv",header=False, index=False, sep="\t")


if __name__ == "__main__": 
    main()