import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

# load files
train_terms = pd.read_csv("./input/cafa-5-protein-function-prediction/Train/train_terms.tsv",sep="\t")
term_aspect_map = train_terms.set_index('term')['aspect'].to_dict()

if os.path.exists('./input/train_sequence_clusters.csv'):
    train_sequence_clusters_df = pd.read_csv('./input/train_sequence_clusters.csv')
else:
    print('cannot find ./input/train_sequence_clusters.csv - continuing without it')
    train_sequence_clusters_df = None


def fold_ensemble_submission(out_dir, test_protein_ids):
    n_folds = np.sum([1 for d in os.listdir(out_dir) if str(d).startswith('fold-')])
    predictions = []
    labels = []

    for fold in range(n_folds):
        test_fn = os.path.join(out_dir, f'fold-{fold}', f'test_preds.npy')
        labels_fn = os.path.join(out_dir, f'fold-{fold}', f'labels_to_consider.txt')
        if os.path.exists(test_fn):
            predictions.append(np.load(test_fn))
        if len(labels) == 0 and os.path.exists(labels_fn):
            with open(labels_fn, "r") as f:
                for line in f:
                    labels.append(line.strip())


    predictions = np.stack(predictions)
    predictions = np.mean(predictions, axis=0)

    df_submission = pd.DataFrame(columns = ['Protein Id', 'GO Term Id','Prediction'])
    l = []
    for k in list(test_protein_ids):
        l += [ k] * predictions.shape[1]   

    df_submission['Protein Id'] = l
    df_submission['GO Term Id'] = labels * predictions.shape[0]
    df_submission['Prediction'] = predictions.ravel()
    df_submission.to_csv(os.path.join(out_dir, "submission.tsv"), header=False, index=False, sep="\t")


def calculate_class_weights(dl, max_weight=20):
    """ Inverse frequency weights to labels clipped to max_weight """
    return np.clip(
            1. / np.concatenate([data['y'].numpy() for data in dl]).mean(axis=0),
            1., max_weight)


def create_labels_df(train_protein_ids, train_terms_updated, num_of_labels, labels_to_consider):
    # Create an empty dataframe of required size for storing the labels,
    # i.e, train_size x num_of_labels (142246 x 1500)
    train_size = train_protein_ids.shape[0] # len(X)
    train_labels = np.zeros((train_size ,num_of_labels))

    # Convert from numpy to pandas series for better handling
    series_train_protein_ids = pd.Series(train_protein_ids)

    # Loop through each label
    for i in tqdm(range(num_of_labels), total=num_of_labels):
        # For each label, fetch the corresponding train_terms data
        n_train_terms = train_terms_updated[train_terms_updated['term'] ==  labels_to_consider[i]]
        
        # Fetch all the unique EntryId aka proteins related to the current label(GO term ID)
        label_related_proteins = n_train_terms['EntryID'].unique()
        
        # In the series_train_protein_ids pandas series, if a protein is related
        # to the current label, then mark it as 1, else 0.
        # Replace the ith column of train_Y with with that pandas series.
        train_labels[:,i] =  series_train_protein_ids.isin(label_related_proteins).astype(float)

    # Convert train_Y numpy into pandas dataframe
    return pd.DataFrame(data = train_labels, columns = labels_to_consider)


def create_out_dir(name=None, output_root='./output/'):
    """
    Creates a new output directory under the given root directory. The new directory is named with the current timestamp, 
    and an optional specific name can also be appended.

    If the root directory doesn't exist, it gets created.

    Args:
        name (str, optional): Specific name to append to the directory name. Defaults to None.
        output_root (str, optional): The root directory where the new directory will be created. Defaults to '../output/'.

    Returns:
        str: The full path of the created directory.

    Raises:
        OSError: If there is an issue creating the directories.
    """
    if not os.path.exists(output_root): os.mkdir(output_root)
    dirname = datetime.now().strftime("%Y%m%d-%H%M%S")
    if name is not None: dirname += f'-{name}'
    dirname = os.path.join(output_root, dirname)
    if not os.path.exists(dirname): os.mkdir(dirname)
    return dirname


def val_probs_to_ontology_columns(val_probs, labels_to_consider):
    """ 
    Maps classes to three different ontolygy columns 
    If input is n x 1500, the output is m x 3 where m > n
    
    This subsamples the set to include similar amount of probs from each three column
    """
    bpo_indices = np.array([i for i,l in enumerate(labels_to_consider) if term_aspect_map[l] == 'BPO'])
    cco_indices = np.array([i for i,l in enumerate(labels_to_consider) if term_aspect_map[l] == 'CCO'])
    mfo_indices = np.array([i for i,l in enumerate(labels_to_consider) if term_aspect_map[l] == 'MFO'])
    bpo_vals = val_probs[:,bpo_indices].reshape(-1,1)
    cco_vals = val_probs[:,cco_indices].reshape(-1,1)
    mfo_vals = val_probs[:,mfo_indices].reshape(-1,1)
    min_len = min([bpo_vals.shape[0], cco_vals.shape[0], mfo_vals.shape[0]])
    
    return np.concatenate([bpo_vals[:min_len,:], cco_vals[:min_len,:], mfo_vals[:min_len,:]], -1)


def prepare_dataframes(n_labels:int=1500, emb_type:str='t5'):
    """ Preload datasets into memory """
    assert emb_type in ['t5', 'esm2_3b'], 'only t5 and esm2_3b emb_types are supported'

    if emb_type == 't5':
        train_protein_ids = np.load('./input/t5embeds/train_ids.npy')
        train_embeddings = np.load('./input/t5embeds/train_embeds.npy')
        test_protein_ids = np.load('./input/t5embeds/test_ids.npy')
        test_embeddings = np.load('./input/t5embeds/test_embeds.npy')
    elif emb_type == 'esm2_3b':
        train_protein_ids = np.load('./input/esm23b/train_ids_esm2_t36_3B_UR50D.npy')
        train_embeddings = np.load('./input/esm23b/train_embeds_esm2_t36_3B_UR50D.npy')
        test_protein_ids = np.load('./input/esm23b/test_ids_esm2_t36_3B_UR50D.npy')
        test_embeddings = np.load('./input/esm23b/test_embeds_esm2_t36_3B_UR50D.npy')
    
    train_df = pd.DataFrame(train_embeddings, columns = ["Column_" + str(i) for i in range(1, train_embeddings.shape[1]+1)])
    test_df = pd.DataFrame(test_embeddings, columns = ["Column_" + str(i) for i in range(1, test_embeddings.shape[1]+1)])
    labels_to_consider = train_terms['term'].value_counts().index[:n_labels].tolist()

    print('Reading data and preparing stuff...')
    # Take value counts in descending order and fetch first 1500 `GO term ID` as labels
    labels_to_consider = train_terms['term'].value_counts().index[:n_labels].tolist()
    # Fetch the train_terms data for the relevant labels only
    train_terms_updated = train_terms.loc[train_terms['term'].isin(labels_to_consider)]

    labels_df_fn = f'./output/t5_train_labels_num_lbl-{n_labels}.csv'
    if os.path.exists(labels_df_fn):
        labels_df = pd.read_csv(labels_df_fn)
    else:
        labels_df = create_labels_df(train_protein_ids, train_terms_updated, n_labels, labels_to_consider)
        labels_df.to_csv(labels_df_fn, index=False)

    print('Preparations done')

    return train_terms_updated, train_protein_ids, test_protein_ids, train_df, test_df, labels_to_consider, labels_df




