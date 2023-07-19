import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
import gc
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# load files
train_terms = pd.read_csv("./input/cafa-5-protein-function-prediction/Train/train_terms.tsv",sep="\t")
train_terms_updated = None # late init
labels = None # late init
term_aspect_map = train_terms.set_index('term')['aspect'].to_dict()
train_protein_ids = np.load('./input/t5embeds/train_ids.npy')
train_embeddings = np.load('./input/t5embeds/train_embeds.npy')
train_df = pd.DataFrame(train_embeddings, columns = ["Column_" + str(i) for i in range(1, train_embeddings.shape[1]+1)])

if os.path.exists('./input/train_sequence_clusters.csv'):
    train_sequence_clusters_df = pd.read_csv('./input/train_sequence_clusters.csv')
else:
    print('cannot find ./input/train_sequence_clusters.csv - continuing without it')
    train_sequence_clusters_df = None

# Set the limit for label 
num_of_labels = 1500
labels_to_consider = train_terms['term'].value_counts().index[:num_of_labels].tolist()
labels_df = None
types_df = None

N_SPLITS = 5
RND_SEED = 2023

device = 'cuda'
eval_every_n = 1
calculate_metric_every = 5

def calculate_class_weights(dl, max_weight=20):
    """ Inverse frequency weights to labels clipped to max_weight """
    return np.clip(
            1. / np.concatenate([data['y'].numpy() for data in dl]).mean(axis=0),
            1., max_weight)


def create_labels_df():
    # Create an empty dataframe of required size for storing the labels,
    # i.e, train_size x num_of_labels (142246 x 1500)
    train_size = train_protein_ids.shape[0] # len(X)
    train_labels = np.zeros((train_size ,num_of_labels))

    # Convert from numpy to pandas series for better handling
    series_train_protein_ids = pd.Series(train_protein_ids)

    # Loop through each label
    for i in tqdm(range(num_of_labels), total=num_of_labels):
        # For each label, fetch the corresponding train_terms data
        n_train_terms = train_terms_updated[train_terms_updated['term'] ==  labels[i]]
        
        # Fetch all the unique EntryId aka proteins related to the current label(GO term ID)
        label_related_proteins = n_train_terms['EntryID'].unique()
        
        # In the series_train_protein_ids pandas series, if a protein is related
        # to the current label, then mark it as 1, else 0.
        # Replace the ith column of train_Y with with that pandas series.
        train_labels[:,i] =  series_train_protein_ids.isin(label_related_proteins).astype(float)

    # Convert train_Y numpy into pandas dataframe
    return pd.DataFrame(data = train_labels, columns = labels)


def create_types_df():
    train_size = train_protein_ids.shape[0] # len(X)
    train_types = np.zeros((train_size ,3))

    # Convert from numpy to pandas series for better handling
    series_train_protein_ids = pd.Series(train_protein_ids)

    # Loop through each type
    type_labels = ['BPO', 'CCO', 'MFO']
    for i, t in enumerate(type_labels):
        # For each label, fetch the corresponding train_terms data
        n_train_terms = train_terms_updated[train_terms_updated['aspect'] ==  t]
        
        # Fetch all the unique EntryId aka proteins related to the current label(GO term ID)
        type_related_proteins = n_train_terms['EntryID'].unique()
        
        # In the series_train_protein_ids pandas series, if a protein is related
        # to the current label, then mark it as 1, else 0.
        # Replace the ith column of train_Y with with that pandas series.
        train_types[:,i] =  series_train_protein_ids.isin(type_related_proteins).astype(float)

    # Convert train_Y numpy into pandas dataframe
    types_df = pd.DataFrame(data = train_types, columns = type_labels)

    # create maps for binary to column name
    map_bpo = {0: '', 1: 'BPO'}
    map_cco = {0: '', 1: 'CCO'}
    map_mfo = {0: '', 1: 'MFO'}

    # replace binary with column name and concatenate
    types_df['combined'] = types_df['BPO'].map(map_bpo) + ',' + types_df['CCO'].map(map_cco) + ',' + types_df['MFO'].map(map_mfo)

    # remove unnecessary leading, trailing, and multiple separators
    types_df['combined'] = types_df['combined'].str.strip(',').str.replace(',+', ',', regex=True)
    return types_df


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


def val_probs_to_ontology_columns(val_probs):
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


def prepare_dataset():
    global num_of_labels, labels_to_consider, labels_df, types_df, train_terms_updated, labels
    num_of_labels = num_of_labels

    # Take value counts in descending order and fetch first 1500 `GO term ID` as labels
    labels = train_terms['term'].value_counts().index[:num_of_labels].tolist()
    # Fetch the train_terms data for the relevant labels only
    train_terms_updated = train_terms.loc[train_terms['term'].isin(labels)]

    labels_df_fn = f'./output/t5_train_labels_num_lbl-{num_of_labels}.csv'
    if os.path.exists(labels_df_fn):
        labels_df = pd.read_csv(labels_df_fn)
    else:
        labels_df = create_labels_df()
        labels_df.to_csv(labels_df_fn, index=False)
    
    types_df_fn = f'./output/train_types_in_num_lbl-{num_of_labels}.csv'
    if os.path.exists(types_df_fn):
        types_df = pd.read_csv(types_df_fn)
    else:
        types_df = create_types_df()
        types_df.to_csv(types_df_fn, index=False)


prepare_dataset()

