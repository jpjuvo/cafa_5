import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class EmbeddingDataset(Dataset):
    def __init__(self, x, y, device='cpu'):
        self.x = x.to(device)
        self.y = y.to(device) if y is not None else None
        self.device = device
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]
    

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).

    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
    

def get_dataloaders(train_index, test_index, train_df, labels_df, 
                    batch_size:int, num_workers:int=4, fast_dataloader=True):
    """ Returns train and val dataloaders """

    # Generate the training data for this split
    X_train = torch.tensor(train_df.iloc[train_index].values.astype(np.float32))
    y_train = torch.tensor(labels_df.iloc[train_index].values.astype(np.float32))
    
    # Generate the validation data for this split
    X_val = torch.tensor(train_df.iloc[test_index].values.astype(np.float32))
    y_val = torch.tensor(labels_df.iloc[test_index].values.astype(np.float32))
    
    if fast_dataloader:
        dl = FastTensorDataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
        dl_val = FastTensorDataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)

    else:
        dataset = EmbeddingDataset(X_train, y_train, device='cpu')
        dataset_val = EmbeddingDataset(X_val, y_val, device='cpu')

        dl = DataLoader(
            dataset,
            sampler = RandomSampler(dataset),
            batch_size  = batch_size,
            drop_last   = True,
            num_workers = num_workers,
            pin_memory  = True,
            worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        )
        
        dl_val = DataLoader(
            dataset_val,
            sampler = SequentialSampler(dataset_val),
            batch_size  = batch_size,
            drop_last   = False,
            num_workers = num_workers,
            pin_memory  = True,
            worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        )
    
    return dl, dl_val

def get_test_dl(test_df, batch_size:int, num_workers:int=4, fast_dataloader=True):
    # Generate the validation data for this split
    X_test = torch.tensor(test_df.values.astype(np.float32))

    if fast_dataloader:
        dl_test = FastTensorDataLoader(X_test, batch_size=batch_size, shuffle=False)

    else:
        dataset_test = EmbeddingDataset(X_test, y=None, device='cpu')
        
        dl_test = DataLoader(
            dataset_test,
            sampler = SequentialSampler(dataset_test),
            batch_size  = batch_size,
            drop_last   = False,
            num_workers = num_workers,
            pin_memory  = True,
            worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        )
    
    return dl_test