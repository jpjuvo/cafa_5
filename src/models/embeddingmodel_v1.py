import torch
import torch.nn as nn

class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

class Model(nn.Module):
    def __init__(self, input_shape, num_of_labels, n_hidden:int=512, dropout1_p:float=0.25,
                use_norm:bool=True, activation=nn.Mish(), use_residual=False):
        super(Model, self).__init__()
        
        self.batch_norm = nn.BatchNorm1d(input_shape)
        self.dropout1 = nn.Dropout(p=dropout1_p)
        
        self.dense1 = nn.Linear(input_shape, n_hidden)
        
        self.dense2 = nn.Linear(n_hidden, n_hidden)
        if use_residual: self.dense2 = Residual(self.dense2)

        self.dense3 = nn.Linear(n_hidden, n_hidden)
        if use_residual: self.dense3 = Residual(self.dense3)
        
        self.dropout2 = nn.Dropout(p=0.1)
        
        self.out = nn.Linear(n_hidden, num_of_labels)
        self.activation = activation
        self.use_norm = use_norm

    def forward(self, x):
        
        if self.use_norm:
            x = self.batch_norm(x)
        x = self.dropout1(x)
        
        # n_feats 
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.activation(self.dense3(x))
        
        x = self.dropout2(x)
        
        x = self.out(x)
        
        return x