import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_shape, num_of_labels, n_hidden:int=512, activation=nn.Mish()):
        super(Model, self).__init__()
        
        self.batch_norm = nn.BatchNorm1d(input_shape)
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.dense1 = nn.Linear(input_shape, n_hidden)
        self.dense2 = nn.Linear(n_hidden, n_hidden)
        self.dense3 = nn.Linear(n_hidden, n_hidden)
        
        self.dropout2 = nn.Dropout(p=0.1)
        
        self.out = nn.Linear(n_hidden, num_of_labels)
        self.activation = activation

    def forward(self, x):
        
        x = self.batch_norm(x)
        x = self.dropout1(x)
        
        # n_feats 
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.activation(self.dense3(x))
        
        x = self.dropout2(x)
        
        x = self.out(x)
        
        return x