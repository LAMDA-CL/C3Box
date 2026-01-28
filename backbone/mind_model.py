import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class VisionClassifier(nn.Module):
    def __init__(self, in_features, num_classes,args, weight_init=None,  activation=None):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=False)
        self.device = args["device"][0]
        self.fc = nn.Parameter(self.fc.weight.data.to(self.device))

  #      self.fc = nn.Parameter(self.fc.weight.data)
        
        if weight_init is not None:
            self.fc.data = weight_init
        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.Identity()
    
    def add_weight(self, weight):
        # weight = weight.to(self.device)
        # device = self.fc.device 
        weight = weight.to(self.device)
        self.fc = nn.Parameter(torch.cat([self.fc, weight], dim=0))

    def set_weight(self, weight):
        # weight = weight.to(self.device)
     #   device = self.fc.device 
        weight = weight.to(self.device)

        self.fc = nn.Parameter(weight)


    def forward(self, x):
        # normalize the weights

        x = F.normalize(x, p=2, dim=-1)
        weight = F.normalize(self.fc, p=2, dim=-1)
        x = F.linear(x, weight)
        x = self.activation(x)
        return x