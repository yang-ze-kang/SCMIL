from torch import nn
import torch

class SoftFilterLayer(nn.Module):
    
    def __init__(self,dim,hidden_size,deep=1) -> None:
        super().__init__()
        layers = []
        for i in range(deep):
            layers.append(nn.Linear(dim,hidden_size))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_size,1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
            
    def forward(self,x):
        logits = self.layers(x)
        h = torch.mul(x,logits)
        return h,logits
    
class SoftFilterLayerSNN(nn.Module):
    
    def __init__(self,dim,hidden_size,deep=1) -> None:
        super().__init__()
        layers = []
        for i in range(deep):
            layers.append(nn.Linear(dim,hidden_size,bias=False))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_size,1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
            
    def forward(self,x):
        h = self.layers(x)
        h = torch.mul(x,h)
        return h