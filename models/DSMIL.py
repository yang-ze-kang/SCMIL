import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FCLayer(nn.Module):
    def __init__(self, in_size):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, 1))
    def forward(self, feats, **kwargs):
        x = self.fc(feats)
        indice = torch.argmax(x.squeeze())
        key_feat = feats[indice]
        return key_feat.unsqueeze(0)

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x, **kwargs):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, dropout_v=0.0): # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )
        
    def forward(self, feats, key_feat, **kwargs): # N x K, N x C
        device = feats.device
        V = self.v(feats)
        Q = self.q(feats).view(feats.shape[0], -1)
        q_max = self.q(key_feat)
        
        A = torch.mm(Q, q_max.transpose(0, 1))
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)
        B = torch.mm(A.transpose(0, 1), V)
        return B, A
        
class DSMILSurvival(nn.Module):
    
    def __init__(self,input_size=1024, as_backbone=False, dropout_rate=0.1,**kwargs):
        super(DSMILSurvival, self).__init__()
        self.as_backbone = as_backbone
        self.i_classifier = FCLayer(in_size=input_size)
        self.b_classifier = BClassifier(input_size=input_size, dropout_v=dropout_rate)
        
    def forward(self, x, label=None, loss_func=None, with_x_grad=False, with_atten=False,grad_cam_x=False,grad_cam_o=False,**kwargs):
        x = x.squeeze(0)
        assert len(x.shape)==2
        key_feat = self.i_classifier(x)
        feat, A = self.b_classifier(x, key_feat)
        h = 0.5*(key_feat+feat)
        return h
        
        
if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = DSMILSurvival().cuda()
    results_dict = model(data)
    print(results_dict)
