import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils.nystrom_attention import NystromAttention
import pdb
import math
from einops import reduce


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, scale=None, return_attn=False):
        if return_attn:
            h,atten = self.attn(self.norm(x), scale = scale, return_attn=True)
            x = x + h
            return x,atten
        else:
            return x + self.attn(self.norm(x), scale = scale),None


class PPEG(nn.Module):
    def __init__(self, dim=512, cls_token=True):
        super(PPEG, self).__init__()
        self.cls_token = cls_token
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        if self.cls_token:
            cls_token, feat_token = x[:, 0], x[:, 1:]
        else:
            feat_token=x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token:
            x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes=2,cls_method='cls_token', with_ppeg=True, dynamic_scale=False, input_size=1024, as_backbone=False):
        super(TransMIL, self).__init__()
        if input_size>=1024:
            hidden_size=512
        else:
            hidden_size = input_size
        self.cls_mehod = cls_method
        if with_ppeg:
            self.pos_layer = PPEG(dim=hidden_size,cls_token=cls_method=='cls_token')
        self._fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        if cls_method:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.n_classes = n_classes
        self.with_ppeg = with_ppeg
        self.dynamic_scale = dynamic_scale
        self.as_backbone = as_backbone
        
        self.layer1 = TransLayer(dim=hidden_size)
        self.layer2 = TransLayer(dim=hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        if not as_backbone:
            self._fc2 = nn.Linear(hidden_size, self.n_classes)


    def forward(self, x, label=None, loss_func=None, with_x_grad=False, with_atten=False,grad_cam_x=False,grad_cam_o=False,**kwargs):
        if len(x.shape)==2:
            x = x.unsqueeze(0)
        
        hidd = self._fc1(x) #[B, n, 512]
        
        if with_x_grad or grad_cam_x:
            hidd.retain_grad()
        h = hidd
        
        #---->pad
        H = h.shape[1]

        if self.dynamic_scale:
            # scale = math.log(H)/math.log(512) * (64 ** -0.5)
            scale = math.log(H) * (64 ** -0.5)
        else:
            scale = None
        
        if self.with_ppeg:
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        if self.cls_mehod=='cls_token':
            cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
            h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h,att1 = self.layer1(h,scale=scale,return_attn=with_atten) #[B, N, 512]
        if att1:
            att1 = att1[:,:,0]

        #---->PPEG
        if self.with_ppeg:
            h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h,att2 = self.layer2(h,scale=scale) #[B, N, 512]
        
        if att2:
            att2=att2[:,:,0]

        #---->cls_token
        o = self.norm(h)
        if grad_cam_o:
            o.retain_grad()
        
        if self.cls_mehod=='cls_token':
            h = o[:,0]
        elif self.cls_mehod=='mean':
            h = reduce(o,'b h c->b c','mean')
        elif self.cls_mehod=='max':
            h = reduce(o,'b h c->b c','max')
        
        if self.as_backbone:
            return h
        
        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'y_prob': Y_prob, 'y_hat': Y_hat}
        
        if  with_x_grad:
            loss = loss_func(logits,label.long())
            loss.backward()
            x_grad = hidd.grad @ self._fc1[0].weight
            x_grad = x_grad.detach().cpu()
            results_dict.update({
                'x_grad':x_grad
            })
        
        if grad_cam_x:
            loss = loss_func(logits,label.long())
            loss.backward()
            x_grad = hidd.grad.detach()
            w = torch.mean(x_grad,dim=1)
            cam = hidd.detach() @ w.T
            results_dict.update({
                'grad_cam_x':cam.squeeze(-1).detach().cpu()
            })
            
        if grad_cam_o:
            start = 0
            if self.cls_mehod=='cls_token':
                start=1
            loss = loss_func(logits,label.long())
            loss.backward()
            o_grad = o.grad.detach()
            w = torch.mean(o_grad,dim=1)
            cam = o.detach() @ w.T
            results_dict.update({
                'grad_cam_o':cam.squeeze(-1)[:,start:H+st].detach().cpu()
            })
        
        if with_atten:
            results_dict.update({
                'att1':att1,
                'att2':att2
            })

        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data)
    print(results_dict)
