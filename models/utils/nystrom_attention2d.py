import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import nn, einsum
import torch.nn.functional as F
from kmeans import minkowski_similarity
from cuml import KMeans
from einops import rearrange, reduce
import numpy as np
import pdb
import copy

# helper functions

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class AMIL_layer(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, activation=None):
        super(AMIL_layer, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, 1)
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        a = self.attention_a(x)
        b = self.attention_b(x)
        attention = a.mul(b)
        attention = self.attention_c(attention).squeeze(dim=2)
        return attention

# main attention class
class NystromAttention2D(nn.Module):
    def __init__(
        self,
        dim,
        pool_method='mean',
        dim_head = 64,
        heads = 8,
        window_h = 16,
        window_w = 16,
        kmeans_max_iter = 5,
        with_feature = False,
        residual = True,
        pinv_iterations = 6,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        
        self.window_h = window_h
        self.window_w = window_w
        self.num_landmarks = window_h * window_w
        
        self.kmeans_max_iter = kmeans_max_iter
        self.with_feature = with_feature
        
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.pool_method = pool_method
        if pool_method=='amil':
            self.amil = AMIL_layer(dim,dim//4)
        
        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)
    
    def pool(self,q,window_sizes,pool_method='mean',attention=None):
        if attention==None:
            attention=torch.ones((q.shape[0],q.shape[2]))
        q_landmarks = []
        q_split = torch.split(q,window_sizes,dim=2)
        a_split = torch.split(attention,window_sizes,dim=1)
        for sp,att in zip(q_split,a_split):
            if pool_method=='mean':
                landmark_einops_eq = 'b h n d -> b h 1 d'
                sp = reduce(sp,landmark_einops_eq,'mean')
            elif pool_method=='amil':
                att = torch.softmax(att,dim=1)
                sp = att@sp
                sp.squeeze(2)
            q_landmarks.append(sp)
        q_landmarks = torch.cat(q_landmarks,dim=2)
        return q_landmarks

    def forward(self, x, coords, coord_weight=1, mask = None, return_attn = False, scale = None):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        
        if x.shape[1]==coords.shape[0]+1:
            with_cls_token = True
        elif x.shape[1]==coords.shape[0]:
            with_cls_token = False
        else:
            raise NotImplementedError
        
        
        with torch.no_grad():
            quo = coords.shape[0]//m
            rem = coords.shape[0]%m
            cluster_size = [m]*quo
            cluster_size.append(rem)
            n_cluster = quo+1
            n_cluster=np.clip(n_cluster,16,64)
            kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=self.kmeans_max_iter)
            if self.with_feature:
                coords = (coords-torch.mean(coords))/torch.std(coords)
                feats = x.squeeze()
                if with_cls_token:
                    feats = feats[:-1]
                feats = feats*(1-coord_weight)
                coords = coords*coord_weight
                coords = torch.cat([feats,coords],dim=1)
            labels = kmeans.fit_predict(coords)
            window_sizes = np.bincount(labels).tolist()
            index = np.argsort(labels).tolist()
            index_ori = np.argsort(index).tolist()
            if with_cls_token:
                window_sizes.append(1)
                index.append(x.shape[1]-1)
                index_ori.append(x.shape[1]-1)
        x = x[:,index]

        if self.pool_method=='amil':
            attention = self.amil(x)
        else:
            attention = None

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))


        # generate landmarks by sum reduction, and then calculate mean using the mask
        q_landmarks = self.pool(q,window_sizes,self.pool_method,attention)
        k_landmarks = self.pool(k,window_sizes,self.pool_method,attention)


        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)


        # eq (15) in the paper and aggregate values
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]
        out = out[:,index_ori]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out,n_cluster

# transformer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Nystromformer2D(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        window_h = 16,
        window_w = 16,
        kmeans_max_iter = 5,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention2D(dim = dim, dim_head = dim_head, heads = heads, window_h=window_h, window_w=window_w, kmeans_max_iter=kmeans_max_iter, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x
