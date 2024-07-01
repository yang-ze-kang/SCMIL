import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F
# from kmeans import KMeans
from kmeans import minkowski_similarity
# from sklearn.cluster import KMeans
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

# main attention class
class NystromAttention2D(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        window_h = 16,
        window_w = 16,
        kmeans_max_iter = 5,
        pinv_iterations = 6,
        residual = True,
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
        
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, coords, mask = None, return_attn = False, scale = None):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        with torch.no_grad():
            quo = coords.shape[0]//m
            rem = coords.shape[0]%m
            cluster_size = [m]*quo
            cluster_size.append(rem)
            # kmeans = KMeans(n_clusters=quo+1,cluster_sizes=cluster_size,device=coords.device,max_iter=self.kmeans_max_iter,use_label_soft_value=True,sim_fun=minkowski_similarity(p=2))
            # kmeans.fit(coords)
            # labels = kmeans.labels
            kmeans = KMeans(n_clusters=quo+1, init='k-means++', max_iter=10, random_state=0)
            labels = kmeans.fit_predict(coords)
            window_sizes = np.bincount(labels).tolist()
            # index = torch.argsort(labels)
            # index_ori = torch.argsort(index)
            index = np.argsort(labels).tolist()
            index_ori = np.argsort(index).tolist()
        x = x[:,index]
        # pad so that sequence can be evenly divided into m landmarks

        # remainder = n % m
        # if remainder > 0:
        #     padding = m - (n % m)
        #     x = F.pad(x, (0, 0, padding, 0), value = 0)

        #     if exists(mask):
        #         mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values
        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        if scale is not None:
            q = q * scale
        else:
            q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask
        l = ceil(n / m)
        landmark_einops_eq = 'b h n d -> b h 1 d'
        q_landmarks = []
        # pdb.set_trace()
        q_split = torch.split(q,window_sizes,dim=2)
        for sp in q_split:
            sp = reduce(sp,landmark_einops_eq,'sum')
            q_landmarks.append(sp)
        q_landmarks = torch.cat(q_landmarks,dim=2)
        k_landmarks = []
        k_split = torch.split(k,window_sizes,dim=2)
        for sp in k_split:
            sp = reduce(sp,landmark_einops_eq,'sum')
            k_landmarks.append(sp)
        k_landmarks = torch.cat(k_landmarks,dim=2)
        # landmark_einops_eq = '... (n l) d -> ... n d'
        # q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        # k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean
        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking
        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

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

        return out

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
