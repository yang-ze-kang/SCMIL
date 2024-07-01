import sys
from typing import Any
import torch
import time
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pdb
from PIL import Image
import random
import h5py
import os
import cv2
from collections import Counter

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.2) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam[mask==0] = img[mask==0]
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def cosine_similarity(dim=1):
    return torch.nn.CosineSimilarity(dim=dim)

class minkowski_similarity:
    
    def __init__(self,p=2) -> None:
        self.p = p
        self.func = torch.nn.PairwiseDistance(p)
        
    def __call__(self, x1,x2) -> Any:
        return 1-self.func(x1,x2)


class KMeans:
    def __init__(self, n_clusters=20, cluster_sizes=None,H=None,W=None,
                 max_iter=None, verbose=False,use_label_soft_value=False,
                 device=torch.device("cuda"), sim_fun=cosine_similarity(dim=1)):
        self.device = device
        self.H = H
        self.W = W
        self.n_clusters = n_clusters
        if isinstance(cluster_sizes,int):
            self.cluster_sizes = [cluster_sizes] * n_clusters
        elif isinstance(cluster_sizes,list):
            self.cluster_sizes = cluster_sizes
        elif cluster_sizes is None:
            self.cluster_sizes = None
        else:
            raise NotImplementedError
        self.sim_fun = sim_fun
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.use_label_soft_value = use_label_soft_value

        
        self.centers = None
        self.sims = None
        self.labels = None
        self.label_soft_values = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.count = 0

    def to(self, deive):
        self.device = deive

    def clear(self):
        self.labels = None
        self.sims = None
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(self.device)
        self.representative_samples = None
        self.started = False
        self.count = 0

    def fit(self, x):
        """
        x:(num_samples,dims)
        """
        x[:,0] = (x[:,0]-min(x[:,0]))/(max(x[:,0])-min(x[:,0]))
        x[:,1] = (x[:,1]-min(x[:,1]))/(max(x[:,1])-min(x[:,1]))
        if not torch.is_tensor(x):
            x = torch.tensor(x,dtype=torch.float32).to(self.device)
        self.clear()
        if self.centers is None:
            init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
            self.centers = x[init_row]
        while True:
            self.nearest_center(x)
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmax(self.sims, dim=1))
            if torch.abs(self.variation) < 1e-4:
                break
            if self.max_iter is not None and self.count >= self.max_iter:
                break

            self.count += 1

        self.representative_sample()

    def nearest_center(self, x):
        sims = torch.empty((x.shape[0], 0)).to(self.device)
        for center in self.centers:
            sim = self.sim_fun(center, x)
            sim = sim.to(self.device)
            sims = torch.cat([sims, sim.unsqueeze(1)], dim=1)
        if self.cluster_sizes is None:
            self.labels = torch.argmax(sims, dim=1)
        else:
            self.labels = torch.argmax(sims, dim=1)
            # labels = []s
            # nums = torch.zeros(self.n_clusters)
            # indexs = torch.argsort(sims,dim=1,descending=True)
            # labels = torch.zeros(indexs.shape[0])-1
            # start = 0
            # for i, index in enumerate(indexs):
            #     for j in index[start:]:
            #         if nums[j]<self.cluster_sizes[j]:
            #             nums[j]+=1
            #             labels[i] = j
            #             break
            #         else:
            #             start +=1
            # self.labels = torch.tensor(labels,dtype=torch.int32)
        if self.use_label_soft_value:
            sims_soft = torch.softmax(sims,dim=1)
            self.label_soft_values = sims_soft[torch.arange(sims_soft.shape[0]),self.labels]
        if self.started:
            self.variation = torch.mean(torch.abs(self.sims - sims))
            assert self.variation.isnan()==False
        self.sims = sims
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            if len(cluster_samples) == 0:
                centers = torch.cat(
                    [centers, self.centers[i].unsqueeze(0).to(self.device)], dim=0)
            else:
                centers = torch.cat([centers, torch.mean(
                    cluster_samples, dim=0).unsqueeze(0).to(self.device)], dim=0)
        self.centers = centers


    def representative_sample(self):
        self.representative_samples = torch.argmax(self.sims, (0))
        
if __name__=='__main__':
    # x_range = torch.linspace(1, 160, 160)
    # y_range = torch.linspace(1, 160, 160)
    # # x_range = torch.linspace(1, 12, 12)
    # # y_range = torch.linspace(1, 12, 12)
    # x, y = torch.meshgrid(x_range, y_range)
    # h,w = x.shape
    # data = torch.concat([x.reshape((h*w,1)),y.reshape((h*w,1))],dim=1)
    camelyon_root = '/mnt/disk3/yzk/datasets/CAMELYON16/features/swin-b_freezebackbone/h5_files'
    img_dir = '/mnt/disk3/yzk/datasets/CAMELYON16/seg'
    
    id = 'tumor_051'
    h5_path = os.path.join(camelyon_root,f"{id}.h5")
    window_h,windom_w = 16,16
    windom_l = windom_w*window_h
    with h5py.File(h5_path,'r') as f:
        coords = np.round(np.array(f['coords'])/32)
    
    quo = coords.shape[0]//windom_l
    rem = coords.shape[0]%windom_l
    cluster_size = [windom_l]*quo
    cluster_size.append(rem)
    
    import copy
    data = copy.deepcopy(coords)
    data[:,0] = (coords[:,0]-min(coords[:,0]))/(max(coords[:,0])-min(coords[:,0]))
    data[:,1] = (coords[:,1]-min(coords[:,1]))/(max(coords[:,1])-min(coords[:,1]))
    
    # kmeans = KMeans(n_clusters=quo+1,cluster_sizes=cluster_size,device='cpu',verbose=True,max_iter=3,use_label_soft_value=True,sim_fun=minkowski_similarity(p=1))
    # kmeans.fit(data)
    from cuml.cluster import KMeans
    import time
    x = time.time()
    kmeans = KMeans(n_clusters=quo+1, init='random', max_iter=10, random_state=0)
    vals = kmeans.fit_predict(coords)
    
    d = Counter(vals)
    d = sorted(d.items(),key=lambda x:x[1])
    for key,val in d:
        print(key,val)
    print(len(d))
    t = time.time()-x
    print(t)
    # print(sorted(kmeans.centers,key=lambda x:x[0]))
    # print(kmeans.centers)

    # vals = kmeans.labels
    vals = (vals-min(vals))/(max(vals)-min(vals))*255

    
    img = np.array(Image.open(os.path.join(img_dir,f"{id}.jpg")))[:,:,:3]
    
    coords = np.array(coords/8,dtype=int)
    att_matrix = np.zeros(img[::8,::8].shape[:2])
    att_matrix[coords[:,1],coords[:,0]] = vals
    
    mask_attention = np.zeros(img.shape[:2])
    print(coords.shape)
    for i in range(8):
        for j in range(8):
            mask_attention[coords[:,1]*8+i,coords[:,0]*8+j] = att_matrix[coords[:,1],coords[:,0]]
    img = np.float32(img/256)
    img = show_cam_on_image(img,mask_attention,use_rgb=True,image_weight=0.6)
    Image.fromarray(img).save(f'z3.png')
    
    # for i in range(100):
    #     import copy
    #     vals = copy.deepcopy(kmeans.labels)
    #     vals[vals!=i] = 0
    #     vals = vals.reshape(h,w).numpy()
    #     vals = (vals-np.min(vals))/(np.max(vals)-np.min(vals))*255
    #     vals = np.float32(vals)
    #     vals = vals.astype(np.uint8)
        

    #     import cv2
    #     vals = cv2.applyColorMap(vals, cv2.COLORMAP_JET)
    #     img = Image.fromarray(vals)
    #     img.save(f'z_{i}.png')
