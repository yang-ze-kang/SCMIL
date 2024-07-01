from PIL import Image
from torch.utils import data
from lifelines import KaplanMeierFitter
import torch
import numpy as np
import pandas as pd
import h5py
import torch
import os
import pdb

class DataGeneratorTCGASurvivalWSIMDN(data.Dataset):

    def __init__(self, anno_file_path, wsi_id_path, clinical_path, shuffle=False,
                 image_processor=None, with_ids=False, with_coords=False, cluster_label_path=None, eps=1e-6):
        self.image_processor = image_processor
        self.with_ids = with_ids
        self.with_coords = with_coords
        self.cluster_label_path = cluster_label_path
        
        annos = np.genfromtxt(anno_file_path, delimiter=',', dtype=str, skip_header=1)
        target_ids = np.genfromtxt(wsi_id_path, delimiter='\n', dtype=str)
        # clinical data
        clinical = pd.read_csv(clinical_path)
        clinical_keys = ['censorship','survival_months']
        
        self.pids = set()
        self.wsis = []
        self.wsi2h5path = {}
        self.wsi2clinical = {}
        ts = []
        cs = []
        for anno in annos:
            id = anno[1]
            if id in target_ids:
                self.pids.add(id[:12])
                self.wsis.append(id)
                self.wsi2h5path[id] = anno[3]
                row = clinical[clinical['case_id']==id[:12]]
                assert len(row)==1
                self.wsi2clinical[id] = {key:row[key].values[0] for key in clinical_keys}
                ts.append(row['survival_months'].values[0])
                cs.append(row['censorship'].values[0])
        ts = torch.Tensor(ts)
        cs = torch.Tensor(cs)
        constant_dict = {}
        idx = np.argsort(ts)
        ts = ts[idx]
        cs = cs[idx]
        # Eval time steps for time-dependent C-index.
        constant_dict["eval_t"] = torch.unique(ts).sort()[0]
        # Eval min and max time steps for Brier Score
        constant_dict['NUM_INT_STEPS'] = 1000
        constant_dict["t_min"] = torch.tensor(ts[0], dtype=torch.float32)
        constant_dict["t_max"] = torch.tensor(ts[-1], dtype=torch.float32)
        kmf = KaplanMeierFitter()
        kmf.fit(ts, event_observed=cs)
        G_T = kmf.predict(ts, interpolate=True).to_numpy()
        for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
            constant_dict["t_max_{}".format(eps)] = torch.tensor(
                max(ts[G_T > eps]), dtype=torch.float32)
        self.constant_dict = constant_dict
        if shuffle:
            np.random.shuffle(self.wsis)

    def summary(self, name='Train'):
        print(f"==={name}===")
        print('Patient-LVL; Number of samples: %d' % (len(self.pids)))
        print('Slide-LVL; Number of samples: %d' % (len(self.wsis)))

    def __len__(self):
        return len(self.wsis)

    def __getitem__(self, index):
        id = self.wsis[index]
        data = {}
        data['wid'] = id
        event_time = self.wsi2clinical[id]['survival_months']
        censorship = self.wsi2clinical[id]['censorship']
        with h5py.File(self.wsi2h5path[id],'r') as f:
            features = torch.tensor(np.array(f['features']),dtype=torch.float32)
            if self.with_coords:
                coords = torch.tensor(np.array(f['coords']),dtype=torch.float32)
                data['coords'] = coords
        data.update({
            'x':features,
            't':torch.tensor(event_time),
            'c':torch.tensor(censorship)
        })
        if self.cluster_label_path is not None:
            with h5py.File(self.cluster_label_path,'r') as f:
                cluster_label = np.array(f[id])
            data['cluster_label'] = cluster_label
        return data, self.constant_dict