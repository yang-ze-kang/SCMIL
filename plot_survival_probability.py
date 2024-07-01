import pdb
import yaml
import argparse
from tqdm import tqdm
from utils.utils import *
from models import create_WSI_model
from lifelines import KaplanMeierFitter
from dataset.dataset_survival_mdn import DataGeneratorTCGASurvivalWSIMDN
from munch import Munch
import torch
import numpy as np
import os
import pdb
import json
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import copy
    
def detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        detached_data = {}
        for key in data:
            detached_data[key] = detach(data[key])
    elif type(data) == list:
        detached_data = []
        for x in data:
            detached_data.append(detach(x))
    else:
        raise NotImplementedError("Type {} not supported.".format(type(data)))
    return detached_data

def set_random_seed(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
def plot_survival_probability_distribution(x,y,t=None,save_path=None,):
    y = [i*100 for i in y]
    plt.figure()
    plt.rcParams.update({'font.size': 28})
    sns.set_theme(style="ticks")
    fig = sns.lineplot(x=x, y=y,color='green',linewidth=5)
    plt.xlim(0,x[-1]+1)
    plt.ylim(0,100)
    plt.xlabel('Time(months)',fontsize=22)
    plt.ylabel('Survival Probability(%)',fontsize=22)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    ax.annotate('', xy=(x[-1]+1, 0), xytext=(-0.5, 0),
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->',lw=3))
    ax.annotate('', xy=(0, 100), xytext=(0, -0.5),
                 arrowprops=dict(facecolor='black', edgecolor='black',arrowstyle='->',lw=3))
    if save_path:
        plt.savefig(save_path,transparent=True)

def plot2(d1,d2,save_path=None):
    d1['y'] = [i*100 for i in d1['y']]
    d2['y'] = [i*100 for i in d2['y']]
    plt.figure()
    plt.rcParams.update({'font.size': 28})
    sns.set_theme(style="ticks")
    sns.lineplot(x=d1['x'], y=d1['y'],color='green',linewidth=3,label='patient 1')
    plt.axvline(x=d1['t'], color='green', linestyle='--', linewidth=3, label=f"Time={d1['t']}")
    sns.lineplot(x=d2['x'], y=d2['y'],color='red',linewidth=3,label='patient 2')
    plt.axvline(x=d2['t'], color='red', linestyle='--', linewidth=3, label=f"Time={d2['t']}")
    xmax = max(d1['x'][-1],d2['x'][-1])+1
    plt.xlim(0,xmax)
    plt.ylim(0,100)
    plt.xlabel('Time(months)',fontsize=22)
    plt.ylabel('Survival Probability(%)',fontsize=22)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.legend(loc='upper right')
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['left'].set_position('zero')
    ax.annotate('', xy=(xmax, 0), xytext=(-0.5, 0),
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->',lw=2))
    ax.annotate('', xy=(0, 100), xytext=(0, -0.5),
                 arrowprops=dict(facecolor='black', edgecolor='black',arrowstyle='->',lw=2))
    # plt.savefig('zz.pdf',transparent=False)
    plt.savefig('z.pdf',transparent=True)
    plt.close()

def plot3(d1,d2,ts,cs,save_path=None):
    """
    with cohort KM
    """
    eps=1e-6
    d1['x'] = np.array(d1['x'])
    d2['x'] = np.array(d2['x'])
    d1['y'] = np.array([i*100 for i in d1['y']])
    d2['y'] = np.array([i*100 for i in d2['y']])
    ids1 = d1['y']>=eps
    ids2 = d2['y']>=eps
    d1['x'] = d1['x'][ids1]
    d1['y'] = d1['y'][ids1]
    d2['x'] = d2['x'][ids2]
    d2['y'] = d2['y'][ids2]
    plt.figure()
    plt.rcParams.update({'font.size': 28})
    sns.set_theme(style="ticks")
    sns.lineplot(x=d1['x'], y=d1['y'],color='green',linewidth=3,label='patient 1')
    plt.axvline(x=d1['t'], color='green', linestyle='--', linewidth=3, label=f"Time={d1['t']}")
    sns.lineplot(x=d2['x'], y=d2['y'],color='red',linewidth=3,label='patient 2')
    plt.axvline(x=d2['t'], color='red', linestyle='--', linewidth=3, label=f"Time={d2['t']}")
    xmax = max(ts)+1
    plt.xlim(0,xmax)
    plt.ylim(0,100)
    plt.xlabel('Time(months)',fontsize=22)
    plt.ylabel('Survival Probability(%)',fontsize=22)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # ax.spines['bottom'].set_position('zero')
    # ax.spines['left'].set_position('zero')
    ax.annotate('', xy=(xmax, 0), xytext=(-0.5, 0),
             arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->',lw=2))
    ax.annotate('', xy=(0, 100), xytext=(0, -0.5),
                 arrowprops=dict(facecolor='black', edgecolor='black',arrowstyle='->',lw=2))
    # plt.savefig('zz.pdf',transparent=False)
    
    kmf = KaplanMeierFitter()
    kmf.fit(ts, event_observed=cs)
    survival_probabilities = kmf.survival_function_
    survival_probabilities=survival_probabilities*100
    time_points = kmf.survival_function_.index
    plt.plot(time_points, survival_probabilities,color='blue', drawstyle='steps-post',marker='+',linewidth=3,label='patient cohort')
    
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.savefig('z.pdf',transparent=True)
    plt.close()

   
def main(args, cfg, save_dir, fold, save_distribution_data=True):
    mini = 1e-6
    set_random_seed(cfg.seed)
    
    model = create_WSI_model(cfg)

    model.to(cfg.device)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'weights','epoch_20.pt')))
    model.eval()

    with_coords = getattr(cfg.datasets,'with_coords',False)
    if cfg.datasets.type == 'tcga-survival-mdn-wsi':
        anno_path = os.path.join(cfg.datasets.root_dir,cfg.datasets.wsi_file_path)
        clinical_path = os.path.join(cfg.datasets.root_dir,cfg.datasets.clinical_file_path)
        val_ids_path = os.path.join(cfg.datasets.root_dir,cfg.datasets.folds_path,f"fold{fold}",'val.txt')
        val_ds = DataGeneratorTCGASurvivalWSIMDN(anno_path,val_ids_path,clinical_path,shuffle=False,with_coords=with_coords,with_ids=True)
        val_ds.summary('Val')
    else:
        raise NotImplementedError
    print(
        f'Datasets loaded! Val sample num: {len(val_ds)}.')
    val_loader = DataLoader(val_ds,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=8)
    data_distribution = {}
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            data,constant_dict = batch
            wid = data['wid'][0]
            t = data['t'].to(cfg.device)
            c = data['c'].to(cfg.device)
            x = {}
            for key in data:
                if key not in ['wid','t','c']:
                    x[key] = data[key].to(cfg.device)
            ret = model.predict_step(x)
            ret = detach(ret)
            x = ret['t'].numpy()
            y = ret['p_survival'].numpy()
            ids = y>mini
            x = x.tolist()
            y = y.tolist()
            if save_distribution_data:
                data_distribution[wid] = {
                    'censorship':c.item(),
                    't':t.item(),
                    'x':x,
                    'y':y
                }
    if save_distribution_data:
        save_path = os.path.join(os.path.join(save_dir,'val_distribution_data.json'))
        with open(save_path,'w') as f:
            json.dump(data_distribution,f)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='--configs/kirc_scmil.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)
    cfg = Munch.fromDict(cfg)

    for fold in [0]:
        save_dir = os.path.join(cfg.save_dir,cfg.config_name,f"fold{fold}")
        save_curves_dir = os.path.join(save_dir,'curves')
        os.makedirs(save_curves_dir,exist_ok=True)
        print(fold)
        main(args, cfg, save_dir, fold, save_distribution_data=True)
        
        with open(os.path.join(save_dir,'val_distribution_data.json'),'r') as f:
            ds = json.load(f)
        
        """
        plot one
        """
        for wid,d in ds.items():
            if d['censorship']==0:
                save_path = os.path.join(save_curves_dir,f'{wid}.png')
                plot_survival_probability_distribution(d['x'],d['y'],d['t'],save_path)

        
        # """
        # plot two compare
        # """
        # ls = [
        #     'TCGA-BP-4981-01Z-00-DX1', # 35
        #     'TCGA-CJ-4637-01Z-00-DX1', # 72
        #     'TCGA-B0-4815-01Z-00-DX1',
        #     'TCGA-B0-4810-01Z-00-DX1',
        #     'TCGA-B0-5107-01Z-00-DX1'
        # ]
        # for i in range(len(ls)):
        #     for j in range(len(ls)):
        #         if i!=j:
        #             plot2(copy.deepcopy(ds[ls[j]]),copy.deepcopy(ds[ls[i]]))
        # plot2(ds[ls[0]],ds[ls[3]])
        
        
        """
        plot two compare with cohort
        """
        # ls = [
        #     'TCGA-BP-4981-01Z-00-DX1', # 35
        #     'TCGA-CJ-4637-01Z-00-DX1', # 72
        #     'TCGA-B0-4815-01Z-00-DX1',
        #     'TCGA-B0-4810-01Z-00-DX1',
        #     'TCGA-B0-5107-01Z-00-DX1'
        # ]
        # ids = list(ds.keys())
        # f = False
        # for i in range(len(ids)):
        #     if ds[ids[i]]['censorship'] == 1:
        #         continue
        #     if ds[ids[i]]['t']<30 or ds[ids[i]]['t']>80:
        #         continue
        #     for j in range(len(ids)):
        #         if ds[ids[j]]['censorship'] == 1:
        #             continue
        #         f = False
        #         if ds[ids[j]]['t']>30:
        #             continue
        #         if i!=j:
        #             plot3(copy.deepcopy(ds[ids[i]]),copy.deepcopy(ds[ids[j]]),ts,cs)
        #         if f:
        #             break
        # plot2(ds[ls[0]],ds[ls[3]])
        