import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

def safe_inverse_softplus(x):
    return torch.log(torch.exp(x) - 1 + 1e-6)

def inverse_softplus_grad(x):
    return torch.exp(x) / (torch.exp(x) - 1 + 1e-6)

def logsumexp(a, dim, b):
    # support subtraction in logsumexp
    a_max = torch.max(a, dim=dim, keepdims=True)[0]
    out = torch.log(torch.sum(b * torch.exp(a - a_max), dim=dim, keepdims=True) + 1e-6)
    out += a_max
    return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers=2,dropout_rate=0.25):
        super(MLP, self).__init__()
        self.hidden_layers = hidden_layers
        layers = []
        for i in range(self.hidden_layers):
            if i == 0:
                in_size = input_size
            else:
                in_size = hidden_size
            layers.extend([nn.Linear(in_size,hidden_size),nn.GELU(),nn.Dropout(dropout_rate)])
            # layers.extend([nn.Linear(in_size,hidden_size,bias=False),nn.SELU(),nn.AlphaDropout(dropout_rate)])
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size,output_size)
            

    def forward(self, x):
        h = self.layers(x)
        o = self.fc(h)
        return o


class SurvivalMDN(nn.Module):
    
    def __init__(self,backbone=None,input_size=384,hidden_size=256,K=10,param_share = ['mu','sigma'],dropout=0.1,**kwargs):
        super().__init__()
        self.K = K
        self.param_share = param_share
        self.param_names = ['w','mu','sigma']
        self.params = {}
        self.backbone = backbone
        self.mlp = MLP(input_size=input_size,hidden_size=hidden_size,output_size=K*(3-len(param_share)),dropout_rate=dropout)
        for param_name in self.param_share:
            if param_name=='mu':
                self.params[param_name] = nn.Parameter(torch.linspace(0,200,self.K,dtype=torch.float32).unsqueeze(0))
                # self.params['mu'] = torch.linspace(0,240,self.K,dtype=torch.float32).unsqueeze(0).to('cuda')
                # self.params[param_name] = nn.Parameter(torch.randn((1,self.K)))
            elif param_name=='sigma':
                self.params[param_name] = nn.Parameter(torch.ones((1,self.K)))
                # self.params['sigma'] = torch.ones((1,self.K)).to('cuda')
                # self.params[param_name] = nn.Parameter(torch.randn((1,self.K)))
        self.params = nn.ParameterDict(self.params)
        self.layer_mu = nn.Sequential(nn.Linear(self.K,self.K))
        self.layer_sigma = nn.Sequential(nn.Linear(self.K,self.K))
    
    def cdf(self,params,times):
        inv_softplus_times = safe_inverse_softplus(times)
        inv_softplus_times = inv_softplus_times.unsqueeze(-1)
        ws,mus,sigmas = params['w'],params['mu'], params['sigma']
        if len(times.shape)==1:
            normal_dists = torch.distributions.Normal(mus, sigmas)
            repeated_inv_softplus_times = inv_softplus_times.repeat(1, self.K)
            log_normal_cdfs = normal_dists.cdf(repeated_inv_softplus_times)
            cdfs = ws * log_normal_cdfs
            cdfs = cdfs.sum(-1)
        elif len(times.shape)==2:
            time_len = times.shape[-1]
            mus = mus.unsqueeze(1).repeat(1, time_len, 1)
            sigmas = sigmas.unsqueeze(1).repeat(1, time_len, 1)
            normal_dists = torch.distributions.Normal(mus, sigmas)
            repeated_inv_softplus_times = inv_softplus_times.repeat(1, 1, self.K)
            log_normal_cdfs = normal_dists.cdf(repeated_inv_softplus_times)
            ws = ws.unsqueeze(1)
            cdfs = ws * log_normal_cdfs
            cdfs = cdfs.sum(-1)
            cdfs[cdfs > 1.] = 1.
        else:
            assert  False, "times shape not supported"
        return cdfs
    
    def log_prob(self,params, times):
        inv_softplus_times = safe_inverse_softplus(times)
        inv_softplus_times = inv_softplus_times.unsqueeze(-1)
        ws, mus, sigmas = params['w'], params['mu'], params['sigma']
        num_components = ws.shape[1]
        normal_dists = torch.distributions.Normal(mus, sigmas)
        repeated_inv_softplus_times = inv_softplus_times.repeat(1, 1, num_components)
        log_normal_pdfs = normal_dists.log_prob(repeated_inv_softplus_times)
        log_pdfs = logsumexp(a=log_normal_pdfs, dim=-1, b=ws).squeeze()
        log_grad = torch.log(inverse_softplus_grad(times))
        return log_pdfs + log_grad
    
    def forward(self,x,**kargs):
        h = self.backbone(x,**kargs)
        if isinstance(h,dict):
            subloss = h['loss']
            h = h['feat']
        else:
            subloss = None
        h = self.mlp(h)
        i = 0
        params = {}
        for param_name in self.param_names:
            if param_name not in self.param_share:
                params[param_name] = h[:,i:i+self.K]
                i+=self.K
            else:
                params[param_name] = self.params[param_name]
        params['w'] = params['w'].softmax(-1)
        if 'mu' in self.param_share:
            params['mu'] = self.layer_mu(params['mu'])
        if 'sigma' in self.param_share:
            params['sigma'] = self.layer_sigma(params['sigma'])
        params['mu'] = params['mu'].clamp(min=-200,max=200)
        params['sigma'] = F.softplus(params['sigma'])
        params['sigma'] = params['sigma'].clamp(min=1e-2,max=1e2)
        
        
        return params, subloss
    
    def train_step(self,x,t,c,constant_dict):
        params, subloss = self(**x)
        survival_func = 1 - self.cdf(params,t)
        pdf = torch.exp(self.log_prob(params,t))
        train_outputs = {
            't':t,
            'c':c,
            'pdf':pdf,
            'survival_func':survival_func,
            'subloss':subloss
        }
        eval_outputs = self.eval_step(x,t,c,constant_dict)
        return train_outputs,eval_outputs
        
    def eval_step(self,x,t,c,constant_dict):
        with torch.no_grad():
            params, _ = self(**x)
            outputs = {}
            # Eval for time-dependent C-index
            outputs["t"] = t
            outputs['c'] = c
            outputs["eval_t"] = constant_dict["eval_t"] # eval_len
            # batch_size = x['x'].shape[0]
            batch_size = 1
            device = t.device
            eval_t = constant_dict["eval_t"].repeat(batch_size, 1).to(device) # batch_size, eval_len
            cdf = self.cdf(params,eval_t) # batch_size, eval_len
            survival_func = 1 - cdf # batch_size, eval_len
            cum_hazard = - torch.log(survival_func) # batch_size, eval_len
            outputs["cum_hazard_seqs"] = cum_hazard.transpose(1, 0) # eval_len, batch_size

            # Eval for Brier Score
            t_min = constant_dict["t_min"]
            t_max = constant_dict["t_max"]
            t = torch.linspace(
                t_min.item(), t_max.item(), constant_dict['NUM_INT_STEPS'].item(), dtype=t_min.dtype,
                device=device)
            eval_t = t.unsqueeze(0).repeat(batch_size, 1) # batch_size, eval_len
            cdf = self.cdf(params,eval_t) # batch_size, eval_len
            survival_func = 1 - cdf # batch_size, eval_len
            outputs["survival_seqs"] = survival_func.transpose(1, 0) # eval_len, batch_size

            for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
                t_min = constant_dict["t_min"]
                t_max = constant_dict["t_max_{}".format(eps)]
                t = torch.linspace(
                    t_min.item(), t_max.item(), constant_dict['NUM_INT_STEPS'].item(), dtype=t_min.dtype,
                    device=device)
                eval_t = t.unsqueeze(0).repeat(batch_size, 1)  # batch_size, eval_len
                cdf = self.cdf(params,eval_t)  # batch_size, eval_len
                survival_func = 1 - cdf  # batch_size, eval_len
                outputs["survival_seqs_{}".format(eps)] = survival_func.transpose(1, 0) # eval_len, batch_size
        return outputs
    
    def predict_step(self,x):
        device = x['x'].device
        with torch.no_grad():
            params,_ = self(**x)
            eval_t = torch.range(0.1,220,0.1).to(device)
            cdf = self.cdf(params,eval_t)
            survival_func = 1 - cdf
        ret = {
            't':eval_t,
            'p_survival':survival_func
        }
        return ret
                
                
        
        