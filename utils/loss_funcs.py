from typing import Any
import torch

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, label, censorship, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, label, censorship, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, label, censorship, alpha=alpha)
        
        
class MDNML(object):
    
    def __init__(self) -> None:
        pass
    
    def __call__(self, datas):
        c = datas['c']
        pdf = datas['pdf']
        survival_func = datas['survival_func']
        loss = -(1-c)*torch.log((pdf+1e-6)/(survival_func+1e-6)) - torch.log(survival_func + 1e-6)
        return loss


# class MDNML(object):
    
#     def __init__(self) -> None:
#         pass
    
#     def __call__(self, datas):
#         c = datas['c']
#         pdf = datas['pdf']
#         survival_func = datas['survival_func']
#         loss = -(1-c)*torch.log((1-survival_func+1e-6)/(survival_func+1e-6)) - torch.log(survival_func + 1e-6)
#         return loss