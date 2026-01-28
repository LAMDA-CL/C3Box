import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='cosface', eps=1e-7, s=20, m=0):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 20.0 if not s else s
            self.m = 0.0 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)
            return -torch.mean(L)
        
class InfoNCELoss(nn.Module):
    """ SimCLR loss @SimCLR
    Adapted from:
    https://github.com/ysharma1126/ssl_identifiability/blob/master/main_3dident.py
    """
    def __init__(self, tau: float = 0.5) -> None:
        super().__init__()
        self._tau = tau
        assert self._tau != 0
        self._metric = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sim_xx = self._metric(x.unsqueeze(-2), x.unsqueeze(-3)) / self._tau
        sim_yy = self._metric(y.unsqueeze(-2), y.unsqueeze(-3)) / self._tau
        sim_xy = self._metric(x.unsqueeze(-2), y.unsqueeze(-3)) / self._tau

        n = sim_xy.shape[-1]
        sim_xx[..., range(n), range(n)] = float("-inf")
        sim_yy[..., range(n), range(n)] = float("-inf")
        scores1 = torch.cat([sim_xy, sim_xx], dim=-1)    
        scores2 = torch.cat([sim_yy, sim_xy.transpose(-1,-2)], dim=-1)     
        scores = torch.cat([scores1, scores2], dim=-2)  
        targets = torch.arange(2 * n, dtype=torch.long, device=scores.device)
        total_loss = self.criterion(scores, targets)
        return total_loss
    
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, x, y):
        cosine_sim = self.cosine_similarity(x, y)
        loss = 1 - cosine_sim.mean()
        return loss

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
 
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0