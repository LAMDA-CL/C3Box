import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers.weight_init import trunc_normal_
from copy import deepcopy

class SimpleContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes, feat_expand=False, with_norm=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.feat_expand = feat_expand
        self.with_norm = with_norm
        heads = []
        single_head = []
        if with_norm:
            single_head.append(nn.LayerNorm(embed_dim))

        single_head.append(nn.Linear(embed_dim, nb_classes, bias=False))
        head = nn.Sequential(*single_head)

        heads.append(head)
        self.heads = nn.ModuleList(heads)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)

    def backup(self):
        self.old_state_dict = deepcopy(self.state_dict())

    def recall(self):
        self.load_state_dict(self.old_state_dict)

    def update(self, nb_classes, freeze_old=True):
        single_head = []
        if self.with_norm:
            single_head.append(nn.LayerNorm(self.embed_dim))

        _fc = nn.Linear(self.embed_dim, nb_classes, bias=False)
        trunc_normal_(_fc.weight, std=.02)
        single_head.append(_fc)
        new_head = nn.Sequential(*single_head)

        if freeze_old:
            for p in self.heads.parameters():
                p.requires_grad = False

        self.heads.append(new_head)

    def forward(self, x):
        out = []
        for ti in range(len(self.heads)):
            fc_inp = x[ti] if self.feat_expand else x
            out.append(1*(F.linear(F.normalize(fc_inp, p=2, dim=1),F.normalize(self.heads[ti][0].weight, p=2, dim=1))))
        out = {'logits': torch.cat(out, dim=1)}
        return out



class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=False):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}




def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


