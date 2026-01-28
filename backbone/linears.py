import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.layers import trunc_normal_

from copy import deepcopy

class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
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


class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': reduce_proxies(out1['logits'], self.nb_proxy),
            'new_scores': reduce_proxies(out2['logits'], self.nb_proxy),
            'logits': out
        }


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

class Adapter(nn.Module):
    def __init__(self, c_in, hidden):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, c_in),
            # nn.GELU(),
        )
        self.init_weights(self.fc)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x_ = self.fc(x)
        return x_

class MLP_Adapter(nn.Module):
    def __init__(self, c_in, hidden):
        super(MLP_Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden),
        )
        # self.init_weights(self.fc)

    # def init_weights(self, m):
    #     if type(m) == nn.Linear:
    #         nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x_ = self.fc(x)
        return x_


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


class EaseCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(EaseCosineLinear, self).__init__()
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

    def reset_parameters_to_zero(self):
        self.weight.data.fill_(0)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out


        return {'logits': out}

    def forward_reweight(self, input, cur_task, alpha=0.1, beta=0.0, init_cls=10, inc=10, out_dim=768,
                         use_init_ptm=False):
        for i in range(cur_task + 1):
            if i == 0:
                start_cls = 0
                end_cls = init_cls
            else:
                start_cls = init_cls + (i - 1) * inc
                end_cls = start_cls + inc

            out = 0.0
            for j in range((self.in_features // out_dim)):
                # PTM feature
                if use_init_ptm and j == 0:
                    input_ptm = F.normalize(input[:, 0:out_dim], p=2, dim=1)
                    weight_ptm = F.normalize(self.weight[start_cls:end_cls, 0:out_dim], p=2, dim=1)
                    out_ptm = beta * F.linear(input_ptm, weight_ptm)
                    out += out_ptm
                    continue

                input1 = F.normalize(input[:, j * out_dim:(j + 1) * out_dim], p=2, dim=1)
                weight1 = F.normalize(self.weight[start_cls:end_cls, j * out_dim:(j + 1) * out_dim], p=2, dim=1)
                if use_init_ptm:
                    if j != (i + 1):
                        out1 = alpha * F.linear(input1, weight1)
                      #  print("1")
                        out1 /= cur_task
                    else:
                      #  print("2")
                        out1 = F.linear(input1, weight1)
                else:
                    if j != i:
                        out1 = alpha * F.linear(input1, weight1)
                     #   print("3")
                        out1 /= cur_task
                    else:
                      #  print("4")
                        out1 = F.linear(input1, weight1)

                out += out1

            if i == 0:
                out_all = out
            else:
                out_all = torch.cat((out_all, out), dim=1) if i != 0 else out

        if self.to_reduce:
            # Reduce_proxy
            out_all = reduce_proxies(out_all, self.nb_proxy)

        if self.sigma is not None:
            out_all = self.sigma * out_all

        return {'logits': out_all}

class OLF(nn.Module):

    def __init__(self, in_features, out_features, W0_torch, rank=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.current_rank = 0  # 当前任务的秩
        W0_torch = W0_torch.cuda()  # 确保W0在GPU上
        self.register_buffer('W0', W0_torch.clone().detach())

        self.W_task = nn.Parameter(W0_torch.clone())
        self.B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.A = torch.zeros(self.in_features, self.rank)
        self.W_task.requires_grad = False  # 初始时冻结
        self.W_fusion = W0_torch.clone().detach()
        self.W_fusion2 = W0_torch.clone().detach()
        self.cov_matrices = []

        self.W_list = []
        self.task_id = 0
        self.start_eval = True

    def forward(self, x, stage2=False):

        if self.training:
            # 在训练时，使用当前的可训练参数 W_task
            if stage2:
                return F.linear(x, self.W0+self.B @ self.A.T)
            else:
                return F.linear(x, self.W_task)
        else:
            return F.linear(x, self.W_fusion)

    def eval_forward(self, x):
        return (F.linear(x, self.W_fusion), F.linear(x, self.W_fusion2))

    def update_old_features(self, features):
        mean = torch.mean(features, dim=0, keepdim=True)
        centered_features = features - mean

        k = centered_features.shape[0]
        cov = (1 / (k - 1)) * (features.T @ features)
        reg = 1e-4 * torch.eye(self.in_features, device=self.W0.device)

        self.cov_matrices.append((cov + reg).detach())

    def prepare_for_new_task(self):

        self.train()
        self. start_eval = False
        print(f"OLF: Preparing for Task {self.task_id}. Unfreezing W_task for fine-tuning.")
        self.W_task.requires_grad = True

    def prepare_for_stage2(self):
        self.train()
        self.start_eval = False
        print(f"OLF: Preparing for Stage 2 of Task {self.task_id}. Computing OSS and initializing B.")

        if self.task_id > 0:
            covs_to_aggregate = self.cov_matrices[:-1]
            if len(covs_to_aggregate) > 0:
                avg_cov = torch.mean(torch.stack(covs_to_aggregate, dim=0), dim=0)
                eigenvalues, eigenvectors = torch.linalg.eigh(avg_cov)
                self.A.data = eigenvectors[:, :self.rank].clone()
        else:
            pass  # A 保持为零，阶段二对task 0无效

        delta_W_tilde = self.W_task.data - self.W0
        with torch.no_grad():
            self.B.data = delta_W_tilde @ self.A

        self.W_task.requires_grad = False
        self.B.requires_grad = True

    def end_task(self):

        print(f"OLF: Ending Task {self.task_id}.")
        self.W_task.requires_grad = False

        if self.task_id == 0:
            self.W_list.append(self.W_task.data.clone().detach())
        else:
            diff_W = self.B @ self.A.T
            self.W_list.append(self.W0 + diff_W.detach().clone())
        self.W_fusion = (sum(self.W_list)) / len(self.W_list)
        self.W_fusion2 = (sum(self.W_list)+self.W0) / (len(self.W_list)+1)
        self.task_id += 1
        self.current_rank += self.rank
        self.start_eval = True
        self.eval()

    def get_trainable_parameters(self):
        return [self.W_task]

    def get_stage2_parameters(self):
        return [self.B]

    def get_projected_weight(self, W_t):
        covs_to_aggregate = self.cov_matrices[:-1]
        avg_cov = torch.mean(torch.stack(covs_to_aggregate, dim=0), dim=0)

        _, V = torch.linalg.eigh(avg_cov)
        Unull = V[:, :self.rank]

        P = Unull @ Unull.T
        projected_W = W_t @ P
        return projected_W


class TunaLinear(nn.Module):
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



