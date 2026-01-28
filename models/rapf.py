import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleClipNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy, get_attribute, ClipLoss
from utils.data_manager import LaionData
import copy
import random
num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleClipNet(args, True)
        self.batch_size = get_attribute(args, "batch_size", 128)
        self.init_lr = get_attribute(args, "init_lr", 0.001)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.args = args
        self.epochs = get_attribute(args, "epochs", 15)
        self.beta = get_attribute(args, "beta", 2)
        self.mix_bias = get_attribute(args, "mix_bias", 0.6)
        self.threshold = get_attribute(args, "threshold", 0.55)
        self.shrinkage = get_attribute(args, "shrinkage", False)
        self.milestones = [4,10]
        self.dtype = torch.float32
        self.text_tokens = None
        self.adapter = nn.Linear(512, 512, bias=False, device=self._device)

        self.old_adapter = None
        self.old_edge_samples = []
        self.old_edge_samples_labels = []
        self.old_edge_samples_nearest_labels = []

        # class stat
        self.class_mean_list = []
        self.class_cov_list = []

        self.class_diff = None
        self.nearest_class = None
        self.class_edge_distance = []


    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.classnames = self.data_manager._class_to_label
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.sample_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self._network.to(self._device)
        self.adaptation(self._cur_task, self.threshold)
        self._train(self.train_loader, self.test_loader, self.sample_loader)

    def _train(self,train_loader,test_loader,sample_loader):
        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.init_lr, weight_decay=0.0000)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, gamma=0.1, last_epoch=-1)
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            loss = torch.tensor(0.0).to(self._device)
            loss_c = torch.tensor(0.0).to(self._device)
            loss_hinge = torch.tensor(0.0).to(self._device)
            if self._cur_task > 0:
                random_class_order_list = list(range(self._known_classes))
                random.shuffle(random_class_order_list)
            batch_id = -1
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                batch_id += 1
                sg_inputs = None
                edge_samples = None
                if self._cur_task > 0:
                    sg_inputs = []
                    sg_targets = []
                    list_for_one_batch = [random_class_order_list[batch_id * 2 % len(random_class_order_list)],
                                          random_class_order_list[
                                              (batch_id * 2 + 1) % len(random_class_order_list)]]
                    for i in list_for_one_batch:
                        sg_inputs.append(sample(self.class_mean_list[i], self.class_cov_list[i], int(10*self.beta), self.shrinkage))
                        sg_targets.append(torch.ones(int(10 * self.beta), dtype=torch.long, device=self._device) * i)
                    sg_inputs = torch.cat(sg_inputs, dim=0)
                    sg_targets = torch.cat(sg_targets, dim=0)
                    targets = torch.cat([targets, sg_targets], dim=0)
                if self.hard_pairs is not None and self.hard_pairs.shape[0] > 0:
                    edge_samples = []
                    edge_p_target = []
                    edge_n_target = []
                    for hard_pair in self.hard_pairs:
                        edge_samples.append(
                            sample(self.class_mean_list[hard_pair[0]], self.class_cov_list[hard_pair[0]], int(20*self.beta),shrink=self.shrinkage)
                        )
                        edge_p_target.append(
                            torch.ones(int(20 * self.beta), dtype=torch.long, device=self._device) * hard_pair[0])
                        edge_n_target.append(
                            torch.ones(int(20 * self.beta), dtype=torch.long, device=self._device) * hard_pair[1])
                    edge_samples = torch.cat(edge_samples, dim=0)
                    edge_p_target = torch.cat(edge_p_target, dim=0)
                    edge_n_target = torch.cat(edge_n_target, dim=0)
                if self._cur_task > 0:
                    not_ini = True
                else:
                    not_ini = False
                outputs, _, __, edge_sample_features = self.forward_once(inputs, memory_data=sg_inputs, not_ini=not_ini,
                                                             edge_sample=edge_samples, prompt=False)
                if self._cur_task > 0:
                    if edge_samples is not None:
                        edge_sample_features = edge_sample_features / edge_sample_features.norm(dim=-1, keepdim=True)
                        edge_target_features = self.class_name_features[edge_p_target].type(edge_sample_features.dtype)
                        edge_target_features = edge_target_features / edge_target_features.norm(dim=-1, keepdim=True)
                        edge_nearest_class_features = self.class_name_features[edge_n_target].type(
                            edge_sample_features.dtype)
                        edge_nearest_class_features = edge_nearest_class_features / edge_nearest_class_features.norm(
                            dim=-1, keepdim=True)
                        loss_hinge = torch.relu(
                            - (edge_sample_features * edge_target_features.clone().detach()).sum(-1) + (
                                    edge_sample_features * edge_nearest_class_features.clone().detach()).sum(
                                -1) + 0.1).mean()

                loss_c = torch.nn.functional.cross_entropy(outputs, targets.detach())
                if edge_samples is not None:
                    loss = loss_c + loss_hinge
                else:
                    loss = loss_c
                #计算train_acc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
           # train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Test_acc {:.2f}".format(
                self._cur_task, epoch + 1, self.args['epochs'], loss / len(train_loader), test_acc, )
            prog_bar.set_description(info)
        sample_data = []
        sample_target = []
        for i, (_, inputs, targets) in enumerate(sample_loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            with torch.no_grad():
                _, ori_ima_feat, after_adapt_feature = self.forward_once(inputs, ori_ima_f=True)
            sample_data.append(ori_ima_feat)
            sample_target.append(targets)
        sample_target = torch.cat(sample_target, dim=0)
        sample_data = torch.cat(sample_data, dim=0)
        self.analyze_mean_cov(sample_data, sample_target)
        self.mix_matrix()
        



    def adaptation(self, task_id, threshold=0):
        self.current_class_names = self.classnames[:self._total_classes]
        self.current_class_num = self._total_classes - self._known_classes
        self.text_tokens = self._network.tokenizer(
            ["a good photo of a {}.".format(c) for c in self.current_class_names]
        ).to(self._device)
        self.text_end = self.text_tokens.max(dim=-1)[1]
        self.class_name_features = self._network.convnet.encode_text(self.text_tokens)
        self.class_name_features = self.class_name_features / self.class_name_features.norm(dim=-1, p=2, keepdim=True)
        self.hard_pairs = None
        if task_id > 0:
            self.old_adapter = copy.deepcopy(self.adapter)
            dist_list = []
            for k, class_name_feature in enumerate(self.class_name_features[:-self.current_class_num]):
                diff = torch.cdist(
                    self.class_name_features[-self.current_class_num:].type(torch.float32),
                    class_name_feature.unsqueeze(0).type(torch.float32)
                ).squeeze()
                dist_list.append(diff)
            dist_list = torch.stack(dist_list)
            self.class_diff = dist_list
            mask = self.class_diff < threshold
            indices = torch.nonzero(mask)
            self.hard_pairs = indices
            self.hard_pairs[:, 1] = self.hard_pairs[:, 1] + self._known_classes

    def forward_once(self, image, ori_ima_f=False, memory_data=None, not_ini=False, edge_sample=None, prompt=False):
        image = image.type(torch.float32)
        with torch.no_grad():
            text_features = self._network.convnet.encode_text(self.text_tokens)
        with torch.no_grad():
            image_features = self._network.convnet.encode_image(image)
            original_image_features = image_features.clone()
        if memory_data is not None:
            memory_data = memory_data.type(self.dtype)
            image_features = torch.cat([image_features, memory_data], dim=0)
        if edge_sample is not None:
            edge_sample = edge_sample.type(self.dtype)
            edge_num = edge_sample.shape[0]
            image_features = torch.cat([image_features, edge_sample], dim=0)
        image_features = self.adapter(image_features.type(self.dtype).detach())
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        if edge_sample is not None:
            edge_sample_features = image_features[-edge_num:]
            image_features = image_features[:-edge_num]
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self._network.convnet.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t().type(image_features.dtype)
        probs = logits_per_image
        if not_ini:
            with torch.no_grad():
                old_memory_feature = self.old_adapter(memory_data)
                old_memory_feature = old_memory_feature / old_memory_feature.norm(dim=1, keepdim=True)
            if edge_sample is not None:
                return probs, image_features, old_memory_feature, edge_sample_features
            return probs, image_features, old_memory_feature, text_features
        if ori_ima_f:
            if memory_data is not None:
                image_features = image_features[:-memory_data.shape[0]]
            return probs, original_image_features, image_features
        return probs, image_features, None, None

    def analyze_mean_cov(self, features, labels):
        label = torch.sort(torch.unique(labels))[0]
        for l in label:
            index = torch.nonzero(labels == l)
            index = index.squeeze()
            class_data = features[index]
            mean = class_data.mean(dim=0)
            cov = torch.cov(class_data.t()) + 1e-4 * torch.eye(class_data.shape[-1], device=class_data.device)
            distance = torch.cdist(class_data, mean.unsqueeze(0)).squeeze()
            max_distance = torch.sort(distance)[0][-10:]
            self.class_edge_distance.append((max_distance.mean() - max_distance.min(),
                                             max_distance.max() - max_distance.mean(), max_distance.mean()))
            self.class_mean_list.append(mean)
            self.class_cov_list.append(cov)
        
    def mix_matrix(self):
        if self.old_adapter is not None:
            weight_new = self.adapter.weight.data
            weight_old = self.old_adapter.weight.data
            dist = (weight_new - weight_old).abs()
            U_old, S_old, V_old = torch.linalg.svd(weight_old)
            P_new = U_old.T @ weight_new
            dist = (P_new - torch.diag(S_old) @ V_old).abs()
            mask = dist / dist.max()
            mask += self.mix_bias
            mask = torch.clamp(mask, max=1)
            right = P_new * mask + torch.diag(S_old) @ V_old * (1 - mask)
            weight = U_old @ right
            self.adapter.weight.data = weight
            return


    def _compute_accuracy(self, model, loader):
        self._network.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs,_,_,_ = self.forward_once(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        print('Accuracy: {:.2f}%'.format(correct * 100 / total))
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs,_,_,_ = self.forward_once(inputs)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

def shrink_cov(cov):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag *mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0], device=cov.device)
    alpha1 = 1
    alpha2  = 1
    cov_ = cov + (alpha1 *diag_mean *iden) + (alpha2 *off_diag_mean *( 1 -iden))
    return cov_
    
def sample(mean, cov, size, shrink=False):
    vec = torch.randn(size, mean.shape[-1], device=mean.device)
    if shrink:
        cov = shrink_cov(cov)
    sqrt_cov = torch.linalg.cholesky(cov)
    vec = vec @ sqrt_cov.t()
    vec = vec + mean
    return vec