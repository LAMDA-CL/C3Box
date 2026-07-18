import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from einops import einsum, rearrange
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import Area
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
from utils.data_manager import LaionData, DataManager
from torch.utils.data import DataLoader
import torchvision.transforms as T
import math
import matplotlib.pyplot as plt
import os
import json
import random
random.seed(1993)
np.random.seed(1993)

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = Area(args)
        self.args = args
        self.normalized_text_features = None
        self.prefix = get_attribute(self.args, 'prefix', 'a photo of a {}.')
        self.init_lr = get_attribute(self.args, 'init_lr', 0.001)
        self.weight_decay = get_attribute(self.args, 'weight_decay', 0.0001)
        self.milestones = get_attribute(self.args, 'milestones', [7, 15])
        self.gamma = get_attribute(self.args, 'scheduler_gamma', 0.1)
        self.epochs = get_attribute(self.args, 'epochs', 20)
        self.dataset = get_attribute(self.args, 'dataset', None)
        self.increment = get_attribute(self.args, 'increment', 10)
        self.beta = get_attribute(self.args, 'beta', 1)
        self.batch_size = get_attribute(self.args, 'batch_size', 128)
        self.samples_per_class_proto = get_attribute(
            self.args, 'samples_per_class', 4)
        self.K = get_attribute(self.args, 'K', 16)
        self.text_des_path = get_attribute(self.args, 'text_des_path', None)
        self.occ_des_path = get_attribute(self.args, 'occ_des_path', None)
        self.aug_des_path = get_attribute(self.args, 'aug_des_path', None)
        self.vib_lambda = get_attribute(self.args, 'vib_lambda', 1.0)
        self.precomputed_basis_path = get_attribute(
            self.args, 'precomputed_basis_path', None)
        self.g_lambda = get_attribute(self.args, 'g_lambda', 0.0)
        self.text_features = None
        self.visual_base_matrices = None
        self.textual_base_matrices = None
        self.task_sizes = [0]

    @property
    def samples_per_class(self):
        return self.samples_per_class_proto

    def after_task(self):
        self._known_classes = self._total_classes

    @torch.no_grad()
    def _get_class_name_features(self):
        self.classnames = []
        self.classnames = [c.replace(" ", "_") for c in self.classnames]
        labels = list(range(len(self.classnames)))
        classnames = self.classnames
        template = "a good photo of a {}."
        texts = [template.format(c.replace('_', ' ')) for c in classnames]
        texts = self._network.tokenizer(texts).to(self._device)
        self.text_features = torch.zeros(
            len(classnames), self._network.model.text_projection.shape[1]).to(self._device)
        for i in labels:
            text_feature = self._network.encode_text(texts)
            self.text_features[i] = text_feature[i]

    @torch.no_grad()
    def log_map(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:

        cos_theta = torch.matmul(x, mu.reshape(-1))  # Shape: (N,)

        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

        theta = torch.acos(cos_theta)  # Shape: (N,)
        sin_theta = torch.sqrt(1 - cos_theta ** 2)  # shape: (N,)

        coeff = theta / (sin_theta + 1e-6)  # shape: (N,)

        mask_stable = theta > 1e-4
        scaling = torch.ones_like(coeff)  # shape: (N,)
        scaling[mask_stable] = coeff[mask_stable]

        # Shape: (N, D)
        vec_diff = x - cos_theta.unsqueeze(1) * mu.unsqueeze(0)

        u = scaling.unsqueeze(1) * vec_diff  # Shape: (N, D)

        return u

    @torch.no_grad()
    def _get_visual_base_matrix(self, data_manager: DataManager):
        logging.info("Computing visual base matrixs...")
        sample_dataset = data_manager.get_dataset(
            range(0, data_manager.get_total_classnum()), "train", "train")
        sample_loader = DataLoader(
            sample_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        sample_data = [[] for _ in range(data_manager.get_total_classnum())]
        self.visual_base_matrices = torch.zeros(
            data_manager.get_total_classnum(), 512, self.K).to(self._device)
        prog_bar = tqdm(sample_loader)
        for _, input, target in prog_bar:
            input = input.to(self._device)
            target = target.to(self._device)
            with torch.no_grad():
                image_features = self._network.encode_image(input)
            sample_data[target.item()].append(image_features)
        sample_data = [torch.cat(features, dim=0) for features in sample_data]
        # SVD decomposition
        for label in range(len(sample_data)):
            data = sample_data[label]
            mu_c = torch.mean(data, dim=0, keepdim=True)
            mu_c = F.normalize(mu_c, dim=-1)
            data = self.log_map(data, mu_c).squeeze(0)
            U, S, Vh = torch.linalg.svd(data.cpu(), full_matrices=False)
            U = U.to(data.device)
            S = S.to(data.device)
            Vh = Vh.to(data.device)
            base_matrix = Vh[:self.K, :].t()  # [D, K]
            self.visual_base_matrices[label] = base_matrix  # store base matrix

    @torch.no_grad()
    def _get_textual_base_matrix(self, data_manager: DataManager):
        logging.info("Computing textual base matrixs...")
        self.textual_base_matrices = torch.zeros(
            data_manager.get_total_classnum(), 512, self.K).to(self._device)
        with open(os.path.join(self.text_des_path, "classnames.txt"), 'r') as f:
            classnames = f.readlines()
        classnames = classnames[:data_manager.get_total_classnum()]
        prog_bar = tqdm(classnames)
        for idx, classname in enumerate(prog_bar):
            classname = classname.strip()
            descriptions = []
            with open(os.path.join(self.text_des_path, classname.replace(" ", "_").replace("/", "_") + "_descriptions.txt"), 'r') as desc_f:
                descriptions = desc_f.readlines()
            desc_features = []
            for desc in descriptions:
                desc = desc.strip()
                text = self._network.tokenizer(
                    [desc]).to(self._device)
                with torch.no_grad():
                    text_feature = self._network.encode_text(text)
                desc_features.append(text_feature)
            desc_features = torch.cat(desc_features, dim=0)  # [N, D]
            mu_c = torch.mean(desc_features, dim=0, keepdim=True)
            mu_c = F.normalize(mu_c, dim=-1)
            desc_features = self.log_map(desc_features, mu_c)
            desc_features = desc_features.squeeze(0)
            U, S, Vh = torch.linalg.svd(
                desc_features.cpu(), full_matrices=False)
            U = U.to(desc_features.device)
            S = S.to(desc_features.device)
            Vh = Vh.to(desc_features.device)
            base_matrix = Vh[:self.K, :].t()  # [D, K]
            self.textual_base_matrices[idx] = base_matrix  # store base matrix
        self.textual_base_matrices = self.textual_base_matrices[data_manager._class_order].to(
            self._device)

    def incremental_train(self, data_manager: DataManager):
        self._cur_task += 1
        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(self._total_classes)
        self._network.append_S(device=self._device)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self._network.to(self._device)
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        if self.text_features is None:
            self._get_class_name_features()
            self.text_features = self.text_features.to(self._device)
            self.text_features = self.text_features[data_manager._class_order]
        if self.textual_base_matrices is None:
            if self.precomputed_basis_path is not None and os.path.exists(os.path.join(self.precomputed_basis_path, 'textual_base_matrices.pth')):
                print("Loading precomputed textual base matrices from {}".format(
                    self.precomputed_basis_path))
                loaded_dict = torch.load(os.path.join(
                    self.precomputed_basis_path, 'textual_base_matrices.pth'))
                self.textual_base_matrices = loaded_dict['textual_base_matrices'].to(
                    self._device)
            else:
                self._get_textual_base_matrix(data_manager)
                if self.precomputed_basis_path is not None:
                    save_path = os.path.join(
                        self.precomputed_basis_path, 'textual_base_matrices.pth')
                    torch.save(
                        {'textual_base_matrices': self.textual_base_matrices.cpu()}, save_path)
        if self.visual_base_matrices is None:
            if self.precomputed_basis_path is not None and os.path.exists(os.path.join(self.precomputed_basis_path, 'visual_base_matrices.pth')):
                print("Loading precomputed visual base matrices from {}".format(
                    self.precomputed_basis_path))
                loaded_dict = torch.load(os.path.join(
                    self.precomputed_basis_path, 'visual_base_matrices.pth'))
                self.visual_base_matrices = loaded_dict['visual_base_matrices'].to(
                    self._device)
            else:
                self._get_visual_base_matrix(data_manager)
                if self.precomputed_basis_path is not None:
                    save_path = os.path.join(
                        self.precomputed_basis_path, 'visual_base_matrices.pth')
                    torch.save(
                        {'visual_base_matrices': self.visual_base_matrices.cpu()}, save_path)
        self._network.update_stat(
            self._known_classes, self._total_classes, self.train_loader, self._device)
        self.train(self.train_loader, self.test_loader, data_manager)
        self._update_stat(self.train_loader, data_manager)

    def train(self, train_loader, test_loader, data_manager: DataManager):
        self._network.train()
        augmentation_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.2, hue=0.1),
        ])
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.parameters(
            ), momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(self._network.parameters(
            ), lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.gamma, last_epoch=-1)
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            loss = torch.tensor(0.0).to(self._device)
            loss_c = torch.tensor(0.0).to(self._device)
            if self._cur_task > 0:
                random_class_order_list = list(range(self._known_classes))
                random.shuffle(random_class_order_list)
            batch_id = -1
            for i, (_, inputs, targets) in enumerate(train_loader):
                batch_id += 1
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                real_targets = targets.clone()
                sg_inputs = None
                sg_targets = None
                if self._cur_task > 0:
                    sg_inputs = []
                    sg_targets = []
                    for i in random_class_order_list:
                        class_mean = self._network.class_mean_list[i]
                        class_cov = self._network.class_cov_list[i]
                        sampled_feature = self.sample(
                            class_mean, class_cov, int(self.samples_per_class), shrink=False)
                        sg_inputs.append(sampled_feature)
                        sg_targets.append(torch.ones(
                            int(self.samples_per_class), dtype=torch.long, device=self._device)*i)
                    sg_inputs = torch.cat(sg_inputs, dim=0)
                    sg_targets = torch.cat(sg_targets, dim=0)
                    targets = torch.cat([targets, sg_targets], dim=0)
                batch_visual_basis = self.visual_base_matrices[:self._total_classes]
                batch_textual_basis = self.textual_base_matrices[:self._total_classes]
                outputs = self._network(
                    inputs, self.text_features[:self._total_classes], batch_visual_basis, batch_textual_basis, self._cur_task, memory_data=sg_inputs)
                loss_c = F.cross_entropy(outputs, targets.detach())
                # Occ Loss
                score_clean_gt = self._network._get_visual_score(
                    inputs, self._cur_task)  # [B, K]

                occ_inputs = self._generate_occluded_inputs(
                    inputs, self._device)
                score_occ_gt = self._network._get_visual_score(
                    occ_inputs, self._cur_task)  # [B, K]

                class_name = self.classnames[data_manager._class_order[real_targets[0].item(
                )]]
                occ_des_file = os.path.join(self.occ_des_path, class_name.replace(
                    " ", "_").replace("/", "_") + "_descriptions.txt")
                with open(occ_des_file, 'r') as f:
                    occ_descriptions = f.readlines()
                occ_description = random.choice(occ_descriptions).strip()
                occ_score = self._network._get_textual_score(
                    occ_description, self._cur_task)  # [K]
                clean_score = self._network._get_textual_score(
                    # [K]
                    "a photo of a {}.".format(class_name.replace("_", " ")), self._cur_task)

                loss_mask = torch.mean(torch.relu(
                    score_occ_gt - score_clean_gt)) + torch.mean(torch.relu(occ_score - clean_score))

                M = 3
                view_scores_list = []
                text_scores_list = []
                aug_des_file = os.path.join(self.aug_des_path, class_name.replace(
                    " ", "_").replace("/", "_") + "_descriptions.txt")
                with open(aug_des_file, 'r') as f:
                    aug_descriptions = f.readlines()
                for _ in range(M):
                    aug_inputs = augmentation_transform(inputs)
                    aug_description = random.choice(aug_descriptions).strip()
                    aug_scores_matrix = self._network._get_visual_score(
                        aug_inputs, self._cur_task)
                    aug_scores_txt = self._network._get_textual_score(
                        aug_description, self._cur_task)
                    text_scores_list.append(aug_scores_txt)
                    view_scores_list.append(aug_scores_matrix)

                aug_description = random.choice(aug_descriptions).strip()
                stacked_scores = torch.stack(view_scores_list)  # [M, B, K]
                mean_scores = stacked_scores.mean(dim=0)       # [B, K]
                stacked_text_scores = torch.stack(text_scores_list)  # [M, K]
                mean_text_scores = stacked_text_scores.mean(dim=0)  # [K]
                loss_cons = torch.tensor(0.0, device=self._device)
                for m in range(M):
                    loss_cons += F.l1_loss(view_scores_list[m], mean_scores)
                    loss_cons += F.l1_loss(
                        text_scores_list[m], mean_text_scores)
                loss = loss_c + self.vib_lambda * (loss_mask + loss_cons)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                prog_bar.set_description("Epoch [{}/{}] Loss: {:.4f} Cls Loss: {:.4f}".format(
                    epoch + 1, self.epochs, loss.item(), loss_c.item()
                ))
            scheduler.step()

    def _update_stat(self, train_loader, data_manager: DataManager):
        sample_loader = DataLoader(
            self.train_dataset, batch_size=128, shuffle=False, num_workers=num_workers)
        sample_data = []
        sample_target = []
        for _, input, target in sample_loader:
            input = input.to(self._device)
            target = target.to(self._device)
            with torch.no_grad():
                ori_ima_feat = self._network.encode_image(input)
            sample_data.append(ori_ima_feat)
            sample_target.append(target)
        sample_data = torch.cat(sample_data, dim=0)
        sample_target = torch.cat(sample_target, dim=0)
        self._network.analyze_mean_cov(sample_data, sample_target)

    def sample(self, mean, cov, size, shrink=False):
        vec = torch.randn(size, mean.shape[-1], device=mean.device)
        if shrink:
            cov = self.shrink_cov(cov)
        sqrt_cov = torch.linalg.cholesky(cov.cpu())
        sqrt_cov = sqrt_cov.to(mean.device)
        vec = vec @ sqrt_cov.t()
        vec = vec + mean
        return vec

    def shrink_cov(self, cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag*mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0], device=cov.device)
        alpha1 = 1
        alpha2 = 1
        cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
        return cov_

    @torch.no_grad()
    def get_most_similar_task(self, inputs):
        assert inputs.shape[0] == 1
        self._network.eval()
        inputs = inputs.to(self._device)
        with torch.no_grad():
            image_features = self._network.encode_image(inputs)
            dists = []
            for task_id in range(self._cur_task + 1):
                # [C_task, D, K]
                visual_basis = self.visual_base_matrices[self.task_sizes[task_id]
                    :self.task_sizes[task_id+1]]
                base_matrix = rearrange(visual_basis, 'C D K -> C K D')
                base_matrix = rearrange(
                    base_matrix, 'C K D -> (C K) D')  # [(C_task*K), D]
                cost_matrix = self._compute_cost_matrix(
                    image_features, base_matrix)  # [1, (C_task*K)]
                sinkhorn_distance = self._sinkhorn_distance(cost_matrix)  # [1]
                dists.append(sinkhorn_distance)
            max_task_id = torch.argmax(torch.tensor(dists))
        return max_task_id.item()

    @torch.no_grad()
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                task_id = self.get_most_similar_task(inputs)
                outputs = self._network.forward_inference(inputs, self.text_features[:self._total_classes],
                                                          self.visual_base_matrices[:self._total_classes],
                                                          self.textual_base_matrices[:self._total_classes], task_id)
                transf_image_features_raw_ = self._network.visual_forward_(
                    inputs)
                transf_image_features_raw_ = transf_image_features_raw_ / \
                    transf_image_features_raw_.norm(dim=-1, keepdim=True)
                outputs_gda = transf_image_features_raw_ @ self._network.W + self._network.b
                outputs = (1 - self.g_lambda) * outputs + \
                    self.g_lambda * outputs_gda
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _generate_occluded_inputs(self, inputs, device):
        B, C, H, W = inputs.shape
        occluded_inputs = inputs.clone()

        rho_min, rho_max = 0.1, 0.4
        eta_min, eta_max = 0.33, 3.0

        for i in range(B):
            rho = np.random.uniform(rho_min, rho_max)
            area = int(rho * H * W)

            eta = np.random.uniform(eta_min, eta_max)

            h = int(np.sqrt(area * eta))
            w = int(np.sqrt(area / eta))
            h = np.clip(h, 1, H)
            w = np.clip(w, 1, W)

            if H - h > 0:
                u = np.random.randint(0, H - h + 1)
            else:
                u = 0

            if W - w > 0:
                v = np.random.randint(0, W - w + 1)
            else:
                v = 0

            noise = torch.rand((C, h, w), device=device)
            occluded_inputs[i, :, u:u+h, v:v+w] = noise

        return occluded_inputs

    def _compute_cost_matrix(self, query_emb: torch.Tensor, task_basis: torch.Tensor):
        query_norm = F.normalize(query_emb, p=2, dim=1)
        basis_norm = F.normalize(task_basis, p=2, dim=1)

        cosine_sim = torch.matmul(query_norm, basis_norm.T)

        cost_matrix = 1.0 - cosine_sim
        return cost_matrix

    def _sinkhorn_distance(self, cost_matrix: torch.Tensor):
        B, N_b = cost_matrix.shape
        device = cost_matrix.device

        mu = torch.ones(B, 1, device=device)

        nu = torch.ones(B, N_b, device=device) / N_b

        f = torch.zeros(B, 1, device=device)
        g = torch.zeros(B, N_b, device=device)

        log_K = -cost_matrix / 0.1

        for _ in range(50):
            term1 = g + log_K
            f = torch.log(mu) - torch.logsumexp(term1, dim=1, keepdim=True)
            term2 = f + log_K
            g = torch.log(nu) - (f + log_K)

        wd = (f * mu).sum(dim=1) + (g * nu).sum(dim=1)
        return wd