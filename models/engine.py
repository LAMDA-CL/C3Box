import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import Engine
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
from utils.data_manager import LaionData
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
        self.args = args

        self._train_transformer = False
        self._network = Engine(args)

        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        self._known_classes = 0
        self.prototype = []

        self.new_des_dict = self._get_text_des(self.args['dataset'])

    def after_task(self):
        self._known_classes = self._total_classes

    def _get_text_des(self, dataname='cifar224'):
        des_path = "./utils/engine/chat/" + dataname + '_des.json'
        with open(des_path, 'r') as f:
            des_dict = json.load(f)
        self.des_dict = des_dict
        new_des_dict = {}
        for key, value in des_dict.items():
            new_key_value = []
            for k, v in value.items():
                new_key_value.extend(v)
            new_des_dict[key] = new_key_value
        return new_des_dict

    def _get_batch_des(self, des_file, classnames):
        batch_des = []
        for classname in classnames:
            batch_des.append(classname + ' with ' + random.choice(des_file[classname]).lower())
        return batch_des

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self._network.to(self._device)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.update_task()
        self._network.update_stat(self._known_classes, self._total_classes, self.train_loader, self._device)
        self.train(self.train_loader, self.test_loader, train_dataset)

    def train(self, train_loader, test_loader, train_dataset):
        self._network.to(self._device)
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,
                                  weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

        class_to_label = self.data_manager._class_to_label
        prog_bar = tqdm(range(self.tuned_epoch))
        total_labels = class_to_label[:self._total_classes]
        templates = self.data_manager._data_to_prompt[0]

        from utils.toolkit import ClipLoss
        cliploss = ClipLoss()
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            correct_clip = []
            if self._cur_task > 0:
                old_class = list(range(self.args['init_cls'] + (self._cur_task - 1) * self.args['increment']))
                random.shuffle(old_class)
            for i, (_, inputs, targets) in enumerate(train_loader):

                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                if self._cur_task > 0:
                    proto_x = []
                    proto_y = []
                    old_class = random.choices(old_class, k=self.args['sample_num']) if self.args[
                                                                                            'sample_num'] > 0 else old_class
                    for j in old_class:
                        proto_x.append(torch.stack(
                            [self.prototype[j] + torch.randn_like(self.prototype[j]) * self.args['sample_noise']],
                            dim=0))
                        proto_y.append(torch.ones(int(1), dtype=torch.long, device=self._device) * j)
                    proto_x = torch.cat(proto_x, dim=0)
                    proto_y = torch.cat(proto_y, dim=0)
                    targets = torch.cat([targets, proto_y], dim=0)

                labels = [class_to_label[y] for y in targets]
                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features = self._network.encode_text(texts)
                text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                image_features = self._network.encode_image(inputs)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True)
                if self._cur_task > 0:
                    sg_image_features = self._network.Image_encode(proto_x)
                    sg_image_features = sg_image_features / sg_image_features.norm(dim=-1, keepdim=True)
                    img_feas = torch.cat([img_feas, sg_image_features], dim=0)
                logit_scale = self._network.model.logit_scale
                logits = img_feas @ text_feas.T

                texts_clip = [templates.format(inst) for inst in labels]
                clip_text_feas = self._network.encode_text(self._network.tokenizer(texts_clip).to(self._device))
                clip_text_norm = clip_text_feas.norm(dim=-1, keepdim=True)
                clip_text_feas = clip_text_feas / clip_text_norm
                clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)

                # calculate image aug loss
                from utils.loss import CosineSimilarityLoss, InfoNCELoss, contrastive_loss
                if self.args['image_aug'] > 0:
                    aug_image = inputs + torch.randn_like(inputs) * 0.25
                    aug_image = torch.clamp(aug_image, 0, 1)
                    aug_image_features = self._network.model.encode_image(aug_image)
                    aug_image_features = aug_image_features / aug_image_features.norm(dim=-1, keepdim=True)
                    aug_image_loss = contrastive_loss(img_feas[:aug_image_features.shape[0]] @ aug_image_features.T)
                elif self.args['image_aug'] == 0:
                    aug_image_loss = 0

                # calculate text aug loss
                if self.args['text_des'] > 0:
                    repeat_ = 1
                    ref_text_loss = []
                    for itera in range(repeat_):
                        ref_texts = self._get_batch_des(self.new_des_dict, labels)
                        tokenizer = self._network.tokenizer
                        ref_emb = tokenizer(ref_texts).to(self._device)
                        ref_text_features = self._network.model.encode_text(ref_emb)
                        ref_text_features = ref_text_features / ref_text_features.norm(dim=-1, keepdim=True)
                        ref_text_loss.append(contrastive_loss(clip_text_feas @ ref_text_features.T))
                    ref_text_loss = sum(ref_text_loss) / repeat_
                elif self.args['text_des'] == 0:
                    ref_text_loss = 0

                loss = clip_loss + ref_text_loss * self.args['text_des'] + aug_image_loss * self.args['image_aug']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                correct_clip.append(preds.eq(targets).cpu())
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader, epoch)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task, epoch + 1, self.args['tuned_epoch'], losses / len(train_loader), train_acc, test_acc, )
            prog_bar.set_description(info)
        self._network.eval()
        # analyze
        sample_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=8)
        x = []
        y = []
        for i, (_, inputs, targets) in enumerate(sample_loader):
            with torch.no_grad():
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                fea = self._network.model.encode_image(inputs)
                x.append(fea)
                y.append(targets)
        y = torch.cat(y, dim=0)
        x = torch.cat(x, dim=0)
        label = torch.sort(torch.unique(y))[0]
        for l in label:
            index = torch.nonzero(y == l).squeeze()
            l_fea = x[index]
            l_mean = l_fea.mean(dim=0)
            self.prototype.append(l_mean)
        self._network.eval()

    def _compute_accuracy(self, model, loader, epoch=0):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]  # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features = self._network.encode_image(inputs)
                transf_image_features = transf_image_features / transf_image_features.norm(dim=-1, keepdim=True)
                transf_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # transf_image_features_raw = self._network.model.encode_image(inputs)
                if epoch == self.args['tuned_epoch'] - 1:
                    transf_image_features_raw_ = self._network.visual_forward_(inputs)
                    transf_image_features_raw = transf_image_features_raw_ @ self._network.visual_proj
                    transf_image_features_raw_ = transf_image_features_raw_ / transf_image_features_raw_.norm(dim=-1,
                                                                                                              keepdim=True)
                    transf_image_features_raw = transf_image_features_raw / transf_image_features_raw.norm(dim=-1,
                                                                                                           keepdim=True)
                    outputs = (transf_image_features @ transf_text_features.T)
                    outputs_gda = transf_image_features_raw_ @ self._network.W + self._network.b
                    outputs_gda = outputs_gda / outputs_gda.norm(dim=-1, keepdim=True)
                    outputs_rerank = self._network.rerank(self.des_dict, outputs, transf_image_features_raw,
                                                          class_to_label, self._device)
                    outputs = outputs_gda * self.args['stat'] + (
                                self.args['rerank'] * outputs_rerank + (1 - self.args['rerank']) * outputs) * (
                                          1 - self.args['stat'])
                else:
                    outputs = (transf_image_features @ transf_text_features.T)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.to(self._device)
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]  # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        text_features = text_features.to(self._device)
        transf_text_features = text_features

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features = self._network.encode_image(inputs)
                transf_image_features = transf_image_features / transf_image_features.norm(dim=-1, keepdim=True)
                transf_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                transf_image_features_raw_ = self._network.visual_forward_(inputs)
                transf_image_features_raw = transf_image_features_raw_ @ self._network.visual_proj
                transf_image_features_raw_ = transf_image_features_raw_ / transf_image_features_raw_.norm(dim=-1,
                                                                                                          keepdim=True)
                transf_image_features_raw = transf_image_features_raw / transf_image_features_raw.norm(dim=-1,
                                                                                                       keepdim=True)
                outputs = (transf_image_features @ transf_text_features.T)
                outputs_gda = transf_image_features_raw_ @ self._network.W + self._network.b
                outputs_gda = outputs_gda / outputs_gda.norm(dim=-1, keepdim=True)
                outputs_rerank = self._network.rerank(self.des_dict, outputs, transf_image_features_raw, class_to_label,
                                                      self._device, self.args['rerank_top'])
                outputs = outputs_gda * self.args['stat'] + (
                            self.args['rerank'] * outputs_rerank + (1 - self.args['rerank']) * outputs) * (
                                      1 - self.args['stat'])
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]