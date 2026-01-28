import os
import json
import pdb
import logging
import torch
import statistics
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans
import copy
from utils.toolkit import target2onehot, tensor2numpy, get_attribute, ClipLoss
from utils.inc_net import MgclipNet
from models.base import BaseLearner
from backbone.mind_model import VisionClassifier
from torch import nn
num_workers = 8
import clip
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MgclipNet(args, False)
        self.batch_size = get_attribute(args, "batch_size", 64)
        self.increment = get_attribute(args, "increment", 10)
        self.initial_increment = get_attribute(args, "initial_increment", 10)
        
        self.args = args
        self.reset = get_attribute(args, "reset", False)
        self.only_reset_B = get_attribute(args, "only_reset_B", False)
        self.freeze_A = get_attribute(args, "freeze_A", False)
        self.current_class_names = []
        # self._cur_task = -1
        self.task_num = get_attribute(args, "task_num", 10)
        self.init_lr = get_attribute(args, "init_lr", 0.001)
        self.epochs = get_attribute(args, "epochs", 10)     

        self.visual_clsf = get_attribute(args, "visual_clsf", False)
        self.visual_clsf_batch_size = get_attribute(args, "visual_clsf_batch_size", 64)
        self.visual_clsf_epochs = get_attribute(args, "visual_clsf_epochs", 10)
        self.visual_clsf_lr = get_attribute(args, "visual_clsf_lr", 0.0001)

        self.all_test = get_attribute(args, "all_test", False)

        self.real_replay = get_attribute(args, "real_replay", False)
        self.balance_ft = get_attribute(args, "balance_ft", False)
        self.balance_epochs = get_attribute(args, "balance_epochs", 10)

        trainable_params = {k: v for k, v in self._network.named_parameters() if v.requires_grad}
        # pdb.set_trace()
        torch.save(trainable_params, f'ori_params.pth')


    def after_task(self):
        self._known_classes = self._total_classes
    
    def adaptation(self, reset=False):
        # self.current_task +=1 
        if reset and self._cur_task >0:
            ori_state = torch.load('ori_state.pth')
            if self.only_reset_B:
                now_state = self._network.model.state_dict()
                lora_params = {k: v for k, v in ori_state.items() if 'lora_B' in k}
                now_state.update(lora_params)
            else:
                now_state = ori_state
            self._network.model.load_state_dict(now_state)

        if self.freeze_A and self._cur_task >0:
            for name, param in self._network.model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = False


        self.current_class_names = self.classnames[:self._total_classes]
        self.current_task_class_names = self.classnames[self._known_classes:self._total_classes]

        self._network.text_tokens = clip.tokenize(
            ["a good photo of a {}.".format(c) for c in self.current_class_names]
        ).to(self._device)
        self.current_task_text_tokens = clip.tokenize(
            ["a good photo of a {}.".format(c) for c in self.current_task_class_names]
        ).to(self._device)

        if self._cur_task == 0:
            self.all_class_names = self.classnames
            self._network.all_text_tokens = clip.tokenize(
                ["a good photo of a {}." for c in self.all_class_names]
            ).to(self._device)
            
        # self.text_tokens = self._network.tokenizer(
        #     ["a good photo of a {}.".format(c) for c in self.current_class_names]
        # ).to(self._device)
        # self.current_task_text_tokens = self._network.tokenizer(
        #     ["a good photo of a {}.".format(c) for c in self.current_task_class_names]
        # ).to(self._device)

        # if self._cur_task == 0:
        #     self.all_class_names = self.classnames
        #     self.all_text_tokens = self._network.tokenizer(
        #         ["a good photo of a {}." for c in self.all_class_names]
        #     ).to(self._device)


    def modality_gap(self, loader):

        trainable_params = torch.load(f'ori_params.pth')
        self._network.load_state_dict(trainable_params, strict=False)
        self._network.eval()
        positive_outputs = []
        negative_outputs = []

        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(loader):
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                outputs = self._network(inputs)
                one_hot_targets = torch.nn.functional.one_hot(targets, outputs.shape[1]).float()
                positive_outputs.append((outputs * one_hot_targets).sum(dim=1).mean())
                mask = 1 - one_hot_targets
                negative_outputs.append(((outputs * mask).sum(dim=1) / mask.sum(dim=1)).mean())

        positive_mean = sum(positive_outputs) / len(positive_outputs)
        negative_mean = sum(negative_outputs) / len(negative_outputs)
        self.negative_records = negative_mean
        # if task_id == 0:
        logit_size = self.increment if self._cur_task > 0 else self.initial_increment

        bias_logit = torch.full((logit_size,), negative_mean, device=self._device)
        bias_logit[0] = positive_mean
        logging.info(f"positive_records: {positive_mean}")
        logging.info(f"negative_records: {negative_mean}")

    def intra_cls(self,logits, y, classes):
        y = y - classes
        logits1 = logits[:, classes:]
        return F.cross_entropy(logits1, y, reduction='none')

    def get_finetuning_dataset(self,finetuning='balanced', oversample_old=1):
        if finetuning == 'balanced':
            x_mem, y_mem = self._get_memory()
            

            if oversample_old > 1:
                x_mem = np.repeat(x_mem, oversample_old , axis=0)
                y_mem = np.repeat(y_mem, oversample_old , axis=0)
            # dataset._x, dataset._y = self.train_dataset

            # x_combined = np.concatenate([x_mem, dataset._x], axis=0)
            # y_combined = np.concatenate([y_mem, dataset._y], axis=0)


        return(x_mem,y_mem)


    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    #    print(self._known_classes, self._total_classes)
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train",appendent=self._get_memory() )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.classnames = self.data_manager._class_to_label
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
   
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test" )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self.visual_loader = DataLoader(train_dataset_for_protonet, batch_size=self.visual_clsf_batch_size, shuffle=True, num_workers=num_workers)

        if self.visual_clsf and self._cur_task==0:
                self.vision_clsf = VisionClassifier(512, self.increment, args= self.args,activation=None)
                # self.vision_clsf(self._device)

        self.adaptation(reset=self.reset)
        self._network.to(self._device)
        
        trainable_params = {k: v for k, v in self._network.named_parameters() if v.requires_grad}
        torch.save(trainable_params, f'trainable_params.pth')

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self.modality_gap(self.train_loader_for_protonet)
        
        self._train(self.train_loader, self.test_loader,self.train_loader_for_protonet)
        self._train_visual(self.visual_loader)
        if self.real_replay:
            self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):

        trainable_params = torch.load(f'trainable_params.pth',weights_only=True)
        self._network.load_state_dict(trainable_params, strict=False)
        
        
        params = filter(lambda p: p.requires_grad, self._network.parameters())

        optimizer = torch.optim.Adam(params, lr=self.init_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=self.init_lr*0.01)

        if self._cur_task == 0:
            self.targets_bais = 0
        else:
            self.targets_bais = self.initial_increment + (self._cur_task - 1) * self.increment

        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
                self._network.train()
                for i, (_,inputs, targets) in enumerate(train_loader):
                    loss_c = torch.tensor(0.0).to(self._device)
                    loss = torch.tensor(0.0).to(self._device)

                    replay_loss = torch.tensor(0.0).to(self._device)

                    inputs = inputs.to(self._device)
                    targets = targets.to(self._device)
                   
                    outputs =  self._network(inputs)
                    if self._cur_task > 0:
                        if self.real_replay:
                            mask_replay = (targets < self.targets_bais)
                            old_targets = targets[mask_replay].clone()
                            old_outputs = outputs[mask_replay].clone()
                            targets = targets[~mask_replay]
                            outputs = outputs[~mask_replay]
                            replay_loss = self.intra_cls(old_outputs, old_targets, 0).mean()*0.1
                        loss_c = self.intra_cls(outputs,targets,self.targets_bais).mean() + replay_loss
                        pass
                    else:
                        loss_c = torch.nn.functional.cross_entropy(outputs, targets) 
                    loss += loss_c
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                if self._cur_task ==0:
                    positive_outputs = []
                    negative_outputs = []
                    with torch.no_grad():
                        self._network.eval()
                        for i, (_, inputs, targets) in enumerate(train_loader_for_protonet):
                            inputs = inputs.to(self._device)
                            targets = targets.to(self._device)
                            outputs =  self._network(inputs)
                            one_hot_targets = torch.nn.functional.one_hot(targets, outputs.shape[1]).float()
                            positive_outputs.append((outputs * one_hot_targets).sum(dim=1).mean())
                            mask = 1 - one_hot_targets
                            negative_outputs.append(((outputs * mask).sum(dim=1) / mask.sum(dim=1)).mean())

                        self._network.train()
                    positive_mean = sum(positive_outputs) / len(positive_outputs)
                    negative_mean = sum(negative_outputs) / len(negative_outputs)
                    all_mean = (sum(positive_outputs)+ sum(positive_outputs))/ (len(positive_outputs)+len(negative_outputs))

                    logging.info(f"positive_mean: {positive_mean}")
                    logging.info(f"negative_mean: {negative_mean}")
                    if (abs(self.negative_records - negative_mean)/self.negative_records)>0.1:
                        if epoch > 0:
                            logging.info(f"Negative records changed too much, epoch {epoch}")
                            self.epochs = epoch
                        else:
                            logging.info(f"Negative records changed too much, epoch 1")
                            self.epochs = 1
                        break
                        

        
    
        if self.balance_ft and self.real_replay and self._cur_task > 0:

            balance_data = self.get_finetuning_dataset( 'balanced',1)
            balance_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train",appendent= balance_data)
            balance_loader = DataLoader(balance_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

            epochs = self.balance_epochs

            params = filter(lambda p: p.requires_grad, self._network.parameters())
            optimizer = torch.optim.Adam(params, lr=self.lr*0.01) 
            # optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)  
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=self.lr*0.001)  

            prog_bar = tqdm(range(epochs))
            for _, epoch in enumerate(prog_bar):
                for i, (_,inputs, targets) in enumerate(balance_loader):
                    loss_c = torch.tensor(0.0).to(self._device)
                    loss = torch.tensor(0.0).to(self._device)

                    replay_loss = torch.tensor(0.0).to(self._device)

                    inputs = inputs.to(self._device)
                    targets = targets.to(self._device)
                    outputs =  self._network(inputs)
                    # image_f, text_f = model(inputs, return_feature=True)
                    loss_c = torch.nn.functional.cross_entropy(outputs, targets)
                    loss += loss_c
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()

    def _train_visual(self, visual_loader):
        if self.visual_clsf:
            self._network.eval()
            e_num = self.visual_clsf_epochs
            
            features_dict = {}

            with torch.no_grad():
                for i, (_, inputs, targets) in enumerate(visual_loader):
                    inputs = inputs.to(self._device)
                    targets = targets.to(self._device)
                    _, features, __ = self._network(inputs, test=True, return_feature=True)
                    for feature, target in zip(features, targets):
                        target = target.item()
                        if target not in features_dict:
                            features_dict[target] = []
                        features_dict[target].append(feature.cpu())
            mean_features = []
            for target in sorted(features_dict.keys()):
                features = torch.stack(features_dict[target])
                mean_feature = features.mean(dim=0)
                mean_features.append(mean_feature.unsqueeze(0))
            mean_features = torch.cat(mean_features).to(self._device)


            if self._cur_task > 0:
                self.vision_clsf.add_weight(mean_features)
                pass
            else:
                self.vision_clsf.set_weight(mean_features)
                pass

            optimizer = torch.optim.Adam(self.vision_clsf.parameters(), lr=self.visual_clsf_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, e_num*len(visual_loader), eta_min=self.visual_clsf_lr*0.01)
            prog_bar = tqdm(range(e_num))
            for _, e in enumerate(prog_bar):
                for i, (_,inputs, targets) in enumerate(visual_loader):
                
                    inputs = inputs.to(self._device)
                    targets = targets.to(self._device)
                    # pdb.set_trace()
                    with torch.no_grad():
                        outputs, _ = self._network(inputs, return_feature=True)
                    # pdb.set_trace()

                    outputs = self.vision_clsf(outputs)
                    

                    # pdb.set_trace()
                    loss = self.intra_cls(outputs,targets,self.targets_bais).mean()
                    # loss = F.cross_entropy(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            if self.balance_ft and self.real_replay and self._cur_task > 0:

                balance_data = self.get_finetuning_dataset( 'balanced',1)
                balance_loader = DataLoader(balance_data, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

                epochs = self.balance_epochs

                optimizer = torch.optim.Adam(self.vision_clsf.parameters(), lr=self.visual_clsf_lr*0.1)
                # optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*len(balance_loader), eta_min=self.lr*0.01)
                prog_bar = tqdm(range(epochs))
                for _, epoch in enumerate(prog_bar):
                    for i, (_,inputs, targets) in enumerate(balance_loader):
                        loss_c = torch.tensor(0.0).to(self._device)
                        loss = torch.tensor(0.0).to(self._device)

                        replay_loss = torch.tensor(0.0).to(self._device)
                        torch.cuda.empty_cache()

                        inputs = inputs.to(self._device)
                        targets = targets.to(self._device)
                        with torch.no_grad():
                            outputs, _ = self._network(inputs, return_feature=True)
                        # pdb.set_trace()
                        outputs = self.vision_clsf(outputs)
                        loss_c = torch.nn.functional.cross_entropy(outputs, targets)
                        loss += loss_c
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()


    def _eval_cnn(self, loader):


        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):

            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            
            with torch.no_grad():
                if self.visual_clsf:
                    a = 1
                    b = 4
                    
                    outputs, image_feature, text_feature  = self._network(inputs, test=True, all_test=self.all_test, return_feature=True)
                    vision_outputs = self.vision_clsf(image_feature)

                    outputs_softmax = F.softmax(outputs, dim=1)
                    vision_outputs_softmax = F.softmax(vision_outputs, dim=1)
                    
                    outputs = (a*outputs_softmax + b*vision_outputs_softmax) / (a + b)      
                else:
                    outputs = self._network(inputs, test=True, all_test=self.all_test)

                predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]

            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]