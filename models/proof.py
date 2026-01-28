import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import  Proof_Net
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
from utils.data_manager import LaionData
import math
import matplotlib.pyplot as plt
import os


num_workers = 8
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args=args

        self._train_transformer=False
        self._network = Proof_Net(args, False)
        
        self.batch_size = get_attribute(args,"batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        
        self._known_classes = 0
        self.use_cos = get_attribute(args, "use_cos", False)

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
    
    def cal_prototype(self,trainloader, model):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.to(self._device)
                label=label.to(self._device)
                embedding=model.convnet.encode_image(data, True)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = list(range(self._known_classes, self._total_classes))
        for class_index in class_list:
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            self._network.img_prototypes[class_index]=proto

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        
        self._network.update_prototype(self._total_classes)
        self._network.update_context_prompt() # add context prompts

        self._network.extend_task()
        
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
            source="train", mode="train", appendent=self._get_memory())
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self._network.to(self._device)
       
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test" )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self.cal_prototype(self.train_loader_for_protonet, self._network)
        self._train_proj(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
    
    def _train_proj(self, train_loader, test_loader, train_loader_for_protonet):
        self._train_transformer=True
        self._network.to(self._device)
       
        for name, param in self._network.convnet.named_parameters():
            if 'logit_scale' not in name:
                param.requires_grad = False
      #  for p in self._network.convnet.parameters():
       #     p.requires_grad = True
        self._network.freeze_projection_weight_new()
        
        if self.args['optimizer']=='sgd':
            optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer']=='adam': 
            optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)

        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        prog_bar = tqdm(range(self.tuned_epoch))
        cliploss = ClipLoss()

        total_labels = class_to_label[:self._total_classes] # mask all known classes
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                labels = [class_to_label[y] for y in targets]
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                
                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features = self._network.encode_text(texts) # [total_classes, dim]
                text_feas = text_features / text_features.norm(dim=-1, keepdim=True)
                image_features = self._network.encode_image(inputs)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True) #[bs, dim]
                image_features, text_features, logit_scale, proto_feas=self._network.forward_transformer(img_feas, text_feas,self._train_transformer)
                logits = image_features@text_features.T # [bs, allclasses]

                texts=[templates.format(inst) for inst in labels]
                clip_text_feas=self._network.encode_text(self._network.tokenizer(texts).to(self._device))#total,dim
                clip_text_norm=clip_text_feas.norm(dim=-1, keepdim=True)
                clip_text_feas = clip_text_feas / clip_text_norm

                clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)

                loss = F.cross_entropy(logits, targets)

                protoloss = F.cross_entropy(image_features @ proto_feas.T, targets)

                total_loss = loss+clip_loss+protoloss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses += total_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task,epoch + 1,self.args['tuned_epoch'],losses / len(train_loader),train_acc, test_acc,  )
            prog_bar.set_description(info)


    def _compute_accuracy(self, model, loader):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
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
                image_features=self._network.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = self._network.forward_transformer(image_features, text_features,self._train_transformer)
                outputs = transf_image_features @ transf_text_features.T
                proto_outputs= transf_image_features @ proto_feas.T
                original_outputs= image_features @ text_features.T
                outputs = original_outputs+outputs+proto_outputs

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


    def _eval_cnn(self, loader):
        
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features=self._network.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = self._network.forward_transformer(image_features, text_features,self._train_transformer)

                outputs = transf_image_features @ transf_text_features.T

                proto_outputs= transf_image_features @ proto_feas.T

                original_outputs= image_features @ text_features.T

                outputs = original_outputs+outputs+proto_outputs

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


