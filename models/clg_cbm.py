import os
import logging
import numpy as np
from tqdm import tqdm
import random
import torch
import copy
from copy import deepcopy
import clip
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import IncrementalNet,PrototypicalNet,Gateway,CLASS_CONCEPT_MATRIX
from utils.toolkit import count_parameters, target2onehot, tensor2numpy,get_attribute
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from utils.data_manager import FeatureDataset

import time
from utils.data_manager import DataManager


num_workers = 8

class Learner(BaseLearner):
    # for validation only
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = PrototypicalNet(args,pretrained=False,device=self._device)
        self._network = self._network.to(self._device)
        self._old_network = None
        self.batch_size = get_attribute(args,"batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.sim = get_attribute(args, "sim", 1)
        self._known_classes,self._total_classes = 0,0
        self.bottleneck = None
        self.score_dict,self.bottle_dict = {},{}
        self.gateway = Gateway(self.args,self._device).to(self._device)
        self.pre_protos, self.ori_protos, self.cpt_count, self.proto_dict, self.ori_covs = [],[],[],[],[]
        self.cpt_table = [0]
        self.cpt_counter_pos, self.cpt_counter_neg, self.raw_concepts = [], [], []
        self.attr_dict = {}
        self.names = None
#        self.rest_cls = self.data_manager.concept_order
        
    # def get_cls_feat(self,cls,features,labels):

    #     train_feature_dataset = self.data_manager.generate_dataset(range(cls, cls+1),features,labels)
    #     return train_feature_dataset.features.float()
    def _compute_relations(self):
        
        # calculate the similarity matrix of old classes to new classes.
        protos = torch.stack(self.ori_protos[self._known_classes:]).float().to(self._device)
        names_emb = self._network.convnet.encode_text(self.names).float().to(self._device)
        
        protos /= protos.norm(dim=-1, keepdim=True)
        names_emb /= names_emb.norm(dim=-1, keepdim=True)
        
        simi_matrix = names_emb[:self._known_classes] @ protos.T
        self._relations = torch.argmax(simi_matrix, dim=1) + self._known_classes
        self._relations = self._relations.cpu().numpy()#索引
            
    def building_protos(self):
        # prototype construction
        print("building protos...")
        with torch.no_grad():
            train_features,train_labels = self.get_image_embeddings(self.train_loader)
            train_features,train_labels = torch.from_numpy(train_features).float(),torch.from_numpy(train_labels).float()
            for i in range(self._known_classes, self._total_classes):
                index = torch.nonzero(train_labels == i)
                index = index.squeeze()
                class_data = train_features[index]
                cls_mean = class_data.mean(dim=0).to(self._device)
                cls_cov = torch.cov(class_data.t()).to(self._device) + 1e-4* torch.eye(class_data.shape[-1], device=self._device)

                self.ori_protos.append(cls_mean)
                self.ori_covs.append(cls_cov)
                        
    def get_image_embeddings(self,loader):

        with torch.no_grad():
            features = []
            labels = []
            for i, (_,images,targets) in enumerate(loader):
                images, targets = images.to(self._device), targets.to(self._device)
                # images: [batch_size, 3, 224, 224]
                image_features = self._network.extract_vector(images)
                # if self.stage: image_features = self._network.unifier(image_features.float())
                # [batch_size, 768]
                image_features /= image_features.norm(dim=-1, keepdim=True)
                # [batch_size, 768]
                features.append(image_features.cpu())
                labels.append(targets.cpu())
            features = torch.cat(features)
            labels = torch.cat(labels)
        features = np.array(features)
        labels = np.array(labels)

        return features,labels
    
    def cluster(self,concept_cls,num_attributes=None):
        # Attributes selection
        self.stage = 0
        attributes, names, counter = self.data_manager.get_attributes(self.args["dataset"].lower(),concept_cls)
        attribute_embeddings = []
        templates = self.data_manager._data_to_prompt[0]
        # extract text features
        for i in range((len(attributes) // self.batch_size) + 1):
        # Prompting batch by batch
            sub_attributes = attributes[i * self.batch_size: (i + 1) * self.batch_size]
           # if self.args['model_type'] == 'clip':
            clip_attributes_embeddings = self._network.tokenizer([self.data_manager.get_prefix(self.args)+ attr for attr in sub_attributes]).to(self._device)
            attribute_embeddings += [embedding.detach().cpu() for embedding in self._network.convnet.encode_text(clip_attributes_embeddings)]
        class_name_embeddings = self._network.tokenizer([ templates.format(name) for name in names]).to(self._device)
        
        train_features,train_labels = self.get_image_embeddings(self.train_loader)
        test_features,test_labels = self.get_image_embeddings(self.test_loader)
        attribute_embeddings = torch.stack(attribute_embeddings).float()
        attribute_embeddings = attribute_embeddings / attribute_embeddings.norm(dim=-1, keepdim=True) #[N,D]

        print ("num_attributes: ", attribute_embeddings.shape[0])

        print("Clustering...")
        
        mu = torch.mean(attribute_embeddings, dim=0)
        sigma_inv = torch.linalg.inv(torch.cov(attribute_embeddings.T))
        configs = {
            'mu': mu,
            'sigma_inv': sigma_inv,
            'mean_distance': np.mean([self.mahalanobis_distance(embed, mu, sigma_inv).cpu() for embed in attribute_embeddings])
        }
        
        out_dim = self._total_classes - self._known_classes
        self.gateway.gate = self.gateway.generate_gate(self.args["linear_model"],input_dim=attribute_embeddings.shape[-1], output_dim=out_dim,num_attributes=attribute_embeddings.shape[0])
        self.gateway = self.gateway.to(self._device)

        # Feature dataloader, construct Learnt Embeddings
        train_score_dataset = FeatureDataset(train_features, train_labels)
        train_loader = DataLoader(train_score_dataset, batch_size=self.batch_size,drop_last=False,shuffle=True,num_workers=num_workers)
        test_score_dataset = FeatureDataset(test_features, test_labels) 
        test_loader = DataLoader(test_score_dataset, batch_size=self.batch_size,drop_last=False, shuffle=False,num_workers=num_workers)

        best_model = self._train(self.gateway,train_loader, test_loader,regularizer='mahalanobis',configs=configs)
        
        centers = best_model[0].weight.detach().cpu().numpy()
        self.gateway.gate = None # reset
        selected_idxes = []
        print("select {} attributes out of {}".format(num_attributes, len(attribute_embeddings)))
        for center in centers:
            center = center / torch.tensor(center).norm().numpy()
            distances = np.sum((attribute_embeddings.numpy() - center.reshape(1, -1)) ** 2, axis=1)
            # sorted_idxes = np.argsort(distances)[::-1]
            sorted_idxes = np.argsort(distances)
            for elem in sorted_idxes:
                if elem not in selected_idxes:
                    selected_idxes.append(elem)
                    break
        selected_idxes = np.array(selected_idxes[:num_attributes])
        for j in selected_idxes: self.raw_concepts.append(attributes[j])
        return attribute_embeddings[selected_idxes].clone().detach(), class_name_embeddings.clone().detach()

            
    def after_task(self):
        self._known_classes = self._total_classes
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        task_size = self._total_classes - self._known_classes
        self.concept_cls = data_manager.concept_order[self._known_classes:self._total_classes]
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        # Preparing dataset
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes,self._total_classes), 
            source="train", mode="train")
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self._network.to(self._device)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        # Evaluation
        eval_test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.eval_test_loader = DataLoader(eval_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        # bottleneck
        self.pool = int(task_size * self.args["pool"])
        # load bottleneck
        attributes_embeddings,class_names = self.cluster(self.concept_cls,self.pool)  

        self.names  = class_names if self.names == None else torch.cat((self.names,class_names),dim=0) 
        self.bottleneck = attributes_embeddings.to(self._device) if self.bottleneck is None else torch.concat((self.bottleneck,attributes_embeddings.to(self._device)),dim=0)
        
        self.bottle_dict[self._cur_task] = self.bottleneck
        self.cpt_table.append(self.bottleneck.shape[0])

        # Building Prototypes
        self.building_protos()
        
        # Building pseudo features
        if self._cur_task: self._compute_relations()
        self._build_feature_set()
        self.train_loader = DataLoader(self._feature_trainset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        
        self.stage=1
        self._network.update_explainer(self.pool,task_size,bias=False)
        self._network.explainer, self._network.unity = self._train(self._network,self.train_loader,self.test_loader)
        
        self.test_loader = self.eval_test_loader
        
    def mahalanobis_distance(self,x,mu,sigma_inv):
        if self.stage: 
            x -= mu
            mahal = torch.sqrt(x @ sigma_inv @ x.T)
        else: 
            x = x - mu.unsqueeze(0)
            mahal = torch.diag(x @ sigma_inv @ x.T).mean()
        return mahal
    
    def _train(self,model,train_loader,test_loader,regularizer=None,configs=None):
        if self.stage:
            lr = self.init_lr # if self._cur_task == 0 else self.args['FB_lr_inc']
        else: lr = self.args["lr"]
        
        self._epoch_num = self.args["epochs"]
        if self.stage: optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()),lr=lr,weight_decay=0.0000)
        else: optimizer = torch.optim.Adam(model.gate.parameters(),lr=lr )
        
        scheduler = None
        if self.stage: scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args["milestones"], gamma=0.1, last_epoch=-1)
        best_model, best_acc = self._train_function(model,train_loader, test_loader, optimizer,scheduler,regularizer,configs)
        return best_model
    
    def _train_function(self,model,train_loader,test_loader,optimizer,scheduler=None,regularizer=None,configs=None):  
        model.train()
        loss_function = torch.nn.CrossEntropyLoss().to(self._device)

        last_best_acc, best_acc = None, 0
        if self.stage == 0: epochs = self.args["epochs"]
        elif self._cur_task: epochs = self.args["FB_epoch_inc"]
        else: epochs = self.args["FB_epoch"]
        random_class_order_list = list(range(self._known_classes))
        random.shuffle(random_class_order_list)
        best_model = model
        offset = self._known_classes
        start = time.time()
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            losses,correct,total = 0, 0, 0
            for idx, batch in enumerate(train_loader):
                if self.stage == 0: inputs, targets = batch[0],batch[1]
                else: inputs, targets = batch[1],batch[2]
                inputs = inputs.float().to(self._device)
                targets = targets.long().to(self._device)
                if self.stage: # training CBL & Classifier
                    if self._cur_task>0 and self.args['sg_num']: 
                        #生成伪样本
                        sg_inputs, sg_targets = self._sample_gussian(idx,random_class_order_list,self.args['sg_num'])
                        inputs = torch.cat([inputs, sg_inputs], dim=0)
                        targets = torch.cat([targets, sg_targets], dim=0)
                    else: sg_inputs = None

                    logits,CSV = model.forward_explainer(inputs)
                    
                    loss = loss_function(logits,targets) 
                    
                    if self.sim: 
                        loss += self.sim * self._similiarity_loss(inputs, self.bottleneck, CSV)
                    loss += self._sparse_linear_loss(self._network.unity)
                    
                else: # Learning to search concepts
                    targets -= offset
                    logits = model(inputs)
                    loss = loss_function(logits,targets)
                    loss += self._compute_loss(regularizer,configs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _,preds = torch.max(logits,dim=1)
                correct += preds.eq(targets.expand_as(preds)).sum()
                total += len(targets)
            if self.stage == 0: end = time.time()
            else: b_end = time.time()
            train_acc = (correct*100 / total)
            if scheduler: scheduler.step()
         #   if epoch % self.args["print_freq"] == 0 or epoch == epochs - 1: 
            test_accuracy = self._compute_accuracy(model,test_loader,offset) 
            if test_accuracy > best_acc: 
                    best_acc = test_accuracy
                    if self.stage == 0: best_model = copy.deepcopy(model.gate)  
                    else: best_model = [copy.deepcopy(model.explainer), copy.deepcopy(model.unity)]
                
                # logging.info('task: %d, epoch:%d, train loss:%.6f,train accuracy: %.5f,test_accuracy:%.5f'  
                #              % (self._cur_task, epoch, losses/len(train_loader), train_acc, test_accuracy))
            info='task:{}, Epoch {}=> Loss {:.3f},Train_acc {:.2f},Test_acc {:.2f}'.format(self._cur_task, epoch + 1, losses/len(train_loader), train_acc, test_accuracy)
                # info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                # self._cur_task,epoch + 1,self.args['tuned_epoch'],losses / len(train_loader),train_acc, test_acc,  )
            prog_bar.set_description(info)
            if last_best_acc is not None and best_acc == last_best_acc and not self.stage:
                    print("early stop")
                    break
            last_best_acc = best_acc
        return best_model, best_acc
    
    def _similiarity_loss(self, inputs, bottleneck, csv, sg=None, cpt_targets=None):
        # concept alignment
        if len(inputs.shape) > 2: img_feats = self._network.extract_vector(inputs.to(self._device)).float()
        else: img_feats = inputs.float()
        distance_loss = nn.MSELoss()
        if sg is not None: img_feats = torch.cat((img_feats,sg),dim=0).float()
        target_feats = img_feats @ bottleneck.T 

        target_feats, csv = target_feats**3, csv**3
        target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)
        csv = csv / torch.norm(csv, p=2, dim=0, keepdim=True)
        similarities = torch.sum(csv * target_feats, dim=0)
        
        return -similarities.mean() 
                    
    def _sparse_linear_loss(self,unity):
        weight_mat = torch.cat([unity[i].weight for i in range(len(unity))]).view(self._total_classes,-1)
        lam = 0.001
        alpha = 0.99
        loss = lam * alpha * weight_mat.norm(p=1) + 0.5 * lam * (1-alpha) * (weight_mat**2).sum()
        # loss+=self.gamma * ((self.classifier.weight-self.original_model_weight)**2).sum()+self.gamma*((self.classifier.bias-self.original_model_bias)**2).sum()    
        return loss
                       

    def _sample_gussian(self,batch_id,random_class_order_list,sg_num):
        sg_inputs = []
        sg_targets = []
        
        list_for_one_batch = [random_class_order_list[batch_id*2%len(random_class_order_list)], random_class_order_list[(batch_id*2+1)%len(random_class_order_list)]]
        for i in list_for_one_batch:
            sg_inputs.append(self._sample(self.ori_protos[i], self.ori_covs[i],int(sg_num), shrink=False))
            sg_targets.append(torch.ones(int(sg_num), dtype=torch.long, device=self._device)*i)
        sg_inputs = torch.cat(sg_inputs, dim=0)
        sg_targets = torch.cat(sg_targets, dim=0)
        
        return sg_inputs, sg_targets
            
    def _shrink_cov(self,cov):
        diag_mean = torch.mean(torch.diagonal(cov))
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        mask = off_diag != 0.0
        off_diag_mean = (off_diag*mask).sum() / mask.sum()
        iden = torch.eye(cov.shape[0], device=cov.device)
        alpha1 = 1
        alpha2  = 1
        cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
        return cov_
        
    def _sample(self,mean, cov, size, shrink=False):
        vec = torch.randn(size, mean.shape[-1], device=self._device)
        if shrink:
            cov = self._shrink_cov(cov)
        sqrt_cov = torch.linalg.cholesky(cov)
        vec = vec @ sqrt_cov.t()
        vec = vec + mean
        return vec
    
    def _build_feature_set(self):
        vectors_train = []
        labels_train = []

        print("constructing pseudo features...")
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train', mode='train', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, labels = self.get_image_embeddings(idx_loader)
            vectors_train.append(vectors)
            labels_train.append([class_idx]*len(vectors))

            
        for class_idx in range(0,self._known_classes):
            new_idx = self._relations[class_idx]
            vectors_train.append(vectors_train[new_idx-self._known_classes]-self.ori_protos[new_idx].cpu().numpy()+self.ori_protos[class_idx].cpu().numpy())
            labels_train.append([class_idx]*len(vectors_train[-1])) 

        vectors_train = np.concatenate(vectors_train)
        labels_train = np.concatenate(labels_train)
        
        self._feature_trainset = Pesudo_FeatureDataset(vectors_train,labels_train)

    
    def _compute_loss(self,regularizer,configs):
        if regularizer == 'mahalanobis':
            mahalanobis_loss = (self.mahalanobis_distance(self.gateway.gate[0].weight/self.gateway.gate[0].weight.data.norm(dim=-1, keepdim=True), configs['mu'].to(self._device),configs['sigma_inv'].to(self._device)) - configs['mean_distance']) / (configs['mean_distance']**self.args['division_power'])
            return torch.abs(mahalanobis_loss)
        elif regularizer == 'cosine':
            weight = self.gateway.gate[0].weight/self.gateway.gate[0].weight.data.norm(dim=-1, keepdim=True)
            return self.args['lambda'] * torch.sum((weight - configs['mu'].unsqueeze(0).to(self._device)) ** 2, dim=-1).mean()
        else :
            return 0
    
    def _compute_accuracy(self,model,test_loader,offset=None):
        model.eval()
        with torch.no_grad():
            predictions = []
            labels = []
            for idx, batch in enumerate(test_loader):
                results = None
                if self.stage == 0: inputs,targets = batch[0],batch[1]
                else: inputs,targets = batch[1],batch[2]
                inputs = inputs.float().to(self._device)
                if self.stage:
                    logits,CSV = model.forward_explainer(inputs)
                else:
                    logits = model.gate(inputs)
                pred = torch.argmax(logits, dim=-1)
                predictions.append(pred)
                labels.append(targets)
            predictions = torch.cat(predictions)
            labels = torch.cat(labels).to(self._device)
            if self.stage == 0: labels -= offset
        acc = (torch.sum(predictions == labels) / len(predictions) * 100)
        return acc
    
    def _eval_cnn(self,loader):
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[1].float().to(self._device)
                targets = batch[2].long().to(self._device)
                result = None

                logits,CSV = self._network.forward_explainer(inputs)
                
                predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]

                y_pred.append(predicts.cpu().numpy())
                y_true.append(targets.cpu().numpy())
        
        return np.concatenate(y_pred), np.concatenate(y_true)  
        
    def eval_task(self, save_conf=False):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if save_conf:
            _pred = y_pred.T[0]
            _pred_path = os.path.join(self.args['logfilename'], "pred.npy")
            _target_path = os.path.join(self.args['logfilename'], "target.npy")
            np.save(_pred_path, _pred)
            np.save(_target_path, y_true)

            _save_dir = os.path.join(f"./results/conf_matrix/{self.args['prefix']}")
            os.makedirs(_save_dir, exist_ok=True)
            _save_path = os.path.join(_save_dir, f"{self.args['csv_name']}.csv")
            with open(_save_path, "a+") as f:
                f.write(f"{self.args['time_str']},{self.args['model_name']},{_pred_path},{_target_path} \n")

        return cnn_accy, nme_accy, None, None, None, None
        
from PIL import Image
class Pesudo_FeatureDataset(Dataset):
    def __init__(self, features, labels, raw=None, use_path=False, trsf=None):
        assert len(features) == len(labels), "Data size error!"
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)
        self.raw = raw
        self.use_path = use_path
        self.trsf = trsf
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.raw is not None:
            if self.use_path:
                raw = Image.open(self.raw[idx]).convert("RGB")
                raw = self.trsf(raw)
                # image = self.trsf(pil_loader(self.images[idx]))
            else:
                raw = self.trsf(Image.fromarray(self.raw[idx]))
        
            return idx, feature, label, raw
        else: return idx, feature, label
