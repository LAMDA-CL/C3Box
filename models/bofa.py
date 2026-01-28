import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from utils.inc_net import BofaAdapter
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute
import random
random.seed(1993)
np.random.seed(1993)

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self._train_transformer = False
        self._network = BofaAdapter(args)
        self._network.eval()

        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)
        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)
        self.stage2_epoch = get_attribute(args, "epoch", 2)
        self._known_classes = 0
        self.prototype = []
        self.loss_type = get_attribute(args, "loss_type", "CE")
        # last_mask
        self.last_mask = get_attribute(args, "last_mask", False)
        self.use_up_cen = get_attribute(args, "use_up_cen", False)
        self.center_type = get_attribute(args, "center_type", "mix")
        self.t_lam = 0
        self.stat = args['stat']
        self.label2task = []
        self.train_loader_list = []
        self.test_loader_list = []
        self.first_task = True

    def after_task(self):
        self._known_classes = self._total_classes

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
        test_dataset_task = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                     source="test", mode="test")
        train_eval_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                      source="train", mode="test")

        train_eval_loader = DataLoader(train_eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.train_loader_list.append(train_eval_loader)
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self._network.to(self._device)
        cur_label2task = [self._cur_task] * (self._total_classes - self._known_classes)
        self.label2task = self.label2task + cur_label2task
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader_task = DataLoader(test_dataset_task, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader_list.append(self.test_loader_task)
        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self._network.update_stat(self._known_classes, self._total_classes, self.train_loader, self._device)
        self.init_accuracy(self.train_loader, self.test_loader_task, self.test_loader)
        # self._network.update_task(self._total_classes - self._known_classes)
        self._network.start_train(self._total_classes - self._known_classes)
        self.train(self.train_loader, self.test_loader, train_dataset)
        self._network.end_train()

    def eval_init(self, eval_loader, text_proto):
        text_correct, all_num = 0, 0
        for i, (_, inputs, targets) in enumerate(eval_loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features, logits, origin_image_features = self._network.encode_image(inputs, return_origin=True)
                origin_image_features = origin_image_features / origin_image_features.norm(dim=-1, keepdim=True)
                text_outputs = (origin_image_features @ text_proto.T)
                text_pred = torch.max(text_outputs, dim=1)[1].cpu()
                text_correct += text_pred.eq(targets).cpu().sum()
                all_num += len(targets)
        return np.around(tensor2numpy(text_correct) * 100 / all_num, decimals=2)

    @torch.no_grad()
    def search_lambda_for_prompt(self, eval_loader, image_proto, text_proto, num_grid: int = 21):
        image_proto = image_proto / image_proto.norm(dim=-1, keepdim=True)  # [C, D]
        text_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)  # [C, D]

        all_feats, all_labels = [], []
        for _, imgs, labels in eval_loader:
            imgs = imgs.to(self._device)
            feats, _, _ = self._network.encode_image(imgs, return_origin=True)  # feats: [B, D]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu())
            all_labels.append(labels.cpu())
        all_feats = torch.cat(all_feats,  dim=0)   # [N, D]
        all_labels = torch.cat(all_labels, dim=0)   # [N]

        best_acc, best_lam, best_proto = -1.0, 0.0, None
        lambdas = torch.linspace(0, 1, steps=num_grid)

        for lam in lambdas:
            new_proto = (1 - lam) * image_proto + lam * text_proto   # [C, D]
            logits = all_feats @ new_proto.T.cpu()   # [N, C]
            pred = logits.argmax(dim=1)
            acc = (pred == all_labels).float().mean().item()      # 0~1

            if acc > best_acc:
                best_acc = acc
                best_lam = lam.item()
                best_proto = new_proto.clone()

        print(f"\n>>> best λ = {best_lam:.3f}")
        return best_lam

    def init_accuracy(self, train_loader, test_new_loader, test_loader):
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        labels = [class_to_label[y] for y in range(self.args['init_cls'] + self._cur_task * self.args['increment'])]
        texts = [templates.format(inst) for inst in labels]
        texts = self._network.tokenizer(texts).to(self._device)
        self.text_proto = self._network.encode_text(texts)
        image_proto = self._network.get_cls_center()
        if self.t_lam == 0:
            self.t_lam = self.search_lambda_for_prompt(train_loader, image_proto, self.text_proto)
        new_proto = image_proto / image_proto.norm(dim=-1, keepdim=True) * (1 - self.t_lam) + \
            self.text_proto / self.text_proto.norm(dim=-1, keepdim=True) * self.t_lam
        test_acc_lam = self.eval_init(test_loader, new_proto)
        logging.info("Eval Test Loader: Zero_Shot_Lam: {:.2f}".format(test_acc_lam))

    def train(self, train_loader, test_loader, train_dataset):
        self._network.to(self._device)
        param_groups = self._network.get_param_group()
        lr = self.init_lr
        weight_decay = self.weight_decay
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params=param_groups, momentum=0.9, lr=lr, weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params=param_groups, lr=self.init_lr, weight_decay=0.001)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [4, 10], gamma=0.1, last_epoch=-1)

        class_to_label = self.data_manager._class_to_label
        prog_bar = tqdm(range(self.tuned_epoch+self.stage2_epoch))
        templates = self.data_manager._data_to_prompt[0]
        if self._cur_task > 0:
            old_class = list(range(self.args['init_cls'] + (self._cur_task - 1) * self.args['increment']))
        from utils.toolkit import ClipLoss
        cliploss = ClipLoss(img_only=True)
        text_proto = self.text_proto
        img_proto = self._network.get_cls_center()
        for _, epoch in enumerate(prog_bar):
            self._network.train_state()
            self._network.train()
            losses = 0.0
            loss_low = 0.0
            loss_clip = 0.0
            correct, total = 0, 0
            if epoch == self.tuned_epoch and self._cur_task > 0:
                self._network.prepare_stage2()
            for i, (_, inputs, targets) in enumerate(train_loader):
                if self.use_up_cen:
                    new_proto = self._network.get_cls_center_last()
                    img_proto = 0.95 * img_proto + 0.05 * new_proto
                self.img_proto = img_proto
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                if self._cur_task > 0:
                    offset_targets = targets - self.args['init_cls'] - (self._cur_task - 1) * self.args['increment']
                else:
                    offset_targets = targets
                logit_scale = self._network.model.logit_scale

                if epoch >= self.tuned_epoch and self._cur_task > 0:
                    image_features, low_logits = self._network.encode_image(inputs, stage2=True, return_origin=False)
                else:
                    image_features, low_logits = self._network.encode_image(inputs, return_origin=False)
                low_logits = low_logits[-1]
                if epoch < 6:
                    low_loss = nn.functional.cross_entropy(low_logits, offset_targets)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True)
                if self.loss_type == "CE":
                    if self.center_type == "img":
                        cls_proto = img_proto / img_proto.norm(dim=-1, keepdim=True)
                    elif self.center_type == "text":
                        cls_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)
                    else: 
                        cls_proto = self.t_lam * (img_proto / img_proto.norm(dim=-1, keepdim=True)) + \
                            (1 - self.t_lam) * text_proto / text_proto.norm(dim=-1, keepdim=True)
                    logits = self._network.model.logit_scale * img_feas @ cls_proto.t()
                    clip_loss = nn.functional.cross_entropy(logits, targets)
                else:
                    labels = [class_to_label[y] for y in targets]
                    texts_clip = [templates.format(inst) for inst in labels]
                    clip_text_feas = self._network.encode_text(self._network.tokenizer(texts_clip).to(self._device))
                    clip_text_norm = clip_text_feas.norm(dim=-1, keepdim=True)
                    clip_text_feas = clip_text_feas / clip_text_norm
                    clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)
                if epoch < 6:
                    loss = low_loss + clip_loss
                else:
                    loss = clip_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                loss_low += low_loss.item()
                loss_clip += clip_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Task {}, Epoch {}/{} => Loss_Clip {:.3f}, Train_acc {:.2f}".format(
                self._cur_task, epoch + 1, self.args['tuned_epoch'], loss_clip / len(train_loader), train_acc)
            logging.info(info)
            prog_bar.set_description(info)

    def _compute_accuracy(self, model, loader, epoch=0):
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]  # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).cuda()
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings.mean(dim=0)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)
        text_proto = text_features.to(self._device)

        img_proto = self.img_proto
        cls_proto = self.t_lam * (img_proto / img_proto.norm(dim=-1, keepdim=True)) + \
            (1 - self.t_lam) * text_proto / text_proto.norm(dim=-1, keepdim=True)
        cls_proto2 = self._network.get_cls_center_lora()
        cls_proto2 = cls_proto2 / cls_proto2.norm(dim=-1, keepdim=True)
        correct, correct_2, total = 0, 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            if self._cur_task > 0:
                offset_targets = targets - self.args['init_cls'] - (self._cur_task - 1) * self.args['increment']
            else:
                offset_targets = targets
            inputs = inputs.to(self._device)
            with torch.no_grad():
                transf_image_features, logits = self._network.encode_image(inputs)
                logits = logits[-1]
                transf_image_features = transf_image_features / transf_image_features.norm(dim=-1, keepdim=True)
                outputs = (transf_image_features @ cls_proto.T)

            predicts = torch.max(logits, dim=1)[1]
            predicts2 = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == offset_targets).sum()
            correct_2 += (predicts2.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2), np.around(tensor2numpy(correct_2) * 100 / total, decimals=2)

    def ens_result(self, logits_list, origin_predicts, outputs_img, outputs_gda):
        class_drift = [0]
        for i in range(len(logits_list)):
            class_drift.append(class_drift[i] + logits_list[i].shape[1])
        logits_sum = torch.cat(logits_list, dim=1)
        best_class_indices = [torch.argmax(logits_list[i], dim=1) + class_drift[i] for i in range(len(logits_list))]
        best_class_indices = torch.stack(best_class_indices, dim=1)
        selected_logits = torch.gather(origin_predicts, 1, best_class_indices)
        final_predicts = torch.argmax(selected_logits, dim=1)
        final_predicts = best_class_indices[torch.arange(best_class_indices.size(0)), final_predicts]

        selected_logits_img = torch.gather(outputs_img, 1, best_class_indices)
        final_predicts_img = torch.argmax(selected_logits_img, dim=1)
        final_predicts_img = best_class_indices[torch.arange(best_class_indices.size(0)), final_predicts_img]

        outputs_gda = self.stat * outputs_gda + (1 - self.stat) * outputs_img
        selected_logits_gda = torch.gather(outputs_gda, 1, best_class_indices)
        final_predicts_gda = torch.argmax(selected_logits_gda, dim=1)
        final_predicts_gda = best_class_indices[torch.arange(best_class_indices.size(0)), final_predicts_gda]
        return logits_sum, final_predicts, final_predicts_img, final_predicts_gda

    def ens_two_stage(self, best_class_indices, outputs):
        selected_logits = torch.gather(outputs, 1, best_class_indices)
        final_predicts = torch.argmax(selected_logits, dim=1)
        final_predicts = best_class_indices[torch.arange(best_class_indices.size(0)), final_predicts]
        return final_predicts

    def get_ens_result(self, logits_list, out_update, out_gda):
        class_drift = [0]
        for i in range(len(logits_list)):
            class_drift.append(class_drift[i] + logits_list[i].shape[1])
        best_class_indices = [torch.argmax(logits_list[i], dim=1) + class_drift[i] for i in range(len(logits_list))]
        best_class_indices = torch.stack(best_class_indices, dim=1)

        out_ens_gda = self.stat * out_gda + (1 - self.stat) * out_update
        out_ens_gda = self.ens_two_stage(best_class_indices, out_ens_gda)
        out_ens = self.ens_two_stage(best_class_indices, out_update)

        return out_ens, out_ens_gda

    def gda_pred(self, inputs):
        transf_image_features_raw_ = self._network.visual_forward_(inputs)
        transf_image_features_raw_ = transf_image_features_raw_ / transf_image_features_raw_.norm(dim=-1, keepdim=True)
        outputs_gda = transf_image_features_raw_ @ self._network.W + self._network.b
        return outputs_gda

    def get_result(self, transf_image_features, cls_proto, inputs, logits):
        transf_image_features = transf_image_features / transf_image_features.norm(dim=-1, keepdim=True)
        out_update = (transf_image_features @ cls_proto.T)
        out_gda = self.gda_pred(inputs)
        out_pred, out_pred_gda = self.get_ens_result(logits, out_update, out_gda)
        out_argmax = torch.argmax(out_update, dim=1)
        return out_pred, out_pred_gda,out_argmax

    def _eval_cnn(self, loader):
        self._network.to(self._device)
        self._network.eval()

        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]


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
        text_proto = torch.stack(text_features, dim=0)  # [C, D]


        img_proto = self._network.get_cls_center_lora()
        if self.center_type == "img":
            cls_proto = img_proto / img_proto.norm(dim=-1, keepdim=True)
        elif self.center_type == "text":
            cls_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)
        else:
            if self.use_up_cen:
                cls_proto = self.t_lam * (img_proto / img_proto.norm(dim=-1, keepdim=True)) + \
                            (1 - self.t_lam) * (text_proto / text_proto.norm(dim=-1, keepdim=True))
            else:
                img_proto2 = self._network.get_cls_center()
                cls_proto = self.t_lam * (img_proto2 / img_proto2.norm(dim=-1, keepdim=True)) + \
                            (1 - self.t_lam) * (text_proto / text_proto.norm(dim=-1, keepdim=True))

        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                transf_image_features, _ = self._network.encode_image_eval(inputs)
                transf_image_features1, _ = transf_image_features

                transf_image_features1 = transf_image_features1 / transf_image_features1.norm(dim=-1, keepdim=True)

                out_update = transf_image_features1 @ cls_proto.T
                out_gda = self.gda_pred(inputs)

                out_ens_gda_mat = self.stat * out_gda + (1 - self.stat) * out_update
                preds = torch.topk(out_ens_gda_mat, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]

            y_pred.append(preds.cpu().numpy())  # [bs, topk]
            y_true.append(targets.cpu().numpy())  # [bs]

        y_pred = np.concatenate(y_pred, axis=0)  # [N, topk]
        y_true = np.concatenate(y_true, axis=0)  # [N]

        # assert y_pred.ndim == 2 and y_pred.shape[1] == self.topk, y_pred.shape
        # assert y_true.ndim == 1 and y_pred.shape[0] == y_true.shape[0], (y_pred.shape, y_true.shape)

        return y_pred, y_true

