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

# zero shot clip

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = SimpleClipNet(args, True)
        self.batch_size = get_attribute(args, "batch_size", 48)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.args = args


    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train", )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        # self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._network.to(self._device)

    def _compute_accuracy(self, model, loader):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]  # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.convnet.encode_text(texts)  # num_str, dim
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)  # num_classes, dim

        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                with torch.no_grad():
                    # outputs = self._network(inputs)["logits"]
                    image_features = self._network.convnet.encode_image(inputs)
                    image_features /= image_features.norm(dim=-1, keepdim=True)  # bs, dim
                    outputs = image_features @ text_features.T  # bs, num_classes
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        print('Accuracy: {:.2f}%'.format(correct * 100 / total))
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[:self._total_classes]  # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.convnet.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features = self._network.convnet.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                outputs = image_features @ text_features.T
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_zero_shot(self):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label  # [:self._total_classes] # mask all known classes
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                # class_embeddings = self._network.encode_text(texts)
                class_embeddings = self._network.convnet.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        test_dataset = self.data_manager.get_dataset(np.arange(0, len(total_labels)), source="test", mode="test")
        loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        y_pred, y_true = [], []
        logits = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                # image_features=self._network.encode_image(inputs)
                image_features = self._network.convnet.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                outputs = image_features @ text_features.T
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            logits.append(outputs.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


