import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + increment))[0]
        label = "{}-{}".format(str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2))

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = (0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2))

    # Harmonic mean of old and new accuracy
    all_acc["harmonic"] = np.around(2 * all_acc["old"] * all_acc["new"] / (all_acc["old"] + all_acc["new"]), decimals=2)
    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def get_attribute(dic, name, default):
    if name in dic:
        return dic[name]
    else:
        print(name, 'not in args, set to', default, ' as default')
        return default


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            img_only=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.img_only = img_only
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        if self.img_only:
            total_loss = F.cross_entropy(logits_per_image, labels)
        else:
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class MarginLoss(nn.Module):
    def __init__(self, last_mask=False):
        super(MarginLoss, self).__init__()
        self.last_mask = last_mask

    def creat_task_mask(self, task_label, task_class):
        mask = (task_label.unsqueeze(1) != task_class.unsqueeze(0)).float()
        if self.last_mask:
            max_task = task_class.max().item()
            is_mask_task = (task_label == max_task).unsqueeze(1).float()
            mask = mask * is_mask_task
        return mask

    def softmax_with_masked_margin(self, logits, similarity_matrix, task_mask, margin_scale=0.1):
        margins = margin_scale * similarity_matrix * task_mask
        adjusted_logits = logits + margins  # (batch, n_classes, n_classes)
        return adjusted_logits

    def compute_class_similarity(self, class_centers, text_proto, labels):

        row_sim = class_centers @ class_centers.T + text_proto @ text_proto.T

        min_sim = (row_sim - torch.diag(torch.diag(row_sim))).min()

        similarity_matrix = class_centers @ class_centers.T + text_proto @ text_proto.T - min_sim
        return similarity_matrix[labels]

    def forward(self, logits, labels, class_centers, text_proto, task_ids, margin_lam=0.1):
        text_proto = text_proto / text_proto.norm(dim=-1, keepdim=True)
        similarity_matrix = self.compute_class_similarity(class_centers, text_proto, labels)
        task_label = task_ids[labels]
        task_class = task_ids[range(class_centers.size(0))]
        task_mask = self.creat_task_mask(task_label, task_class)
        adj_logits = self.softmax_with_masked_margin(logits, similarity_matrix, task_mask, margin_lam)
        loss = F.cross_entropy(adj_logits, labels)
        return loss
