import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 128


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),decimals=2)
        return ret
    
    def _evaluate_zs(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._total_classes) # indx< total are old classes, >= are new unseen classes.
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around((y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),decimals=2)
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None
        return cnn_accy, nme_accy, None, None, None, None
        

    def _eval_zero_shot(self):  
        self._network.eval()
        class_to_label=self.data_manager._class_to_label
        templates=self.data_manager._data_to_prompt
        total_labels=class_to_label#[:self._total_classes] # mask all known classes
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

        test_dataset = self.data_manager.get_dataset(np.arange(0, len(total_labels)), source="test", mode="test" )
        loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        y_pred, y_true = [], []
        logits=[]
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features=self._network.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                outputs= image_features @ text_features.T
            predicts = torch.topk( outputs, k=self.topk, dim=1, largest=True, sorted=True )[  1 ]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            logits.append(outputs.cpu().numpy())
        

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)


    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    

    def _eval_nme(self, loader, class_means):
      #  self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
      #  self._network.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy( self._network.module.extract_vector(_inputs.to(self._device)))
                else:
                    _vectors = tensor2numpy( self._network.extract_vector(_inputs.to(self._device)) )
                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            # self._class_covs = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        radius = []
        for class_idx in range(self._known_classes, self._total_classes):

            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            if self._cur_task == 0:
                cov = np.cov(vectors.T)+ np.eye(class_mean.shape[-1]) * 1e-4
                radius.append(np.trace(cov) /768)
            # class_cov = np.cov(vectors.T)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov

        if self._cur_task == 0:
                self.radius = np.sqrt(np.mean(radius))
                print(self.radius)
            # self._class_covs.append(class_cov)

    def displacement_cov(self, Y, class_mean, embedding_old, sigma):
        cov = None
        start_time = time.time()
        for _class in range(self._known_classes):
            loop_start_time = time.time()
            DY = self.cov_computation(Y, class_mean[_class])
            distance = np.sum((np.tile(Y[None, :, :], [1, 1, 1]) - np.tile(
                embedding_old[_class, None, :], [1, Y.shape[0], 1])) ** 2, axis=2)
            W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
            W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
            if cov is None:
                cov = np.sum(np.tile(W_norm[:, :, None, None], [
                    1, 1, DY.shape[1], DY.shape[2]]) * np.tile(DY[None, :, :, :], [W.shape[0], 1, 1, 1]), axis=1)
            else:
                displacement = np.sum(np.tile(W_norm[:, :, None, None], [
                    1, 1, DY.shape[1], DY.shape[2]]) * np.tile(DY[None, :, :, :], [W.shape[0], 1, 1, 1]), axis=1)
                cov = np.concatenate((cov, displacement))
            loop_end_time = time.time()
            print("single loop time: ", loop_end_time - loop_start_time)
        end_time = time.time()
        print("total loop time: ", end_time - start_time)

        cov = torch.tensor(cov)
        return cov

    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2 - Y1
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement


    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
