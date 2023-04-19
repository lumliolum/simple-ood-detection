import os
import json
import random
from typing import List, Tuple, Dict

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    return data


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


class MovingAverage:
    def __init__(self, name, rd=4):
        self.name = name
        # avg value
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.rd = rd

    def update(self, x):
        self.sum += x
        self.count += 1

        # update self.value
        self.val = round(self.sum / self.count, self.rd)

    def value(self) -> float:
        return self.val


def set_seed(seed: int, device=torch.device("cuda")) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and device == torch.device("cuda"):
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False # type: ignore


# see -> https://pytorch.org/docs/stable/notes/randomness.html#dataloader
def set_worker_seed(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_image_paths_and_labels(dirpath: str, include_classes: List[str]) -> Tuple[List[str], List[str]]:
    image_paths = []
    labels = []
    for classname in sorted(os.listdir(dirpath)):
        if classname in include_classes and os.path.isdir(os.path.join(dirpath, classname)):
            for filename in os.listdir(os.path.join(dirpath, classname)):
                if filename.lower().endswith((".png", ".jpeg", ".jpg")):
                    filepath = os.path.join(dirpath, classname, filename)
                    image_paths.append(filepath)
                    labels.append(classname)

    return image_paths, labels


def check_split_values(split_values: List[float]):
    assert len(split_values) == 3, f"only 3 values for train test split is allowed, but {len(split_values)} were received."
    assert any([x > 0 for x in split_values]), f"only positive train test split is allowed, but {split_values} were received."
    assert np.sum(split_values) == 1.0, f"sum of split values should be one, but received sum is {sum(split_values)}"


# copied from https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L406
# then did some changes.
def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: float,
):
    norm_classes = [
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue

            if isinstance(module, norm_classes):
                params["norm"].append(p)
            else:
                params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})

    return param_groups


def gather_preds_and_embeddings(model, data_loader, device, gather_labels=False):
    embds_array, preds_array, labels_array = None, None, None

    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(data_loader):
            img = batch["image"].float().to(device)

            out, embds = model(img)
            preds = torch.argmax(out, dim=1)
            if index == 0:
                embds_array = [x.cpu().numpy() for x in embds]
                preds_array = preds.cpu().numpy()
                if gather_labels:
                    labels_array = batch["label"].long().numpy()
            else:
                embds_array = [np.vstack((embds_array[idx], x.cpu().numpy())) for idx, x in enumerate(embds)] # type: ignore
                preds_array = np.concatenate((preds_array, preds.cpu().numpy())) # type: ignore
                if gather_labels:
                    labels_array = np.concatenate((labels_array, batch["label"].long().numpy())) # type: ignore

    # if gather_labels is False, then second argument will be always none.
    return embds_array, preds_array, labels_array


def mahalanobis(embds, mean, cov):
    """
        embds -> (N, D)
        mean -> (D,)
        cov -> (D, D)
        where output represents mahalanobis distances for each example.
        output -> (N,)
    """
    # (N, D) - (1, D) = (N, D)
    ec = embds - mean[None, :]
    # NxD x DxD x DxN = NxN
    distance_matrix = np.dot(ec, np.dot(np.linalg.inv(cov), np.transpose(ec)))
    # to get the distances, take out value of the main diagonal of the matrix.
    # to get the confidence, multiply the distance by -1.
    confidences = -np.diag(distance_matrix)
    return confidences


def calculate_mean_and_cov_matrix(embds: List[np.ndarray], labels: np.ndarray) -> Tuple[List[Dict], List]:
    """
        embds : It is list containing the training embeddings (training image embeddings) for each layer.
        if the list is of length 1, then it means the embeddings were extraced from one layer
        labels: This contains the label information (model index to be precise) for the training images.

    This function will calculate mean (for each class, layer) and tied covariance matrix (for each layer)
    as per the paper : https://arxiv.org/pdf/1807.03888.pdf
    """
    unique_labels = np.unique(labels)
    # in means, we store per layer per class, whereas in covs we store tied-covs per layer
    means, tied_covs = [], []
    for layer_embds in embds:
        if len(layer_embds) != len(labels):
            raise RuntimeError(f"Length of layer embeddings and labels are not same, recieved {len(layer_embds)} and {len(labels)}")

        num_samples = len(labels)
        layer_means = {}
        layer_tied_cov = 0
        for label in unique_labels:
            # layer_label_embds -> (N, D)
            layer_label_embds = layer_embds[labels == label]
            # mean -> (D,)
            mean_layer_label_embds = np.mean(layer_label_embds, axis=0)
            # X -> (N, D)
            X = layer_embds - mean_layer_label_embds[None, :]
            # here I am not dividing it by number of samples, so technically it is not covariance matrix
            cov_layer_label_embds = np.dot(np.transpose(X), X)

            layer_tied_cov = layer_tied_cov + cov_layer_label_embds

            # store the mean with label as the key.
            layer_means[label] = mean_layer_label_embds

        layer_tied_cov = layer_tied_cov / num_samples
        means.append(layer_means)
        tied_covs.append(layer_tied_cov)

    return means, tied_covs


def calculate_ood_metrics(data: pd.DataFrame, known_classes: List, unknown_classes: List) -> Dict[str, float]:
    """
        As of now we are calculating 3 metrics that is
        1. TNR@TPR = 0.95 : True negative rate when the true positive rate is at 95%
        2. AUROC : Area under receiver operating characterstics curve.
        3. Accuracy @ TPR = 0.95 : At the threshold where the TPR is 95%, calculate the classification accuracy.
        For out of distribution classes,
            a. if confidence greater than or equal to threshold, sample will get score 0.
            b. otherwise sample will get score 0.
        For in distribution classes,
            a. if the confidence greater than or equal to threshold and predicted class is ground truth, sample will get score 1.
            b. otherwise sample will get score 0.
    """
    # reqtpr means required tpr.
    reqtpr = 0.95
    num_samples = len(data)
    sorted_data = data.sort_values(by=["confidence"], ascending=False).reset_index()
    # in_gt -> 1 means in-distribution class and 0 mean out-distribution class.
    sorted_data["in_gt"] = 0
    sorted_data.loc[sorted_data["gt"].isin(known_classes), "in_gt"] = 1
    index = np.searchsorted(sorted_data["in_gt"].cumsum(), int(reqtpr * num_samples))
    threshold = sorted_data["confidence"][index]

    # tnr, tpr @ tpr = 0.95
    sorted_data["in_pred"] = 0
    # for the samples which have confidence greater than or equal to threshold, make the prediction 1
    sorted_data.loc[sorted_data["confidence"] >= threshold, "in_pred"] = 1
    cm = confusion_matrix(sorted_data["in_gt"].values, sorted_data["in_pred"].values) # type: ignore
    # tnr = TN / (TN + FP)
    # tpr = TP / (TP + FN)
    tnr = (cm[0, 0]) / (cm[0, 0] + cm[0, 1])
    # tpr will be approximately 95%
    # recalculating for my satisfaction.
    tpr = (cm[1, 1]) / (cm[1, 1] + cm[1, 0])

    auroc = roc_auc_score(sorted_data["in_gt"].values, sorted_data["confidence"].values) # type: ignore

    # accuarcy @ tpr = 0.95
    sorted_data["gt_class"] = sorted_data["gt"].copy()
    sorted_data.loc[sorted_data["gt"].isin(unknown_classes), "gt_class"] = "OODSAMPLE"
    sorted_data["pred_class"] = sorted_data["pred"].copy()
    sorted_data.loc[sorted_data["confidence"] < threshold, "pred_class"] = "OODSAMPLE"
    acc = (sorted_data["gt_class"] == sorted_data["pred_class"]).mean()

    return {
        "tpr@tpr=0.95": round(float(tpr), 4),
        "tnr@tpr=0.95": round(float(tnr), 4),
        "auroc": round(float(auroc), 4),
        "acc@tpr=0.95": round(float(acc), 4)
    }
