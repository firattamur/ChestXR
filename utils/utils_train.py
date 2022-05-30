import os
import glob
import torch

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def save_checkpoint(checkpoint: dict, path: str) -> None:
    """
    Save torch model state dict to specified path.
    :param model: input dict contains state dicts for models and optimizers
    :param path : path to save state dict
    """

    torch.save({

        "epoch"         : checkpoint["epoch"],
        "optimizer"     : checkpoint["optimizer"].state_dict(),
        "model"         : checkpoint["model"].state_dict(),
        "scheduler"     : checkpoint["scheduler"].state_dict(),
        "best_accuracy" : checkpoint["best_accuracy"]

    }, path)


def load_checkpoint(path: str, map_location = None) -> dict:
    """
    Load torch model state dict from specified path.
    :param path : path to load state dict
    """
    checkpoint = torch.load(path, map_location)

    return checkpoint


def best_or_last_checkpoint(path: str) -> str:
    """

    Sort checkpoints saved date and return last store checkpoint path.

    :param path: path for checkpoints folder.
    :return    : last stored checkpoint path

    """

    checkpoints = glob.glob(os.path.join(path, "*.pth"))

    if len(checkpoints) == 0:
        return None

    for checkpoint in checkpoints:
        if "_best" in checkpoint:
            return checkpoint

    latest = max(checkpoints, key=os.path.getctime)

    return latest


def accuracy(output, target, topk: tuple = (1,)) -> torch.Tensor:
    """

    Compute the precision@k for the specified values of k.

    """
       
    output = (output > 0.5).astype(int)
 
    return f1_score(target.astype(int), output, average='weighted', zero_division=1)

    # maxk = max(topk)
    # batch_size = target.size(0)

    # _, pred = output.topk(maxk, 1, True, True)
    # pred = pred.t()
    # correct = pred.eq(target.view(1, -1).expand_as(pred))

    # res = []

    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0)
    #    res.append(correct_k.mul_(100.0 / batch_size))

    # return res



