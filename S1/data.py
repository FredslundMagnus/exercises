import torch
from os import listdir
from os.path import join
import numpy as np

def onehot(v: int):
    a = [0]*10
    a[v] = 1
    return a

def mnist():
    folder = "S1/corruptmnist"
    train: torch.Tensor = None
    train_label: torch.Tensor = None
    for file in listdir(folder):
        data = np.load(join(folder,file))
        if file.startswith("train"):
            if train is None:
                train = torch.as_tensor(data["images"].reshape((-1, 28*28)), dtype=float)
                train_label = torch.as_tensor([onehot(v) for v in data["labels"]], dtype=float)
            else:
                train = torch.cat((train, torch.as_tensor(data["images"].reshape((-1, 28*28)), dtype=float)))
                train_label = torch.cat((train_label, torch.as_tensor([onehot(v) for v in data["labels"]], dtype=float)))
        else:
            test: torch.Tensor = torch.as_tensor(data["images"].reshape((-1, 28*28)), dtype=float)
            test_label: torch.Tensor = torch.as_tensor([onehot(v) for v in data["labels"]], dtype=float)

    return train, train_label, test, test_label