import torch
from os import listdir
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def onehot(v: int):
    a = [0]*10
    a[v] = 1
    return a

def mnist():
    folder = "corruptmnist"
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
    # test_std = (test-test.mean())/test.std()

    # d = {
    #     0: test_std[101].reshape(28,28),
    #     1: test_std[107].reshape(28,28),
    #     2: test_std[106].reshape(28,28),
    #     3: test_std[112].reshape(28,28),
    #     4: test_std[103].reshape(28,28),
    #     5: test_std[102].reshape(28,28),
    #     6: test_std[100].reshape(28,28),
    #     7: test_std[111].reshape(28,28),
    #     8: test_std[110].reshape(28,28),
    #     9: test_std[105].reshape(28,28),
    # }
    # np.save("example_images.npy", d)
    return train, train_label, test, test_label

if __name__ == "__main__":
    mnist()