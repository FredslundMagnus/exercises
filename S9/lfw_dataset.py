"""
LFW dataloading
"""
import argparse
import time
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt


class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.transform = transform
        self.files: list[str] = []
        for folder in os.listdir(path_to_folder):
            for file in os.listdir(os.path.join(path_to_folder, folder)):
                self.files.append(os.path.join(path_to_folder, folder, file))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.files[index])
        return self.transform(img)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='lfw', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', action='store_true')
    parser.add_argument('-get_timing', action='store_true')
    parser.add_argument('-batches_to_check', default=100, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    if args.visualize_batch:
        imgs = [dataset[0], dataset[1]]
        if not isinstance(imgs, list):
            imgs = [imgs]
        fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        
    if args.get_timing:
        # lets do some repetitions
        res = [ ]
        for _ in range(5):
            start = time.time()
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx > args.batches_to_check:
                    break
            end = time.time()

            res.append(end - start)
            
        res = np.array(res)
        print(f'Timing: {np.mean(res)}+-{np.std(res)}')

