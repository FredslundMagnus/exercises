import multiprocessing
cores = multiprocessing.cpu_count()
print(f"Number of cores: {cores}, Number of threads: {2*cores}")

from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self):
        # whatever logic is needed to init the data set
        self.data = [(i, i+1) for i in range(100)]

    def __getitem__(self, idx):
        # return one item
        return self.data[idx]

dataset = MyDataset()
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4  # this is the number of threds we want to parallize workload over
)