from torch.utils.data import Dataset
import pandas as pd
import torch

# Define dataset
class QuasarDataset(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        # read pkl
        aux = pd.read_pickle(data_path)
        self.labels = aux['Z']
        self.data = aux['flux']
        del aux
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Read data
        quasar = self.data[idx]

        # Read label
        label = self.labels[idx]

        # Transform data
        if self.transform:
            quasar = self.transform(quasar)

        # Transform label
        if self.target_transform:
            label = self.target_transform(label)

        return quasar, label
