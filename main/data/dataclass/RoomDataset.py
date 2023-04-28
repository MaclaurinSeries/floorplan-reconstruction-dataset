import os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data, InMemoryDataset
import itertools

__all__ = ['RoomDataset']


def inMem(root, train_file, val_file, test_file):
    data = []
    for idx,file in enumerate([train_file, val_file, test_file]):
        filenames = None
        with open(os.path.join(root,file), 'r') as f:
            filenames = f.read().split('\n')
        dataset = []
        for filename in filenames:
            dt = torch.load(os.path.join(root, 'graph', f'{filename}.pt'))
            if np.random.rand() < 0.5:
                dt.x[9:] = 0
            # dataset.append(Data(
            #     x= dt.x,
            #     edge_index= dt.edge_index,
            #     edge_attr= dt.edge_attr,
            #     y= dt.y
            # ))
            dataset.append(dt)
        dat,slice = InMemoryDataset.collate(dataset)
        data.append(dat)
    data,slice = InMemoryDataset.collate(data)

    data.train_mask = torch.zeros(slice['x'][-1], dtype=torch.bool)
    data.train_mask[slice['x'][0]: slice['x'][1]] = True

    data.val_mask = torch.zeros(slice['x'][-1], dtype=torch.bool)
    data.val_mask[slice['x'][1]: slice['x'][2]] = True

    data.test_mask = torch.zeros(slice['x'][-1], dtype=torch.bool)
    data.test_mask[slice['x'][2]: slice['x'][3]] = True

    return data

class RoomDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, filenames=None):
        self.filenames = filenames
        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)


    @property
    def processed_dir(self):
        return os.path.join(self.root, 'graph')


    @property
    def raw_dir(self):
        return os.path.join(self.root, 'graph')


    @property
    def raw_file_names(self):
        filenames = None
        with open(os.path.join(self.root, self.filenames), 'r') as f:
            filenames = f.read().split('\n')
        return [f'{filename}.pt' for filename in filenames]


    @property
    def processed_file_names(self):
        filenames = None
        with open(os.path.join(self.root, self.filenames), 'r') as f:
            filenames = f.read().split('\n')
        return [f'{filename}.pt' for filename in filenames]


    def process(self):
        raise NotImplementedError


    def len(self):
        return len(self.processed_file_names)


    def get(self, idx):
        filename = self.processed_file_names[idx]
        data = torch.load(os.path.join(self.processed_dir, filename))
        return data