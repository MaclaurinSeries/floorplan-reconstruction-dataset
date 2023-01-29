import os
import torch
from torch_geometric.data import Dataset, Data
import itertools

__all__ = ['RoomDataset']

class RoomDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        
    @property
    def processed_dir(self):
        return os.path.join(self.root, 'graph-processed')

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'graph-raw')

    @property
    def raw_file_names(self):
        return list(
            itertools.chain.from_iterable(
                [files for _,_,files in os.walk(self.raw_dir, topdown=False)]
            )
        )

    @property
    def file_IDs(self):
        return [filename.split('.')[0] for filename in self.raw_file_names]

    @property
    def processed_file_names(self):
        return [f'graph.{file_ID}.pt' for file_ID in self.file_IDs]
    
    def process(self):
        '''
            file format:
            n m
            d0
            d1
            :
            dn
            a0 b0 c0
            a1 b1 c1
            :
            am bm cm
            e0
            e1
            :
            en

            penjelasan:
            n banyaknya kamar (vertex)
            m banyaknya hubungan antar kamar (edge)
            a dan b adalah dua kamar yang berhubungan, masing2 pada range [0, n)
            c adalah vector representasi jenis hubungan (pintu[1]/tanpa sekat[0])
            d adalah vector representasi hasil agregasi onehot-enc dari furniture2 dalam kamar
            e adalah vector representasi jenis kamar (y label)
        '''
        for i,raw_path in enumerate(self.raw_paths):
            file_ID = self.file_IDs[i]

            with open(raw_path, 'r') as f:
                lines = [line.rstrip() for line in f]
                n,m = list(map(
                    int,
                    lines[0].split()
                ))

                x = []
                edge_index = []
                edge_attr = []
                y = []
                for i in range(n):
                    x.append(
                        list(map(
                            int,
                            lines[i + 1].split()
                        ))
                    )
                for i in range(m):
                    edge = list(map(
                        int,
                        lines[i + n + 1].split()
                    ))
                    edge_index.append(edge[:2])
                    edge_attr.append(edge[2:])
                for i in range(n):
                    y.append(
                        list(map(
                            int,
                            lines[i + m + n + 1].split()
                        ))
                    )

                x = torch.tensor(x, dtype=torch.float)
                
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

                torch.save(data, os.path.join(self.processed_dir, f'graph.{file_ID}.pt'))
    
    def len(self):
        return len(self.file_IDs)

    def get(self, idx):
        file_ID = self.file_IDs[idx]
        data = torch.load(os.path.join(self.processed_dir, f'graph.{file_ID}.pt'))
        return data