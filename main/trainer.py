from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.loggers import CometLogger

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader, ImbalancedSampler, NeighborLoader
from torch_geometric.utils import degree
from data.dataclass.RoomDataset import RoomDataset, inMem
import torch
import warnings

from comet_ml import Experiment

import click
import shutil
from pathlib import Path

from models.GraphSAGE import Model as GraphSAGE
from models.GCN import Model as GCN
from models.GIN import Model as GIN
from models.GAT import Model as GAT
from models.PNA import Model as PNA
from models.GraphSAGEjk import Model as GraphSAGEjk
from models.PNAjk import Model as PNAjk

NODE_FEATURE_LENGTH = 19
NODE_CLASS_LENGTH = 10
BATCH_SIZE = 128
WORKERS = 8

CHECKPOINT_DIRPATH = './run/checkpoints/'
SAVED_MODEL_DIRPATH = './run/saved_models/'

MODEL_COLLECTION = {
    'graph-sage': GraphSAGE,
    'gin': GIN,
    'gcn': GCN,
    'gat': GAT,
    'pna': PNA,
    'graph-sage-jk': GraphSAGEjk,
    'pna-jk': PNAjk
}



def train(model_name, trials, epoch, root):
    model_class = MODEL_COLLECTION[model_name]
    warnings.filterwarnings("ignore")

    for trial in range(trials):
        comet_logger = CometLogger(
            api_key="nUIRrqnYAp8FshrI633txTUZ0",
            project_name="gnn-refined",
            workspace="maclaurinseries",
            save_dir="./run/comet-logger",
            experiment_name=f'{model_name}-{trial}-trials-{epoch}-epochs'
        )

        ckpt_dir = Path(CHECKPOINT_DIRPATH) / model_name / str(trial)
        save_dir = Path(SAVED_MODEL_DIRPATH) / model_name / str(trial)

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        data = inMem(root=root, train_file='train.txt', test_file='test.txt', val_file='validation.txt')

        train_sampler = ImbalancedSampler(data.y.argmax(-1), input_nodes=data.train_mask)
        val_sampler = ImbalancedSampler(data.y.argmax(-1), input_nodes=data.val_mask)

        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask,
            num_neighbors=[-1,-1],
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            sampler=train_sampler,
            persistent_workers=True,
        )
        test_loader = NeighborLoader(
            data,
            input_nodes=data.test_mask,
            num_neighbors=[-1,-1],
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
        )
        val_loader = NeighborLoader(
            data,
            input_nodes=data.val_mask,
            num_neighbors=[-1,-1],
            batch_size=BATCH_SIZE,
            num_workers=WORKERS,
            sampler=val_sampler,
            persistent_workers=True,
        )

        val_check_interval = len(train_loader)

        if str(model_name).startswith('pna'):
            max_degree = -1
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            if len(d) > 0:
                max_degree = max(max_degree, int(d.max()))

            # Compute the in-degree histogram tensor
            deg = torch.zeros(max_degree + 1, dtype=torch.long)
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            if len(d) > 0:
                deg += torch.bincount(d, minlength=deg.numel())

            model = model_class(NODE_FEATURE_LENGTH, NODE_CLASS_LENGTH, BATCH_SIZE, degree=deg)
        else:
            model = model_class(NODE_FEATURE_LENGTH, NODE_CLASS_LENGTH, BATCH_SIZE)


        checkpoint = ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor='acc/val',
            save_top_k=1,
            mode='max',
        )

        trainer = Trainer(
            accelerator='gpu',
            strategy=SingleDeviceStrategy('cuda:0'),
            devices=1,
            max_epochs=epoch,
            val_check_interval=val_check_interval,
            log_every_n_steps=8,
            logger=comet_logger,
            callbacks=[checkpoint]
        )

        trainer.fit(model, train_loader, val_loader)
        trainer.test(ckpt_path='best', dataloaders=test_loader)

        print('best checkpoint: ', checkpoint.best_model_path)
        shutil.copyfile(checkpoint.best_model_path, save_dir / 'weight.ckpt')


@click.command()
@click.option("--epochs",
              default=25,
              type=int,
              help="Specify the number of epochs you want to run the experiment for.")
@click.option("--architecture",
              default=None,
              type=str,
              help="Specify the architecture.")
@click.option("--trials",
              default=1,
              type=int,
              help="Specify the number of trials you want to run the experiment for.")
def main(epochs, architecture, trials):
    train(architecture, trials, epochs, './__collection__/room-classification')


if __name__ == "__main__":
    main()