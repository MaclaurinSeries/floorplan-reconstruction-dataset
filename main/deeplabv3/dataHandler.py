from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import SegmentationDataset

__all__ = ['get_dataloader']


def get_dataloader(data_dir: str,
                    image_folder: str = 'images',
                    mask_folder: str = 'masks',
                    batch_size: int = 4,
                    num_workers: int = 8):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.
    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Validation set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.
    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Validation dataloaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               set_path=f'{x.lower()}.txt',
                               subset=x,
                               transforms=data_transforms)
        for x in ['Train', 'Validation']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=False)
        for x in ['Train', 'Validation']
    }
    return dataloaders