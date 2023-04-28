from pathlib import Path
from typing import Any, Callable, Optional
import sys, time

from tqdm import tqdm

import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset

__all__ = ['SegmentationDataset']


class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 set_path: str,
                 transforms: Optional[Callable] = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            set_path (str): Name of the file that contains correspond data.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            subset (str, optional): 'Train' or 'Validation' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.
        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Validation'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode
        
        if subset not in ["Train", "Validation"]:
            raise (ValueError(
                f"{subset} is not a valid input. Acceptable values are Train and Validation."
            ))

        file_path = Path(self.root) / set_path

        fnames = None
        with open(file_path, 'r') as f:
            fnames = [f'{fname}.png' for fname in f.read().split('\n')]

        # perform file checking
        sys.stdout.write(f'Perform sanity checking on {subset} data\n')
        scanned = 0
        corrupt = 0
        target_len = len(str(len(fnames)))

        self.image_names = []
        self.mask_names = []

        progress = tqdm(fnames)
        for fname in progress:
            progress.set_description(f'{scanned: >{target_len}d} scanned, {corrupt: >{target_len}d} corrupted')

            image_path = image_folder_path / fname
            mask_path = mask_folder_path / fname

            if not image_path.is_file() or not mask_path.is_file() or Image.open(image_path, 'r').size != Image.open(mask_path, 'r').size:
                corrupt += 1
            else:
                self.image_names.append(image_path)
                self.mask_names.append(mask_path)

            scanned += 1

        sys.stdout.write(f'{corrupt} files removed from {scanned} images\n')


    def __len__(self) -> int:
        return len(self.image_names)


    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]

        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")

            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")

            sample = {"image": image, "mask": mask}

            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
            return sample