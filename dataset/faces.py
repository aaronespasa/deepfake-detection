"""
PyTorch faces dataset. It uses the faces from the FaceForensics dataset.

It's a dataset for binary image classification, so the images are labeled as
'fakes' or 'real'.

Copyright (c), Aarón, Jaime, Nacho & Conchi - All Rights Reserved

This source code is licensed under the Apache license found in the
LICENSE file in the root directory of this source tree:
https://github.com/aaronespasa/deepfake-detection/blob/main/LICENSE
"""

import os
import pandas as pd
import torch
import torch.utils.data as data
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import matplotlib.pyplot as plt

from constants import FACES_FOLDER, FACES_CSV

class Faces(data.Dataset):
    """
    PyTorch faces dataset. It uses the faces from the FaceForensics dataset.

    It's a dataset for binary image classification, so the images are labeled as
    'fakes' or 'real'.
    """
    def __init__(self,
                 root:str=FACES_FOLDER,
                 csv:str=FACES_CSV,
                 split:str="training",
                 transform:bool=None):
        """
        Args:
            root (string): Root directory of the dataset.
            csv (string): Path to the csv file with the faces.
            split (string): Split of the dataset to use: 'training' or 'validation'
            transform (bool, optional): Optional transform to be applied
                                        on a sample.
        """
        self.root = root
        self.csv = pd.read_csv(csv)
        self.transform = transform
        self.split = split
        self.real_label = "real"
        self.fake_label = "fake"
    
    def __len__(self):
        """Total number of images in the dataset"""
        return len(self.csv)

    @staticmethod
    def get_train_transformations():
        """Get training transformations for data augmentation.
        """
        return A.Compose([
            A.Resize(256, 256),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            ToTensorV2(p=1.0)
        ])

    @staticmethod
    def get_val_transformations():
        """Get validation transformations for data augmentation."""
        return A.Compose([
            A.Resize(256, 256),
            ToTensorV2(p=1.0)
        ])
    
    def _read_image(self, image_path:str, label:str):
        """Read image and normalize it
        
        Args:
            image_path (str): Path to the image.
            label (str): Label of the image.
        """
        # Get the full path to the image
        image = ""
        if label == self.real_label:
            image = os.path.join(self.root, "real", image_path)
        else:
            image = os.path.join(self.root, "fake", image_path)
        
        # Read the image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize the image
        image = image / 255.0

        # Convert the image to floating point to use it as
        # an input to the PyTorch model
        image = image.astype(np.float32)

        return image

    def __getitem__(self, idx:int):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, label) where label 'real' or 'fake'
        """
        row = self.csv.iloc[idx]
        image = row["name"]
        label = row["label"]

        image = self._read_image(image, label)

        if self.transform:
            if self.split == "training":
                transformation = self.get_train_transformations()
            else:
                transformation = self.get_val_transformations()
            image = transformation(image=image)["image"]

        return image, label

if __name__ == "__main__":
    dataset = Faces(root=FACES_FOLDER,
                    csv=FACES_CSV,
                    split="training",
                    transform=True)
    
    image, label = dataset[0]
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = image * 255.0
    image = image.astype(np.uint8)

    plt.title(label)
    plt.imshow(image)
    plt.show()
    