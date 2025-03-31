import os
import torch  # Add this import
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RNACustomDataset(Dataset):
    def __init__(self, rna_data, image_filenames, image_directory, transform=None):
        self.rna_data = rna_data
        self.image_filenames = image_filenames
        self.image_directory = image_directory  # Add image_directory as a class attribute
        self.transform = transform

    def __len__(self):
        return len(self.rna_data)

    def __getitem__(self, idx):
        # RNA Data
        rna_sample = torch.tensor(self.rna_data.iloc[idx].values.astype(float)).float()

        # Image Data
        img_path = os.path.join(self.image_directory, self.image_filenames[idx])
        nii_image = nib.load(img_path)
        image_volume = nii_image.get_fdata()


        mid_slice = image_volume[:, :, image_volume.shape[2] // 2]


        rgb_slice = np.stack([mid_slice] * 3, axis=-1)
        image = Image.fromarray(np.uint8(rgb_slice))

        if self.transform:
            image = self.transform(image)

        return rna_sample, image
