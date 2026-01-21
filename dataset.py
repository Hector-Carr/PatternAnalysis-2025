import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
import nibabel as nib

from nifti import load_data_3D

MR_PATH = "data/semantic_MRs_anon/"
LAB_PATH = "data/semantic_labels_anon/"

BATCH_SIZE = 1

class HipMRI_Dataset(Dataset):
    def __init__(self, MRs, labels, normalised=True):
        assert len(MRs) == len(labels), "Number of images and labels must match"
        self.images = load_data_3D(MRs, normImage=normalised)
        self.labels = load_data_3D(labels, dtype=np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # --- Ensure correct shapes ---
        # Input: [D, H, W] → [1, D, H, W]
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # --- One-hot encode label ---
        label_tensor = torch.from_numpy(label).long()              # [D, H, W]
        label_onehot = F.one_hot(label_tensor, num_classes=6)  # [D, H, W, 6]
        label_onehot = label_onehot.permute(3, 0, 1, 2).float()    # [6, D, H, W]

        # --- Convert to tensors ---
        image = torch.from_numpy(image).float()     # [1, D, H, W]

        return image, label_onehot

# function to automatically split data into test and train dataloaders
def get_dataloaders(train_val=False, test=False, normalised=True):
    """
    return requested dataloaders for later use
    """
    # load files availble
    MRs = [MR_PATH+f for f in os.listdir(MR_PATH)]
    MRs.sort()
    labels = [LAB_PATH+f for f in os.listdir(LAB_PATH)]
    labels.sort()

    # basic check for files
    if len(MRs) != len(labels):
        raise Exception("different number of samples to labels")

    # do train test validation split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        MRs, labels, test_size=0.2, 
        random_state=42, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, 
        random_state=42, shuffle=True
    )
    
    if train_val:
        # load datasets
        train_dataset = HipMRI_Dataset(X_train, y_train, normalised=normalised)
        val_dataset = HipMRI_Dataset(X_val, y_val, normalised=normalised)

        # put datasets into dataloaders 
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        return (
            train_loader,
            val_loader
        )
    
    elif test:
        # load dataset
        test_dataset = HipMRI_Dataset(X_test, y_test, normalised=normalised)

        # put dataset into dataloader
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        return test_loader

    else:
        return None

"""
5: prostate
4: anal cavity
3: bladder
2: bone
1: miscilanious flesh
0: background
"""


