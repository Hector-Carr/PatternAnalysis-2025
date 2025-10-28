import torch
import torch.nn.functional as F
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
    def __init__(self, MRs, labels, transfrom=None):
        assert len(MRs) == len(labels), "Number of images and labels must match"
        #self.image_paths = MRs
        #self.label_paths = labels
        self.images = load_data_3D(MRs)
        self.labels = load_data_3D(labels, dtype=np.uint8)
        self.transform = transfrom

    def __len__(self):
        return len(self.image_paths)
        #return len(self.images)

    def __getitem__(self, idx):
        # --- Load image ---
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        # Example: using nibabel (for .nii/.nii.gz)
        #image = nib.load(img_path).get_fdata().astype(np.float32)
        #label = nib.load(lbl_path).get_fdata().astype(np.float32)
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

        # --- Apply transforms (if any) ---
        if self.transform:
            image, label_onehot = self.transform(image, label_onehot)

        # --- Convert to tensors ---
        image = torch.from_numpy(image).float()     # [1, D, H, W]

        return image, label_onehot

# function to automatically split data into test and train dataloaders
def get_dataloaders():
    # load files availble
    MRs = [MR_PATH+f for f in os.listdir(MR_PATH)][:10]
    labels = [LAB_PATH+f for f in os.listdir(LAB_PATH)][:10]

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
    
    # load datasets
    train_dataset = HipMRI_Dataset(X_train, y_train)
    #test_dataset = HipMRI_Dataset(X_test, y_test)
    val_dataset = HipMRI_Dataset(X_val, y_val)

    # put datasets into dataloaders 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return (
        train_loader,
        #test_loader,
        val_loader
    )

if __name__ == "__main__":
    #a = load_data_3D(["data/semantic_labels_anon/Case_042_Week0_SEMANTIC_LFOV.nii.gz"], dtype=np.uint8)
    #print(a.shape)
    #print(a.dtype)
    #print(np.unique(a))
    get_dataloaders()
    exit()
    a, b = get_dataloaders()
    print(a[2])
    print(len(b))

    # test nibabel things
    img = nib.load(MR_PATH+"Case_004_Week1_LFOV.nii.gz")

    # Get the image data as a NumPy array
    data = img.get_fdata()

    # Display some information
    print("Shape:", data.shape)
    print("Data type:", data.dtype)

    # Get affine matrix (maps voxel coordinates to world coordinates)
    print("Affine:\n", img.affine)

"""
5: prostate
4: anal cavity
3: bladder
2: bone
1: miscilanious flesh
0: background
"""


