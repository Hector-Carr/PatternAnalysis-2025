import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np
import nibabel as nib

from nifti import load_data_3D

MR_PATH = "data/semantic_MRs_anon/"
LAB_PATH = "data/semantic_labels_anon/"

class HipMRI_Dataset(Dataset):
    def __init__(self, MRs, labels, *args, **kwargs):
        print("Loading MRIs")
        self._MRs = MRs
        #self._MRs = torch.from_numpy(load_data_3D(MRs))
        print("Loading Labels")
        self._labels = labels
        #self._labels = torch.from_numpy(load_data_3D(labels))
            

    def __len__(self):
        return len(self._MRs)

    def __getitem__(self, i):
        return (
            torch.from_numpy(load_data_3D([self._MRs[i]])),
            torch.from_numpy(load_data_3D([self._labels[i]]))
        )
        return self._MRs[i], self._labels[i]
        #return self._MRs[self._index[i]], self._labels[self._index[i]]


# function to automatically split data into test and train dataloaders
def get_dataloaders():
    # load files availble
    MRs = [MR_PATH+f for f in os.listdir("data/semantic_MRs_anon")]
    labels = [LAB_PATH+f for f in os.listdir("data/semantic_labels_anon")]

    if len(MRs) != len(labels):
        raise Exception("different number of samples to labels")

    # do train test split without fully loading data
    X_train, X_test, y_train, y_test = train_test_split(
        MRs, labels, test_size=0.2, 
        random_state=42, shuffle=True
    )

    train_dataset = HipMRI_Dataset(X_train, y_train)
    test_dataset = HipMRI_Dataset(X_test, y_test) 

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

    return (
        train_loader,
        test_loader
    )

if __name__ == "__main__":
    #dl = DataLoader()
    MRs = [MR_PATH+f for f in os.listdir("data/semantic_MRs_anon")]
    labels = [LAB_PATH+f for f in os.listdir("data/semantic_labels_anon")]
    
    X_train, X_test, y_train, y_test = train_test_split(
        MRs, labels, test_size=0.2, 
        random_state=42, shuffle=True
    )
    
    a = HipMRI_Dataset(X_test, MRs, labels)
    
    print(a)

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

    


