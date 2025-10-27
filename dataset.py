from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from sklearn.model_selection import train_test_split
import os
import numpy as np

from nifti import load_data_3D

MR_PATH = "data/semantic_MRs_anon/"
LAB_PATH = "data/semantic_labels_anon/"

class DataLoader(Dataset):
    def __init__(self, index, MRs, labels, *args, **kwargs):
        self._index = index
        self._MRs = MRs
        self._labels = labels

        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        return self._MRs[self._index[i]], self._labels[self._index[i]]


# function to automatically split data into test and train dataloaders
def get_dataloaders():
    # load files availble
    MRs = os.listdir("data/semantic_MRs_anon")
    labels = os.listdir("data/semantic_labels_anon")

    if len(MRs) != len(labels):
        raise Exception("different number of samples to labels")

    # do train test split without fully loading data
    samples = np.arange(len(MRs))
    X_train, X_test, y_train, y_test = train_test_split(
        samples, samples.copy(), test_size=0.2, 
        random_state=42, shuffle=True
    )

    return (
        DataLoader(X_train, MRs, labels), # train loader
        DataLoader(X_test, MRs, labels), # test loader
    )

if __name__ == "__main__":
    #dl = DataLoader()
    a, b = get_dataloaders()
    print(a[2])
    print(len(b))


