import torch
import numpy as np
from torch.utils.data import Dataset


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def to_image_array(image, n_channels=1024):
    r = image[0:n_channels]
    g = image[n_channels:2*n_channels]
    b = image[2*n_channels:3*n_channels]
    return np.array(list(zip(r, g, b))).reshape(32, 32,3)



class Cifar10(Dataset):
    classes = (
            'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck',
            )

    def __init__(self, root_dir, transform=None):
        self.data = unpickle(root_dir)
        self.transform = transform

    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()

        image = to_image_array(self.data[b'data'][idx])
        label = self.data[b'labels'][idx]
        sample = (image, label)
        if(self.transform):
            sample = (self.transform(sample[0]), sample[1])
        return sample

    def __len__(self):
        return len(self.data[b'labels'])