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

def concatenate(datasets:list):
    result = datasets[0].copy()
    data_list = datasets[1:]

    for data in data_list:
        result.data[b'data'] = np.concatenate([result.data[b'data'], data.data[b'data']])
        result.data[b'labels'] = list(np.concatenate([result.data[b'labels'], data.data[b'labels']]))

    return result

    return result

def class_count(dataset):
    class_count = {}
    for sample in dataset:
        label = dataset.classes[sample[1]]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    return class_count

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

    def copy(self):
        import copy
        return copy.deepcopy(self)
