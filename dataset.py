import torch
import pickle
import copy
import numpy as np
from torch.utils.data import Dataset


def unpickle(file):
    """
        load pickle file as a dictionary
    """

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def to_image_array(image, n_channels=1024, dshape=(32, 32,3)):
    """
        convert image vector(n_channels*R, n_channels*G, n_channels*B)
        to image matrix (n_channels*(R, G, B))
    """

    r = image[0:n_channels]
    g = image[n_channels:2*n_channels]
    b = image[2*n_channels:3*n_channels]
    return np.array(list(zip(r, g, b))).reshape(*dshape)

def concatenate(datasets:list):
    """
        convert list of dataset objects to
        one dataset object
    """

    # copy one of the datasets
    result = datasets[0].copy()
    data_list = datasets[1:]

    # concat rest of dataset objects
    # with the first dataset in the list
    for data in data_list:
        result.images = np.concatenate([result.images, data.images])
        result.labels = list(np.concatenate([result.labels, data.labels]))

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
        data = unpickle(root_dir)
        self.images = list(map(to_image_array, data[b'data']))
        self.labels = data[b'labels']
        self.transform = transform

    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.tolist()

        image = self.images[idx]
        label = self.labels[idx]
        sample = (image, label)
        if(self.transform):
            sample = (self.transform(sample[0]), sample[1])
        return sample

    def __len__(self):
        return self.labels.__len__()

    def copy(self):
        return copy.deepcopy(self)
