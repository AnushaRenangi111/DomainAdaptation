from __future__ import print_function
import numpy as np
import cv2
import os
import warnings
import scipy.io
import pickle as pkl
import skimage.transform
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

warnings.filterwarnings('ignore')

ARRAY_BD_NAMES = ['gtsrb', 'syn_signs']

VERBOSE_NB_SHOW = 20

class TrafficSignsDataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        self.data = self.__read_gtsrb_train_images(data_path)
        self.labels = self.__read_labels(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(self.data[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __read_gtsrb_train_images(self, path):
        images = []
        for c in range(0, 43):
            subpath = os.path.join(path, format(c, '05d'))
            for fname in os.listdir(subpath):
                if fname.endswith('.ppm'):
                    images.append(os.path.join(subpath, fname))
        return images

    def __read_labels(self, path):
        df = pd.read_csv(path, sep=';')
        return df['ClassId'].values

def load_dataset_gtsrb(data_path, labels_path, img_size, verbose):
    dataset = TrafficSignsDataset(data_path, labels_path)
    if verbose:
        for i in range(VERBOSE_NB_SHOW):
            image, label = dataset[i]
            print('Label:', label)
            cv2.imshow("Img", image)
            cv2.waitKey(0)

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    train_index, test_index = next(skf.split(dataset.data, dataset.labels))

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

class SynSignsDataset(Dataset):
    def __init__(self, data_path, labels_path, transform=None):
        self.data = self.__read_syn_signs_images(data_path)
        self.labels = self.__read_labels(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(self.data[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __read_syn_signs_images(self, path):
        images = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.png'):
                    images.append(os.path.join(root, file))
        return images

    def __read_labels(self, path):
        df = pd.read_csv(path, sep=' ', header=None)
        return df[1].values

def load_dataset_syn_signs(data_path, labels_path, img_size, verbose):
    transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor()])
    dataset = SynSignsDataset(data_path, labels_path, transform=transform)
    if verbose:
        for i in range(VERBOSE_NB_SHOW):
            image, label = dataset[i]
            print('Label:', label)
            cv2.imshow("Img", image.permute(1, 2, 0).numpy())
            cv2.waitKey(0)

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    train_index, test_index = next(skf.split(dataset.data, dataset.labels))

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

def load_datasets(selected_bds, scale_to, verbose):
    return_array = []

    if selected_bds is None or len(selected_bds) == 0:
        selected_bds = ARRAY_BD_NAMES

    img_size = 40
    if scale_to > 0:
        img_size = scale_to

    for bd_name in selected_bds:
        assert bd_name in ARRAY_BD_NAMES
        print('Loading', bd_name, 'dataset...')
        if bd_name == 'gtsrb':
            data_path = os.path.join('datasets/GTSRB', 'Final_Training/Images')
            labels_path = os.path.join('datasets/GTSRB', 'Final_Test/GT-final_test.csv')
            train_loader, test_loader = load_dataset_gtsrb(data_path, labels_path, img_size, verbose)
        elif bd_name == 'syn_signs':
            data_path = os.path.join('datasets/SYN_SIGNS/synthetic_data', 'train')
            labels_path = os.path.join('datasets/SYN_SIGNS/synthetic_data', 'train_labelling.txt')
            train_loader, test_loader = load_dataset_syn_signs(data_path, labels_path, img_size, verbose)
        else:
            raise ValueError("Invalid dataset name:", bd_name)
        return_array.append({'name': bd_name,
                             'train_loader': train_loader, 'test_loader': test_loader})

    return return_array
