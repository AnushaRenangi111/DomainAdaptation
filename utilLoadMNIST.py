from __future__ import print_function
import numpy as np
import cv2
import os
import scipy.io
import pickle as pkl
import skimage.transform
from sklearn.datasets import load_svmlight_file
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

ARRAY_BD_NAMES = ['mnist', 'mnist_m', 'svhn', 'syn_numbers']

VERBOSE_NB_SHOW = 20


# ----------------------------------------------------------------------------
def resize_array_images(array_images, size):
    new_array = []
    for i in range(len(array_images)):
        img = cv2.resize(array_images[i], (size, size), interpolation=cv2.INTER_CUBIC)
        new_array.append(img)
    return np.array(new_array)


# ----------------------------------------------------------------------------
# MNIST
def load_dataset_mnist(img_size, verbose):
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    x_train = mnist_train.data.numpy()[:, :, :, None]
    y_train = mnist_train.targets.numpy()
    x_test = mnist_test.data.numpy()[:, :, :, None]
    y_test = mnist_test.targets.numpy()

    if img_size != 28:
        x_train = resize_array_images(x_train, img_size)
        x_test = resize_array_images(x_test, img_size)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# MNIST-M
def load_dataset_mnist_m(img_size, verbose):
    with open('datasets/dataset_target_mnist_data.pkl', 'rb') as f:
        mnistm_mnist = pkl.load(f, encoding='latin1')

    x_train = mnistm_mnist['x_train']
    y_train = mnistm_mnist['y_train']
    x_test = mnistm_mnist['x_test']
    y_test = mnistm_mnist['y_test']

    if img_size != 28:
        x_train = resize_array_images(x_train, img_size)
        x_test = resize_array_images(x_test, img_size)

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# SVHN dataset
def load_dataset_svhn(img_size, verbose):
    svhn_train = datasets.SVHN(root='./data', split='train', download=True, transform=transforms.ToTensor())
    svhn_test = datasets.SVHN(root='./data', split='test', download=True, transform=transforms.ToTensor())

    x_train = svhn_train.data.numpy().transpose((0, 2, 3, 1))
    y_train = svhn_train.labels
    x_test = svhn_test.data.numpy().transpose((0, 2, 3, 1))
    y_test = svhn_test.labels

    x_train = resize_array_images(x_train, img_size)
    x_test = resize_array_images(x_test, img_size)

    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# Synth numbers or Synth
def load_dataset_syn_numbers(img_size, verbose):
    mat_train = scipy.io.loadmat('datasets/SynthDigits/synth_train_32x32.mat')
    mat_test = scipy.io.loadmat('datasets/SynthDigits/synth_test_32x32.mat')

    x_train = mat_train['X']
    y_train = mat_train['y'].flatten()
    x_test = mat_test['X']
    y_test = mat_test['y'].flatten()

    x_train = np.rollaxis(x_train, 3, 0)
    x_test = np.rollaxis(x_test, 3, 0)

    x_train = resize_array_images(x_train, img_size)
    x_test = resize_array_images(x_test, img_size)

    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


# ----------------------------------------------------------------------------
# Parameter:
# - selected_bds = array of bd names to load. None to load all them
def load_datasets(selected_bds, scale_to, verbose):
    return_array = []

    if selected_bds is None or len(selected_bds) == 0:
        selected_bds = ARRAY_BD_NAMES

    img_size = 28
    if scale_to > 0:
        img_size = scale_to

    for bd_name in selected_bds:
        assert bd_name in ARRAY_BD_NAMES
