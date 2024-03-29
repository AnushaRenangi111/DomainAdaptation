import sys
import os
import warnings
import numpy as np
import argparse
import gc
import matplotlib

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

import util
import utilLoad
import utilCNN
import utilDANN
import utilDANNModel

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set your desired GPU index here
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable Tensorflow CUDA load statements
warnings.filterwarnings('ignore')

util.init()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

WEIGHTS_CNN_FOLDERNAME = 'WEIGHTS_CNN'
WEIGHTS_DANN_FOLDERNAME = 'WEIGHTS_DANN'
util.mkdirp(WEIGHTS_CNN_FOLDERNAME + '/truncated')
util.mkdirp(WEIGHTS_DANN_FOLDERNAME + '/truncated')

def train_cnn(datasets, input_shape, num_labels, WEIGHTS_CNN_FOLDERNAME, config):
    utilCNN.train_cnn(datasets, input_shape, num_labels, WEIGHTS_CNN_FOLDERNAME, config)

def train_dann(datasets, input_shape, num_labels, weights_foldername, config):
    for i in range(len(datasets)):
        if config.from_db is not None and config.from_db != datasets[i]['name']:
            continue
        for j in range(len(datasets)):
            if i == j or (config.to_db is not None and config.to_db != datasets[j]['name']):
                continue
            print(80 * '-')
            print('SOURCE: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
                datasets[i]['name'], datasets[i]['x_train'].shape, datasets[i]['y_train'].shape,
                datasets[i]['x_test'].shape, datasets[i]['y_test'].shape))
            print('TARGET: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
                datasets[j]['name'], datasets[j]['x_train'].shape, datasets[j]['y_train'].shape,
                datasets[j]['x_test'].shape, datasets[j]['y_test'].shape))
            dann = utilDANNModel.DANNModel(config.model, input_shape, num_labels, config.batch)
            weights_filename = utilDANN.get_dann_weights_filename(weights_foldername,
                                                                   datasets[i]['name'], datasets[j]['name'], config)
            if not config.load:
                utilDANN.train_dann(dann,
                                    datasets[i]['x_train'], datasets[i]['y_train'],
                                    datasets[j]['x_train'], datasets[j]['y_train'],
                                    config.epochs, config.batch, weights_filename)
            print('# Evaluate...')
            dann.load(weights_filename)
            source_loss, source_acc = dann.label_model.evaluate(datasets[i]['x_test'], datasets[i]['y_test'],
                                                                batch_size=32, verbose=0)
            target_loss, target_acc = dann.label_model.evaluate(datasets[j]['x_test'], datasets[j]['y_test'],
                                                                batch_size=32, verbose=0)
            print('Result: {}\t{}\t{:.4f}\t{:.4f}'.format(datasets[i]['name'], datasets[j]['name'],
                                                           source_acc, target_acc))
            gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DANN')
    group0 = parser.add_argument_group('Training type')
    group0.add_argument('-type', default='dann', type=str, choices=['dann', 'cnn'], help='Training type')
    group0.add_argument('-model', default=1, type=int, help='Model number (1,2,3,4)')
    group1 = parser.add_argument_group('Input data parameters')
    group1.add_argument('-db', default='mnist', type=str, help='Dataset',
                        choices=['mnist', 'signs'])
    group1.add_argument('-select', default=None, type=str,
                        help='Comma separated list of BD names to load. Set None to load all.')
    group1.add_argument('-from', default=None, dest='from_db', type=str,
                        help='Use only this dataset as source.')
    group1.add_argument('-to', default=None, dest='to_db', type=str,
                        help='Use only this dataset as target.')
    group1.add_argument('-norm', default='standard', type=str,
                        choices=['mean', 'standard', '255', 'keras'], help='Input data normalization type')
    group1.add_argument('-size', default=-1, type=int, help='Scale to this size. -1 to use default size.')
    group3 = parser.add_argument_group('Training parameters')
    group3.add_argument('-e', default=200, type=int, dest='epochs', help='Number of epochs')
    group3.add_argument('-b', default=128, type=int, dest='batch', help='Batch size')
    parser.add_argument('-lda', default=0.01, type=float, help='Reversal gradient lambda')
    parser.add_argument('-lr', default=1.0, type=float, help='Learning rate')
    parser.add_argument('--lsmooth', action='store_true', help='Activate label smoothing.')
    parser.add_argument('--tsne', action='store_true', help='Activate TSNE')
    parser.add_argument('--aug', action='store_true', help='Use data augmentation')
    parser.add_argument('--load', action='store_true', help='Load weights.')
    parser.add_argument('--truncate', action='store_true', help='Truncate datasets.')
    parser.add_argument('--v', action='store_true', help='Activate verbose.')
    parser.add_argument('-gpu', default='0', type=str, help='GPU')
    args = parser.parse_args()
    if args.select is not None and args.select.lower() != 'none':
        args.select = args.select.split(',')
    else:
        args.select = []
    if args.from_db is not None and args.from_db.lower() == 'none':
        args.from_db = None
    if args.to_db is not None and args.to_db.lower() == 'none':
        args.to_db = None
    assert isinstance(args.select, (list, tuple, np.ndarray)) and (len(args.select) == 0 or len(args.select) == 2)
    datasets, input_shape, num_labels = utilLoad.load(args)
    print('CONFIG:\n -', str(args).replace('Namespace(', '').replace(')', '').replace(', ', '\n - '))
    print(' - Input shape:', input_shape)
    print(' - Num labels:', num_labels)
    if args.type == 'cnn':
        print('Train CNN...')
        train_cnn(datasets, input_shape, num_labels, WEIGHTS_CNN_FOLDERNAME, args)
    elif args.type == 'dann':
        print('Train DANN...')
        train_dann(datasets, input_shape, num_labels, WEIGHTS_DANN_FOLDERNAME, args)
    else:
        raise Exception('Unknown training type')
