import torch
import utilModels
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import os

# ----------------------------------------------------------------------------
def get_cnn_weights_filename(weights_folder, dataset_name, config):
    return '{}{}/weights_cnn_model_v{}_for_{}_e{}_b{}{}.npy'.format(
                            weights_folder, ('/truncated' if config.truncate else ''),
                            str(config.model), dataset_name,
                            str(config.epochs), str(config.batch),
                            ('_aug' if config.aug else ''))

# -------------------------------------------------------------------------
def label_smoothing_loss(y_true, y_pred):
    criterion = nn.BCEWithLogitsLoss()
    return criterion(y_pred, y_true)

# -------------------------------------------------------------------------
'''Create the source or label model separately'''
def build_source_model(model_number, input_shape, nb_classes, config):
    auxModel = getattr(utilModels, "ModelV" + str(model_number))(input_shape)

    net = auxModel.get_model_features()
    net = auxModel.get_model_labels(net)
    net = nn.Sequential(net, nn.Linear(net[-1].out_features, nb_classes))

    if config.v == True:
        print(net)

    return net

# ----------------------------------------------------------------------------
def train_cnn_on_one_dataset(model, source_loader, test_loader, weights_filename, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    early_stopping = None
    if config.patience > 0:
        early_stopping = EarlyStopping(patience=config.patience)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in source_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = label_smoothing_loss(labels, outputs)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(source_loader.dataset)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, config.epochs, epoch_loss))

        if early_stopping is not None and early_stopping.check_stop(epoch_loss):
            print(' - Early stopping!')
            break

    torch.save(model.state_dict(), weights_filename)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# ----------------------------------------------------------------------------
def train_cnn(datasets, input_shape, num_labels, weights_folder, config):
    transform = transforms.Compose([
        transforms.Resize(input_shape[1:]),
        transforms.ToTensor()
    ])

    for i in range(len(datasets)):
        if config.from_db is not None and config.from_db != datasets[i]['name']:
            continue

        source_dataset = ImageFolder(datasets[i]['path'], transform=transform)
        source_loader = DataLoader(source_dataset, batch_size=config.batch, shuffle=True)

        test_dataset = ImageFolder(datasets[i]['path'], transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=config.batch, shuffle=False)

        model = build_source_model(config.model, input_shape, num_labels, config)

        print('BD: {} \tx_train:{}\ty_train:{}\tx_test:{}\ty_test:{}'.format(
                    datasets[i]['name'],
                    len(source_loader.dataset), len(source_loader.dataset),
                    len(test_loader.dataset), len(test_loader.dataset)))

        weights_filename = get_cnn_weights_filename(weights_folder, datasets[i]['name'], config)

        if config.load == False:
            train_cnn_on_one_dataset(model, source_loader, test_loader, weights_filename, config)

        # Final evaluation
        print(80*'-')
        print('FINAL VALIDATION:')
        model.load_state_dict(torch.load(weights_filename))
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

