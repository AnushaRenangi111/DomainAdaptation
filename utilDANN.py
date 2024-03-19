import torch
import numpy as np
import util
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------------
def get_dann_weights_filename(folder, from_dataset, to_dataset, config):
    return '{}{}/weights_dann_model_v{}_from_{}_to_{}_e{}_b{}.npy'.format(
                            folder,
                            ('/truncated' if config.truncate else ''),
                            str(config.model), from_dataset, to_dataset,
                            str(config.epochs), str(config.batch))

# ----------------------------------------------------------------------------
def batch_generator(x_data, y_data=None, batch_size=1, shuffle_data=True):
    len_data = len(x_data)
    index_arr = np.arange(len_data)
    if shuffle_data:
        np.random.shuffle(index_arr)

    start = 0
    while len_data > start + batch_size:
        batch_ids = index_arr[start:start + batch_size]
        start += batch_size
        if y_data is not None:
            x_batch = torch.tensor(x_data[batch_ids]).float()
            y_batch = torch.tensor(y_data[batch_ids]).long()
            yield x_batch, y_batch
        else:
            x_batch = torch.tensor(x_data[batch_ids]).float()
            yield x_batch

# ----------------------------------------------------------------------------
def train_dann_batch(dann_model, src_generator, target_generator, batch_size):
    dann_model.train()

    for batchXs, batchYs in src_generator:
        try:
            batchXd = next(target_generator)
        except: # Restart...
            target_generator = batch_generator(target_x_train, None, batch_size=batch_size // 2)
            batchXd = next(target_generator)

        combined_batchX = torch.cat((batchXs, batchXd))
        batch2Ys = torch.cat((batchYs, batchYs))
        batchYd = torch.cat((torch.tensor([[0, 1]]).repeat(batch_size // 2, 1),
                            torch.tensor([[1, 0]]).repeat(batch_size // 2, 1)))

        dann_model.zero_grad()

        output = dann_model(combined_batchX)
        label_output, domain_output = output['classifier_output'], output['domain_output']

        criterion_label = torch.nn.CrossEntropyLoss()
        criterion_domain = torch.nn.CrossEntropyLoss()

        loss_label = criterion_label(label_output, batch2Ys)
        loss_domain = criterion_domain(domain_output, batchYd)

        total_loss = loss_label + loss_domain
        total_loss.backward()
        dann_model.optimizer.step()

        result = total_loss.item()
    return result

# ----------------------------------------------------------------------------
def train_dann(dann_model, source_loader, target_loader, nb_epochs, batch_size, weights_filename,
               initial_hp_lambda=0.01, target_test_loader=None):

    print('Training DANN model')
    best_label_acc = 0
    dann_model.set_hp_lambda(initial_hp_lambda)

    for e in range(nb_epochs):
        src_generator = source_loader
        target_generator = target_loader

        # Update learning rates
        lr = dann_model.optimizer.param_groups[0]['lr']
        print(' - Lr:', lr, ' / Lambda:', dann_model.grl_layer.get_hp_lambda())

        dann_model.increment_hp_lambda_by(1e-4)

        # Train batch
        loss = train_dann_batch(dann_model, src_generator, target_generator, batch_size)

        saved = ""
        if best_label_acc <= label_acc:
            best_label_acc = label_acc
            torch.save(dann_model.state_dict(), weights_filename)
            saved = "SAVED"

        if target_test_loader is not None:
            target_loss, target_acc = dann_model.evaluate(target_test_loader)
        else:
            target_loss, target_acc = -1, -1

        print("Epoch [{}/{}]: source label loss = {:.4f}, acc = {:.4f} | target label loss = {:.4f}, acc = {:.4f} | {}".format(
                            e+1, nb_epochs, loss, label_acc, target_loss, target_acc, saved))
