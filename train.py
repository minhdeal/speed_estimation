import os
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

import constants
import custom_datasets
from model import SpeedCNN
from predict_and_evaluate import predict_and_evaluate

# set seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
# torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# python RNG
np.random.seed(seed)
random.seed(seed)


def train(epoch, device):
    print(f"running train epoch: {epoch}", flush=True)
    model_.train()
    train_loss = 0
    pbar = tqdm(total=train_length)
    for i, data in enumerate(train_dataloader):
        flow, label = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
        pred = model_(flow)

        loss = loss_fn(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item() * args.batch_size
        pbar.update(args.batch_size)

    train_loss_per_item = train_loss / train_length
    print(f"train_loss_per_item: {train_loss_per_item}")

    return train_loss_per_item


def validate(epoch, device):
    print(f"running validation epoch: {epoch}", flush=True)
    val_loss = 0
    model_.eval()
    pbar = tqdm(total=val_length)

    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            flow, label = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
            val_pred = model_(flow)
            val_loss += loss_fn(val_pred, label).item() * args.batch_size
            pbar.update(args.batch_size)

    val_loss_per_item = val_loss / val_length
    print(f"val_loss_per_item: {val_loss_per_item}")

    return val_loss_per_item


def save_model(model, epoch, train_loss, val_loss, run_folder, best_loss, save_all_epochs=False):
    train_results_file = os.path.join(run_folder, 'train_loss.txt')
    if not os.path.isfile(train_results_file):
        open(train_results_file, 'w+')
    with open(train_results_file, 'a') as f:
        f.write('epoch: {}, train_loss_per_item: {}'.format(epoch, train_loss))

    val_results_file = os.path.join(run_folder, 'val_loss.txt')
    if not os.path.isfile(val_results_file):
        open(val_results_file, 'w+')
    with open(val_results_file, 'a') as f:
        f.write('epoch: {}, val_loss_per_item: {}'.format(epoch, val_loss))

    if best_loss:
        print('best loss of {} updated in epoch {}'.format(best_loss, epoch))
        torch.save(model.state_dict(), os.path.join(run_folder, 'best_loss.pt'))
        print('saved model for epoch {}'.format(epoch))

    if save_all_epochs:
        saved_models_folder = os.path.join(run_folder, 'saved_models')
        os.makedirs(saved_models_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(saved_models_folder, '{}.pt'.format(epoch)))
        print('saved model and losses for epoch {}'.format(epoch))


def get_next_run_folder(params_folder):
    cur_run_folders = os.listdir(params_folder)
    last_run = 0
    for folder in cur_run_folders:
        if folder[:3] != 'run':
            continue
        if int(folder[3:]) > last_run:
            last_run = int(folder[3:])
    return os.path.join(params_folder, 'run{}'.format(last_run + 1))


def loss_plot(train_losses, val_losses, run_folder):
    print('saving loss plot\n')
    epoch_range = range(1, len(train_losses) + 1)
    plt.plot(epoch_range, train_losses, label='train')
    plt.plot(epoch_range, val_losses, label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss_per_item')
    plt.title('Train and validation loss per item (mse)')
    plt.legend()
    plot_file = os.path.join(run_folder, 'loss_plot.png')
    plt.savefig(plot_file)
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2*torch.cuda.device_count(),
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    args = parser.parse_args()

    train_length = len(os.listdir(constants.TRAIN_FLOW_FOLDER))
    val_length = len(os.listdir(constants.VAL_FLOW_FOLDER))

    train_dataset = custom_datasets.ImageDataset(constants.TRAIN_IMAGE_FOLDER, constants.TRAIN_FLOW_FOLDER, 
            constants.TRAIN_LABEL_FILE, first_file=1)
    val_dataset = custom_datasets.ImageDataset(constants.VAL_IMAGE_FOLDER, constants.VAL_FLOW_FOLDER, 
            constants.VAL_LABEL_FILE, first_file=len(os.listdir(constants.TRAIN_FLOW_FOLDER)) + 2)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    model_ = SpeedCNN()
    optimizer = torch.optim.Adam(model_.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()

    _DEVICE = torch.device("cuda:0")
    if torch.cuda.device_count() > 1:
        print('running on {} GPUs!'.format(torch.cuda.device_count()))
        model_ = torch.nn.DataParallel(model_)
    model_.to(_DEVICE)

    os.makedirs(constants.RESULTS_FOLDER, exist_ok=True)
    run_folder = get_next_run_folder(constants.RESULTS_FOLDER)

    best_loss = 99999
    train_losses = []
    val_losses = []
    for epoch in range(1, args.num_epochs + 1):
        try:
            print('\ntraining and validation for epoch: {}'.format(epoch))
            train_loss = train(epoch, _DEVICE)
            val_loss = validate(epoch, _DEVICE)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            os.makedirs(run_folder, exist_ok=True)
            if val_loss < best_loss:
                best_loss = val_loss
                save_model(model_, epoch, train_loss, val_loss, run_folder, best_loss)
            else:
                save_model(model_, epoch, train_loss, val_loss, run_folder, None)
        except:
            raise

    loss_plot(train_losses, val_losses, run_folder)

    predict_and_evaluate(constants.VAL_FLOW_FOLDER, constants.VAL_IMAGE_FOLDER, 
            run_folder, label_file=constants.VAL_LABEL_FILE)
    predict_and_evaluate(constants.TEST_FLOW_FOLDER, constants.TEST_IMAGE_FOLDER,
            run_folder)

