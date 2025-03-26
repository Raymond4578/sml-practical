import argparse
import os
import time
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# check if required directories exists
if not os.path.exists("./nn_embed_models"):
    os.mkdir("./nn_embed_models")

###########################
# utility setup
###########################

# a function that load the data
def load_data(feature: list, data_path="./data/", shuffle=False):
    # The training data and the public test set for the embed representation
    data_dict = dict()
    y_train = pd.read_csv(
        data_path + 'y_train.csv', index_col=0
    ).to_numpy()  # outputs of the training set
    y_public_test = pd.read_csv(
        data_path + 'y_public_test.csv', index_col=0
    ).to_numpy()  # outputs of the public test set
    if shuffle:
        np.random.seed(8888)
        random_idx = np.random.permutation(y_train.shape[0])
        y_train = y_train[random_idx]

    if not isinstance(feature, list):
        feature = [feature]
    for f in feature:
        X_train = pd.read_csv(
            data_path + f'X_{f}_train.csv', index_col=0
        )  # inputs of the training set
        if shuffle:
            X_train = X_train.iloc[random_idx].reset_index(drop=True)
        X_public_test = pd.read_csv(
            data_path + f'X_{f}_public_test.csv', index_col=0
        )  # inputs of the public test set
        data_dict[f] = {
            "train": (X_train, y_train),
            "test": (X_public_test, y_public_test)
        }
    return data_dict

def mse_calculator(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

def mae_calculator(y_pred, y_true):
    return torch.abs(y_pred - y_true).mean()

##############################
# Define Model Structure
##############################

class FeedForwardNet(nn.Module):
    def __init__(
            self, input_size=256, n_layers=1, unit_size=256, dropout_rate=0.5
    ):
        super(FeedForwardNet, self).__init__()
        # hidden_layers build the hidden linear layers based on number of layers
        hidden_layers = [
            nn.Linear(input_size, (2 ** (n_layers - 1)) * unit_size),
            nn.ReLU(inplace=True)
        ]
        for layer_idx in range(n_layers - 1, 0, -1):
            hidden_layers += [
                nn.Dropout(dropout_rate),
                nn.Linear(
                    unit_size * (2 ** layer_idx), unit_size * (2 ** (layer_idx - 1))
                ),
                nn.ReLU(inplace=True)
            ]
        self.hidden = nn.Sequential(*hidden_layers)
        # final output linear layer
        self.linear = nn.Linear(unit_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        return self.linear(x)

###########################
# model training
###########################

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_mps', action='store_true', default=False,
                    help='Disables MPS training.')
parser.add_argument('--seed', type=int, default=666, help='Random seed.')
parser.add_argument(
    '--epochs', type=int, default=1000, help='Number of epochs to train.'
)
parser.add_argument(
    '--decay_epochs', type=int, default=700,
    help='Number of epochs to start lr decay.'
)
parser.add_argument(
    '--lr', type=float, default=1e-4, help='Initial generator learning rate.'
)
# parameters for FeedForwardNet model
parser.add_argument(
    '--ffn_n_layers', type=int, default=3,
    help='The number of hidden layers for FeedForwardNet model.'
)
parser.add_argument(
    '--ffn_unit_size', type=int, default=256,
    help='The unit size of hidden layers for FeedForwardNet model.'
)
parser.add_argument(
    '--dropout_rate', type=float, default=0.5,
    help='The dropout rate in the model.'
)

args = parser.parse_args()
# add a variable for whether use mps
args.mps = not args.no_mps and torch.backends.mps.is_available()

# set random seeds for the training
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

#####################
# load the data
#####################

embed_data = load_data(["embed"], shuffle=True)
X, y = embed_data["embed"]["train"]

# shuffle and split the training set to training and validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=args.seed
)

X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_valid = torch.tensor(X_valid.to_numpy(), dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

# prepare the data for training
train_loader = DataLoader(train_dataset,  batch_size=32, shuffle=True)

#####################
# setup model training
#####################

# Initialize model
model = FeedForwardNet(
    input_size=X_train.shape[1], n_layers=args.ffn_n_layers,
    unit_size=args.ffn_unit_size, dropout_rate=args.dropout_rate
)

# set up loss function
loss = torch.nn.MSELoss()

# if use mps, move all the data into GPU memory
if args.mps:
    print('mps used.')
    model.to("mps")
    loss.to("mps")

# set up optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
# set up learning rate decay
def lr_lambda(epoch):
    # before a certain epoch, lr does not change
    if epoch < args.decay_epochs:
        return 1.0
    # after a certain epoch, the lr decrease linearly to 0
    return 1.0 - (epoch - args.decay_epochs) / (args.epochs - args.decay_epochs)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

#####################
# train the model
#####################

# define a model for training one epoch
def train(train_loader):
    for i, (X_train, y_train) in enumerate(train_loader):
        if args.mps:
            X_train = X_train.to("mps")
            y_train = y_train.to("mps")
        # start training
        y_pred = model(X_train)
        # calculate loss
        total_loss = loss(y_pred, y_train)
        # back propagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

def validation(valid_dataset):
    # set to evaluation mode
    model.eval()
    X_valid, y_valid = valid_dataset.tensors
    if args.mps:
        X_valid = X_valid.to("mps")
        y_valid = y_valid.to("mps")
    y_hat = model(X_valid)
    model.train()
    print(
        f'Epoch: {epoch + 1:03d}/{args.epochs},',
        f'MSE: {mse_calculator(y_hat, y_valid):.4f},',
        f'MAE: {mae_calculator(y_hat, y_valid):.4f}.',
        f'Time: {time.time() - t:.4f}s.'
    )

# run the training by a certain number of epochs
train_start_time = time.time()
t = time.time()
for epoch in range(args.epochs):
    epoch_start_time = time.time()
    train(train_loader)
    scheduler.step()

    if epoch != 0 and epoch % 10 == 9:
        validation(valid_dataset)
        t = time.time()
print(f"The training takes {time.time() - train_start_time:.4f}s.")

# save the model
model_path = ("./nn_embed_models/" +
    f"FFN_seed_{args.seed}_{time.strftime("%y%m%d%H%M")}.pt")
torch.save(model, model_path)

#####################
# test the model
#####################

X_test, y_test = embed_data["embed"]["test"]
X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
test_dataset = TensorDataset(X_test, y_test)

def test(test_dataset):
    model.eval()
    X_test, y_test = test_dataset.tensors
    if args.mps:
        X_test = X_test.to("mps")
        y_test = y_test.to("mps")
    y_hat = model(X_test)
    print(
        f'MSE: {mse_calculator(y_hat, y_test):.4f},',
        f'MAE: {mae_calculator(y_hat, y_test):.4f}.'
    )

print()
print("------------------------------------------------------------")
print("Performance on test set:")
print("------------------------------------------------------------")
test(test_dataset)