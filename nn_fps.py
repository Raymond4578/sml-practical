import argparse
import os
import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# check if required directories exists
if not os.path.exists("./nn_fps_models"):
    os.mkdir("./nn_fps_models")

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

# define the embeding module
class SparseEmbed1d(nn.Module):
    def __init__(self, input_size=1024, embed_size=256):
        super(SparseEmbed1d, self).__init__()
        # embed each input variable to a latent space of embed_size
        self.embeddings = nn.Embedding(num_embeddings=input_size, embedding_dim=embed_size)

    def forward(self, x):
        # find where the molecule has a certain substructure
        indices = x.nonzero(as_tuple=False)
        embed = self.embeddings(indices[:, 1])
        embed_sum = torch.zeros(x.size(0), embed.size(1), device=x.device)
        # add all the embedding up according to where a substructure exists for each molecule
        embed_sum.index_add_(0, indices[:, 0], embed)

        # calculate the number of 1 in each molecule
        count = torch.zeros(x.size(0), device=x.device)
        count.index_add_(0, indices[:, 0], torch.ones(indices.size(0), device=x.device))
        # avoid dividing 0
        count = count.unsqueeze(1) + 1e-6

        # get the average embedding
        embed_normalized = embed_sum / count
        return embed_normalized

class FeedForwardNet(nn.Module):
    def __init__(self, input_size=256, n_layers=1, unit_size=256, dropout_rate=0.1):
        super(FeedForwardNet, self).__init__()
        # hidden_layers build the hidden linear layers based on number of layers
        hidden_layers = [
            nn.Linear(input_size, n_layers * unit_size),
            nn.ReLU(inplace=True)
        ]
        for layer_idx in range(n_layers - 1, 0, -1):
            hidden_layers += [
                nn.Dropout(dropout_rate),
                nn.Linear((layer_idx + 1) * unit_size, layer_idx * unit_size),
                nn.ReLU(inplace=True)
            ]

        self.hidden = nn.Sequential(*hidden_layers)
        # final output linear layer
        self.linear = nn.Linear(unit_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        return self.linear(x)

# one ensemble model involving all the modules
class EmbeddingFFN(nn.Module):
    def __init__(
            self,
            input_size=1024, feat_size=256,
            ffn_n_layers=1, ffn_unit_size=256,
            dropout_rate=0.1
    ):
        super(EmbeddingFFN, self).__init__()
        self.sparse_embedding = SparseEmbed1d(input_size=input_size, embed_size=feat_size)
        self.ffn = FeedForwardNet(
            input_size=feat_size, n_layers=ffn_n_layers, unit_size=ffn_unit_size, dropout_rate=dropout_rate
        )

    def forward(self, x):
        embed_x = self.sparse_embedding(x)
        return self.ffn(embed_x)

###########################
# model training
###########################

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_mps', action='store_true', default=False,
                    help='Disables MPS training.')
parser.add_argument('--seed', type=int, default=666, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--decay_epochs', type=int, default=400, help='Number of epochs to start lr decay.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
# model parameters
parser.add_argument('--feat_size', type=int, default=256, help='The feature size for the data.')
parser.add_argument('--ffn_n_layers', type=int, default=3, help='The number of layers for the feedforward net.')
parser.add_argument('--ffn_unit_size', type=int, default=256, help='The unit size of feedforward net layers.')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='The dropout rate in the model.')

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

embed_data = load_data(["fps"])
X, y = embed_data["fps"]["train"]

# shuffle and split the training set to training and validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=args.seed, shuffle=True
)

X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_valid = torch.tensor(X_valid.to_numpy(), dtype=torch.float32)
y_valid = torch.tensor(y_valid, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

# prepare the data for training
train_loader = DataLoader(train_dataset, batch_size=32)

#####################
# setup model training
#####################

# Initialize model
model = EmbeddingFFN(
    input_size=X_train.shape[1], feat_size=args.feat_size,
    ffn_n_layers=args.ffn_n_layers, ffn_unit_size=args.ffn_unit_size,
    dropout_rate=args.dropout_rate
)

# Set up loss function
mse_loss = torch.nn.MSELoss()

# if use mps, move all the data into GPU memory
if args.mps:
    print('mps used.')
    model.to("mps")
    mse_loss.to("mps")

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

def train(train_loader):
    for i, (X_train, y_train) in enumerate(train_loader):
        if args.mps:
            X_train = X_train.to("mps")
            y_train = y_train.to("mps")
        # start training
        y_pred = model(X_train)
        # calculate loss
        total_loss = mse_loss(y_pred, y_train)

        # back propagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def validation(valid_dataset, epoch, t):
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
        f'MAE: {mae_calculator(y_hat, y_valid):.4f},',
        f'Time: {time.time() - t:.4f}s.'
    )

# run the training by a certain number of epochs
train_start_time = time.time()
t = time.time()
for epoch in range(args.epochs):
    train(train_loader)
    scheduler.step()

    if epoch != 0 and epoch % 10 == 9:
        validation(valid_dataset, epoch, t)
        t = time.time()

print(f"The training takes {time.time() - train_start_time:.4f}s.")

# save the model
model_path = ("./nn_fps_models/" +
    f"EmbeddingFFN_seed_{args.seed}_{time.strftime("%y%m%d%H%M")}.pt")
torch.save(model, model_path)

#####################
# test the model
#####################

X_test, y_test = embed_data["fps"]["test"]
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
        f'MAE: {mae_calculator(y_hat, y_test):.4f}.',
    )

print()
print("------------------------------------------------------------")
print("Performance on test set:")
print("------------------------------------------------------------")
test(test_dataset)