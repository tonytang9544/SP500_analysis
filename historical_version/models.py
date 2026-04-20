import torch
from torch.nn.utils.parametrizations import weight_norm
import numpy as np
import os
from data_utils import daily_changes
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F



class MLP_predictor(torch.nn.Module):
    def __init__(self, number_stocks, hidden_dim=256, num_hidden_layers=3, activation=torch.nn.Tanh(), dropout_p=0.1):
        super().__init__()
        self.projection = weight_norm(torch.nn.Linear(number_stocks, hidden_dim))
        self.layers = torch.nn.ModuleList([weight_norm(torch.nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_hidden_layers)])
        self.output = weight_norm(torch.nn.Linear(hidden_dim, number_stocks))
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_p)
    
    def forward(self, state):
        '''
        state is vector of one-day price change
        '''
        x = self.projection(state)
        for each_layer in self.layers:
            x = self.dropout(self.activation(each_layer(x)))
        return self.output(x)


class SeqAnalyser(torch.nn.Module):
    def __init__(self, num_stocks, time_window, time_feature_dim=64, stock_feature_dim=128, num_stock_layer=2, activation=torch.nn.Tanh(), dropout_p=0.1):
        super().__init__()
        self.time_layer = weight_norm(torch.nn.Linear(time_window, time_feature_dim))
        self.stock_pred = weight_norm(torch.nn.Linear(num_stocks, stock_feature_dim))
        self.stock_hidden_layers = torch.nn.ModuleList([weight_norm(torch.nn.Linear(stock_feature_dim, stock_feature_dim)) for _ in range(num_stock_layer)])
        self.time_squeeze = weight_norm(torch.nn.Linear(time_feature_dim, 1))
        self.output_pred = weight_norm(torch.nn.Linear(stock_feature_dim, num_stocks))
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, input):
        '''
        input is array (num_stocks, time_window)
        output the predicted vector of one-day price change (num_stocks, )
        '''
        x = input
        x = self.activation(self.time_layer(x))
        x = x.transpose(-1, -2)
        x = self.activation(self.stock_pred(x))
        for layer in self.stock_hidden_layers:
            x = self.activation(self.dropout(layer(x)))
        x = x.transpose(-1, -2)
        x = self.activation(self.time_squeeze(x))
        x = torch.squeeze(x)
        return self.output_pred(x)


class StockDataset(Dataset):
    def __init__(self, numpy_array, time_window=1):
        self.data = numpy_array
        self.time_window = time_window
        assert time_window >= 1 and isinstance(time_window, int), "invalid time_window"

    def __len__(self,):
        return self.data.shape[-1] - self.time_window
    
    def __getitem__(self, idx):
        return self.data[:, idx:idx+self.time_window], self.data[:, idx+self.time_window]



def load_models(folder, number_stocks):
    for root, dirs, files in os.walk(folder):
        models = []
        for file in files:
            if file.endswith(".pt"):
                model = MLP_predictor(number_stocks)
                model.load_state_dict(torch.load(os.path.join(root, file)))
                models.append(model)

    return models


def stock_train_test_split(data, train_proportion=0.8):

    # delta = daily_changes(market_database)
   
    train_idx = int(data.shape[-1] * train_proportion)

    train_data = data[:, :train_idx].astype(np.float32)
    test_data = data[:, train_idx:].astype(np.float32)

    return train_data, test_data


def train(model, train_data, test_data, device="cpu", epochs=100, loss_fn=torch.nn.MSELoss(), print_log=False):

    train_data = torch.tensor(train_data).transpose(1, 0).to(device)
    test_data = torch.tensor(test_data).transpose(1, 0).to(device)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    train_losses = []
    test_losses = []
    
    for i in range(epochs):

        model.train()
        optimizer.zero_grad()
        prediction = model(train_data)[:-1, :]
        loss = loss_fn(prediction, train_data[1:, :])
        # loss = ce_loss(prediction, train_data[1:, :].argmax(dim=1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss = loss.detach().cpu().numpy()
        train_losses.append(train_loss)

        model.eval()
        prediction = model(test_data)[:-1, :]
        loss = loss_fn(prediction, test_data[1:, :])
        test_loss = loss.detach().cpu().numpy()
        test_losses.append(test_loss)

    if print_log:
        plt.plot(train_losses, label="train losses")
        plt.plot(test_losses, label="test losses")
        plt.yscale("log")
        plt.xlabel("epoches")
        plt.legend()
        plt.show()

def supervised_train(model, train_dataloader, test_dataloader, device="cpu", epochs=100, optimizer_lr=3e-3, loss_fn=torch.nn.MSELoss(), print_log=False, scheduler_step_size=5, scheduler_gamma=0.8):

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    train_loss = []
    test_loss = []

    for i in range(epochs):

        epoch_train_loss = []
        epoch_test_loss = []

        model.train()
        optimizer.zero_grad()
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            prediction = model(X)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.detach().cpu().numpy())
            scheduler.step()
        train_loss.append(epoch_train_loss)
        if print_log:
            print(f"train loss for epoch {i} = {np.mean(epoch_train_loss)}")

        model.eval()
        for X, y in test_dataloader:
            if isinstance(X, np.ndarray):
                X = torch.Tensor(X)
            if isinstance(y, np.ndarray):
                y = torch.Tensor(y)
            X = X.to(device)
            y = y.to(device)
            prediction = model(X)
            loss = loss_fn(prediction, y)
            epoch_test_loss.append(loss.detach().cpu().numpy())
        test_loss.append(epoch_test_loss)
        if print_log:
            print(f"test loss for epoch {i} = {np.mean(epoch_test_loss)}")
    
    return train_loss, test_loss


def model_predictions(model, test_set, print_log=False, device="cpu"):
    test_set = torch.tensor(test_set).transpose(1, 0).to(device)
    model.to(device)
    model.eval()
    prediction = model(test_set)
    pred = prediction.detach().cpu().numpy()
    pred = np.argmax(pred, axis=-1)
    
    truth = test_set.detach().cpu().numpy()
    truth = np.argmax(truth, axis=-1)
    if print_log:
        print(pred)
        print(truth)
    return pred, truth


def model_predictions_using_test_dataloader(model, test_dataloader, print_log=False, device="cpu", time_window=1):

    model.to(device)
    model.eval()

    predictions = []
    truths = []

    for X, y in test_dataloader:
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X)
        if isinstance(y, np.ndarray):
            y = torch.Tensor(y)
        X = X.to(device)
        prediction = model(X)
        predictions.append(np.argmax(prediction.detach().cpu().numpy(), axis=-1))

        truths.append(np.argmax(y.detach().cpu().numpy(), axis=-1))

    predictions = np.append(np.stack(predictions[:-1]).flatten(), predictions[-1])
    truths = np.append(np.stack(truths[:-1]).flatten(), truths[-1])
    if print_log:
        print(predictions)
        print(truths)
    return predictions, truths


def mse_ce_loss_fn(prediction, truth):
    pred = prediction.clone()
    true = truth.clone()

    pred[pred>0] = 1
    pred[pred<=0] = 0
    true[true>0] = 1
    true[true<=0] = 0

    return F.mse_loss(prediction, truth) + F.binary_cross_entropy(pred, true)