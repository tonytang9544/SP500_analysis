from dataset_utils import get_csv_list, batch_load
from models import MLP_predictor
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchinfo import summary

train_config_dict = {
    "data_folder": "/Users/tangm/Downloads/github/market/data",
    "train_split": 0.8,
    "val_split": 0.1,
    "num_epochs": 500,
    'learning_rate': 1e-2,
    "num_of_stocks": 10,
    "model_dim": 1024,
    "num_layers": 3,
    "dataset_as_delta": True,
    "has_scheduler": True,
    "scheduler_step_size": 20,
    "scheduler_gamma": 0.9
}

print(f"training parameters: {train_config_dict}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# get the first 8 stocks from the dataset, and re-order the dataset into ascending time
dataset = np.flip(batch_load(get_csv_list(train_config_dict["data_folder"]), pd_df=True, as_delta=train_config_dict["dataset_as_delta"]).to_numpy(), axis=0)[:, :train_config_dict["num_of_stocks"]]

total_dataset_length = dataset.shape[0]
train_index = int(train_config_dict["train_split"] * total_dataset_length)
val_index = train_index + int(train_config_dict["val_split"] * total_dataset_length)

train_set = torch.tensor(dataset[:train_index].copy(), dtype=torch.float32)
val_set = torch.tensor(dataset[train_index:val_index].copy(), dtype=torch.float32)
test_set = torch.tensor(dataset[val_index:].copy(), dtype=torch.float32)

# print(train_set.shape)

loss_fn = torch.nn.MSELoss()

model = MLP_predictor(
    number_stocks=train_config_dict["num_of_stocks"],
    hidden_dim=train_config_dict["model_dim"],
    num_hidden_layers=train_config_dict["num_layers"]
)
summary(model)


optimiser = torch.optim.Adam(model.parameters(), lr=train_config_dict["learning_rate"])
if train_config_dict["has_scheduler"]:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, 
        step_size=train_config_dict["scheduler_step_size"], 
        gamma=train_config_dict["scheduler_gamma"]
    )

model = model.to(device)
train_set = train_set.to(device=device, dtype=torch.float32)
val_set = val_set.to(device=device, dtype=torch.float32)

train_losses = []
val_losses = []
lowest_val_loss = np.inf

for i in range(train_config_dict['num_epochs']):
    model.train()
    optimiser.zero_grad()

    output = model(train_set[:-2])

    loss = loss_fn(train_set[2:], output)

    loss.backward()
    optimiser.step()
    if train_config_dict["has_scheduler"]:
        scheduler.step()

    ave_train_loss = loss.item()/train_set.shape[0]
    train_losses.append(ave_train_loss)
    print(f"Epoch {i + 1} / {train_config_dict['num_epochs']} train loss: {ave_train_loss}")

    model.eval()
    
    output = model(val_set[:-2])
    loss = loss_fn(val_set[2:], output)

    ave_val_loss = loss.item()/val_set.shape[0]
    val_losses.append(ave_val_loss)
    print(f"Epoch {i + 1} / {train_config_dict['num_epochs']} val loss: {ave_val_loss}")

    if ave_val_loss < lowest_val_loss:
        lowest_val_loss = ave_val_loss
        torch.save(model, "model_save.pt")

model = torch.load("model_save.pt", weights_only=False)
model.eval()
test_set = test_set.to(device=device, dtype=torch.float32)

test_losses = []

output = model(test_set[:-2])
loss = loss_fn(test_set[2:], output)

ave_test_loss = loss.item()/test_set.shape[0]
test_losses.append(ave_test_loss)
print(f"average test loss: {ave_test_loss}")



# visualise predictions
input = test_set[:-2]


predictions = model(input)

predictions = np.array(predictions.detach().cpu())
test_set = test_set[2:].cpu()

for i in range(train_config_dict["num_of_stocks"]):
    plt.plot(predictions[:, i], label="predictions")
    plt.plot(test_set[:, i], label="truth")
    plt.legend()
    plt.savefig(f"pred_vs_actual for stock {i}")
    plt.cla()
    plt.close()
