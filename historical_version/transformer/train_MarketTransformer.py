from dataset_utils import get_csv_list, batch_load, assemble_dataset
from transformerModel import MarketTransformerEncoder
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchinfo import summary

train_config_dict = {
    "data_folder": "/Users/tangm/Downloads/github/market/data",
    "train_split": 0.8,
    "val_split": 0.1,
    "num_epochs": 50,
    "look_back_time_window": 129,
    'learning_rate': 1e-3,
    "optimiser_step_size": 6,
    "optimiser_gamma": 0.5,
    "num_of_stocks": 10,
    "model_dim": 64,
    "feed_forward_dim": 256,
    "num_heads": 2,
    "encoder_layers": 6,
    "dataset_as_delta": False
}

print(f"training parameters: {train_config_dict}")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# get the first 8 stocks from the dataset, and re-order the dataset into ascending time
dataset = np.flip(batch_load(get_csv_list(train_config_dict["data_folder"]), pd_df=True, as_delta=train_config_dict["dataset_as_delta"]).to_numpy(), axis=0)[:, :train_config_dict["num_of_stocks"]]

total_dataset_length = dataset.shape[0]
train_index = int(train_config_dict["train_split"] * total_dataset_length)
val_index = train_index + int(train_config_dict["val_split"] * total_dataset_length)

train_set = torch.tensor(assemble_dataset(dataset[:train_index], look_back_window=train_config_dict["look_back_time_window"]))
val_set = torch.tensor(assemble_dataset(dataset[train_index:val_index], look_back_window=train_config_dict["look_back_time_window"]))
test_set = torch.tensor(assemble_dataset(dataset[val_index:], look_back_window=train_config_dict["look_back_time_window"]))

# print(train_set.shape)

loss_fn = torch.nn.MSELoss()

model = MarketTransformerEncoder(
    num_stocks=train_config_dict["num_of_stocks"],
    model_dim=train_config_dict["model_dim"],
    feedforward_dim=train_config_dict["feed_forward_dim"],
    num_head=train_config_dict["num_heads"],
    num_layer=train_config_dict["encoder_layers"]
)
summary(model)


optimiser = torch.optim.AdamW(model.parameters(), lr=train_config_dict["learning_rate"])
scheduler = torch.optim.lr_scheduler.StepLR(
    optimiser, 
    step_size=train_config_dict["optimiser_step_size"], 
    gamma=train_config_dict["optimiser_gamma"]
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

    output = model(train_set[:, :train_config_dict["look_back_time_window"]-1, :])

    loss = loss_fn(train_set[:, -1, :], output[:, 0, :])

    loss.backward()
    optimiser.step()
    scheduler.step()

    ave_train_loss = loss.item()/train_set.shape[0]
    train_losses.append(ave_train_loss)
    print(f"Epoch {i + 1} / {train_config_dict['num_epochs']} train loss: {ave_train_loss}")

    model.eval()
    
    output = model(val_set[:, :train_config_dict["look_back_time_window"]-1, :])
    loss = loss_fn(val_set[:, -1, :], output[:, 0, :])

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

output = model(test_set[:, :train_config_dict["look_back_time_window"]-1, :])
loss = loss_fn(test_set[:, -1, :], output[:, 0, :])

ave_test_loss = loss.item()/test_set.shape[0]
test_losses.append(ave_test_loss)
print(f"average test loss: {ave_test_loss}")



# visualise predictions
predictions=[]
test_set = torch.tensor(dataset[val_index:].copy())

input = np.concatenate([[np.zeros(train_config_dict["num_of_stocks"])], dataset[val_index-train_config_dict["look_back_time_window"]+1: val_index]], axis=0)
# print(input)
input = input[:-2]
# print(input)

for i in range(test_set.shape[0]):
    output = model(torch.tensor(input, dtype=torch.float32, device=device))
    predictions.append(output[0, :].detach().cpu())
    input = input[:-2]
    input = np.concatenate([[np.zeros(train_config_dict["num_of_stocks"])], input], axis=0)

predictions = np.array(predictions)

for i in range(train_config_dict["num_of_stocks"]):
    plt.plot(predictions[:, i], label="predictions")
    plt.plot(test_set[:, i], label="truth")
    plt.legend()
    plt.savefig(f"pred_vs_actual for stock {i}")
    plt.cla()
    plt.close()
