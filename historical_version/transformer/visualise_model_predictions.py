import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset_utils import get_csv_list, batch_load

train_config_dict = {
    "data_folder": "/Users/tangm/Downloads/github/market/data",
    "train_split": 0.8,
    "val_split": 0.1,
    "look_back_time_window": 65,
    "num_of_stocks": 10,
}


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = torch.load("model_save.pt", weights_only=False).to(device)


predictions = []

# get the first 8 stocks from the dataset, and re-order the dataset into ascending time
dataset = np.flip(batch_load(get_csv_list(train_config_dict["data_folder"]), pd_df=True, as_delta=True).to_numpy(), axis=0)[:, :train_config_dict["num_of_stocks"]]

total_dataset_length = dataset.shape[0]
# print(dataset.shape)
train_index = int(train_config_dict["train_split"] * total_dataset_length)
val_index = train_index + int(train_config_dict["val_split"] * total_dataset_length)

# print(val_index)
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