import torch
from models import MLP_predictor, stock_train_test_split, train, model_predictions
from data_utils import get_csv_list, batch_load, calculate_performance, plot_performance
import numpy as np


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
folder = "./data"
time_window = 10
batch_size = 32

market_df = batch_load(
    get_csv_list(folder),
    pd_df=True,
)

stocks = market_df.columns
sp500_idx = stocks.get_loc("sp500")
market_database = np.flip(market_df.to_numpy().transpose(), axis=-1)

train_set, test_set = stock_train_test_split(market_database)


num_stocks = train_set.shape[0]
num_models = 5
models = []

for i in range(num_models):
    model = MLP_predictor(num_stocks, num_hidden_layers=3, hidden_dim=128)
    train(
        model, 
        train_set, 
        test_set, 
        device, 
        # print_log=True
    )
    models.append(model)


model_performance = {}

test_market_data = market_database[:, -test_set.shape[1]:]

norm_test_market_data = test_market_data / np.outer(test_market_data[:, 0], np.ones(test_market_data.shape[1]))
print(norm_test_market_data)

for i in range(len(models)):
    model_pred, truth = model_predictions(models[i], test_set, device=device)
    print(model_pred)
    model_performance[f"model_{i}"] = calculate_performance(norm_test_market_data, model_pred, transaction_cost=1)

print(truth)


plot_performance(
    norm_test_market_data, 
    model_performance,
    sp500_idx  
)