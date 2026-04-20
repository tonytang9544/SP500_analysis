import torch
from models import MLP_predictor, stock_train_test_split, load_models, train, model_predictions, StockDataset
from data_utils import get_csv_list, batch_load, perfect_prediction, calculate_performance, plot_performance
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
folder = "./data"
time_window = 10
batch_size = 32

market_database = batch_load(
    get_csv_list(folder),
    pd_df=True,
)

train_set, test_set = stock_train_test_split(market_database)

train_data = StockDataset(
    train_set, 
    time_window=time_window
)

test_data = StockDataset(
    test_set,
    time_window=time_window
)

train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size
)

