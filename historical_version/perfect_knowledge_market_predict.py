from data_utils import get_csv_list, batch_load, perfect_prediction, calculate_performance, plot_performance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


market_df = batch_load(
    get_csv_list("./data"), 
    pd_df=True, 
    # normalise=True,
)
stock_names = market_df.columns
sp500_idx = stock_names.get_loc("sp500")
print(stock_names)
market_database = market_df.to_numpy().transpose()
market_database = np.flip(market_database, axis=-1)
# print(market_database)
norm_market_database = np.divide(market_database, np.outer(market_database[:, 0], np.ones(market_database.shape[1])))
# print(norm_market_database)



optimal_chance = 0.33

perfect_pred = perfect_prediction(market_database, optimal_chance=optimal_chance)
perfect_performance = calculate_performance(market_database, perfect_pred, transaction_cost=1)

plot_performance(
    norm_market_database, 
    {f"perfect performance with chance {optimal_chance}" : perfect_performance},
    sp500_idx  
)