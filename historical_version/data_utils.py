import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def batch_load(list_of_csvs, pd_df=False, normalise=False):
    '''
    load all csv files in the list_of_csv into dictionary of numpy arrays

    input:
        list_of_csv: 
            list of filepath for csv files
        normalise:
            if True, normalise each stock based on the very first price of that stock.
        pd_df: 
            if False, return dictionary of numpy array for each stock prices
            if True, clip to fit the shortest numpy array and return everything as pandas dataframe
        
        
    return:
        see above for pd_df
    '''
    stocks = {}
    if len(list_of_csvs) > 0:
        for csv in list_of_csvs:
            stock = pd.read_csv(csv, thousands=',')
            cols = stock.columns
            cols = cols.drop("Date")
            cols = cols.drop("Volume")
            name = csv.split("/")[-1].split(".csv")[0]
            # data = stock["Adj Close"].astype(float).to_numpy()
            data = stock["Open"].astype(float).to_numpy()
            if normalise:
                data = data / data[-1]
            stocks[name] = data
    
    if pd_df:
        shortest_array = min([len(i) for i in stocks.values()])
        for name in stocks.keys():            
            data = stocks[name][:shortest_array]
            if normalise:
                data = data / data[-1]
            stocks[name] = data
        stocks = pd.DataFrame(stocks)
    return stocks


def get_csv_list(folder):
    '''
    input:
        folder: folder containing the csv files.

    return
        list of file paths for csv files
    '''
    csv_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                csv_list.append(os.path.join(root, file))
    return csv_list


def daily_changes(market_data):
    '''
    input:
        market_data: numpy array of market data for each stock, assuming first time point at array position 0, with dimension (number_of_stocks, time_points)
    
    return:
        daily price deltas as a numpy array, with dimension (number_of_stocks, time_points-1)
    '''
    return market_data[:, 1:] / market_data[:, :-1]


def normalise_to_day_one(market_data):
    '''
    input:
        market_data: numpy array of market data for each stock, assuming first time point at array position 0, with dimension (number_of_stocks, time_points)
    
    return:
        all stock prices normalised so that day 1 = 1, with dimension (number_of_stocks, time_points)
    '''

    return np.divide(market_data, np.outer(market_data[:, 0], np.ones(market_data.shape[1])))


def calculate_performance(market_data, prediction, transaction_cost=0.5):
    '''
    input:
        market_data: numpy array of market data for each stock, assuming first time point at array position 0, with dimension (number_of_stocks, time_points)
        prediction: stocks selected by the model at each day, with dimension (time_points, )
        transaction_cost: percentage cost associated with each buying (or switching stocks) event
    
    return:
        prediction_performance: performance of the stocks based on prediction
    '''
    delta = daily_changes(market_data)

    prediction_performance = np.ones(market_data.shape[1])

    for i in range(market_data.shape[1] -1):
        prediction_performance[i+1] = delta[prediction[i], i] * prediction_performance[i]
        if i>=1 and prediction[i] != prediction[i-1]:
            prediction_performance[i+1] *= (1 - transaction_cost/100)

    return prediction_performance


def perfect_prediction(market_data, optimal_chance=1):
    '''
    input:
        market_data: numpy array of market data for each stock, assuming first time point at array position 0, with dimension (number_of_stocks, time_points)
        optimal_chance: chance that the optimal stock is picked. Otherwise select all stocks at random.

    return:
        predictions assuming future stock movements are known beforehand
    '''
    delta = daily_changes(market_data)
    num_of_stocks = market_data.shape[0]
    perfect_pred = np.argmax(delta, axis=0)
    for i in range(perfect_pred.shape[0]):
        if np.random.random() > optimal_chance:
            perfect_pred[i] = np.random.choice(np.arange(num_of_stocks))
    return perfect_pred


def plot_performance(norm_market_data, performance_dictionary, sp500_idx, y_scale="log"):
    print(f"average return = {np.mean(norm_market_data[:, -1])}")
    print(f"sp500 return = {norm_market_data[sp500_idx, -1]}")
    print(f"best stock performance = {np.max(norm_market_data[:, -1])}")

    for i in range(norm_market_data.shape[0]):
        if i != sp500_idx:
            plt.plot(
                norm_market_data[i,:],
                color="#c0c0c0"
            )
        else:
            plt.plot(
                norm_market_data[i,:],
                label = "sp500",
                color="#000000"
            )
    
    for k, v in performance_dictionary.items():
        plt.plot(v, label=k)

    plt.title("stock_movement")
    plt.yscale(y_scale)
    plt.legend()
    plt.show()

