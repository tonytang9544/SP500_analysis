import pandas as pd
import numpy as np
import os

def batch_load(list_of_csvs, pd_df=False, as_delta=False, print_debug=False):
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
            if as_delta:
                data = data[2:] / data[:-2]
                if print_debug:
                    print(data)
            stocks[name] = data
    
    if pd_df:
        shortest_array = min([len(i) for i in stocks.values()])
        for name in stocks.keys():            
            data = stocks[name][:shortest_array]
            if as_delta:
                data = data[2:] / data[:-2]
                if print_debug:
                    print(data)
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
