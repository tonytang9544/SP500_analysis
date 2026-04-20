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

def assemble_dataset(np_dataset, look_back_window=64):
    '''
    Turn a numpy dataset into a stack of look_back_window of stock prices. The first entry is a zero vector corresponding to the [CLS] token.

    input:
        Numpy array of dimension (number_of_time_points, number_of_stocks)
    
    output:
        Numpy array of dimension (number_of_time_points-look_back_window, look_back_window+1, number_of_stocks)
    '''

    dataset_len, num_stocks = np_dataset.shape

    if dataset_len <= look_back_window:
        raise IndexError(f"number of time points {dataset_len} is less than the look_back_window {look_back_window}")
    
    assembled_list = []
    # print(np.zeros([1, int(num_stocks)]))
    # print(np.concatenate(([[0,0,0,0]], [[1,2,3,4], [5,6,7,8]]), axis=0))

    for i in range(dataset_len - look_back_window):
        assembled_list.append(np.concatenate([[np.zeros(int(num_stocks))], np_dataset[i:i+look_back_window, :]], axis=0))
    
    assembled_dataset = np.array(assembled_list)

    return assembled_dataset