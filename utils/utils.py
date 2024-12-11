import pandas as pd
import requests
import yfinance as yf
import os
from datetime import datetime

class getYahooFinanceData:
    '''class to help fetch data from Yahoo Finance - both currency and equities'''
    def __init__(self):
        print("object initialized")

    def fetch_data(self, start_date, end_date, currency_list, interval='1h'):
        # Loop through each ticker and download 1-hour data
        dict_ = {}
        for ticker in currency_list:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            
            # Drop certain columns if they exist
            drop_columns = ['Volume', 'Adj Close']
            for col in drop_columns:
                if col in data.columns:
                    data = data.drop(columns=[col])
        
            dict_[ticker] = data

            print("data pull complete for: ", ticker)
    
        return dict_
    

def get_alphavantage_1hr_data(self, symbol, api_key, output_size="compact"):
    """
    Fetches 1-hour interval intraday stock data from Alpha Vantage for the specified symbol.

    Parameters:
    symbol (str): The stock symbol to fetch data for (e.g., 'AAPL').
    api_key (str): Your Alpha Vantage API key.
    output_size (str): The amount of data to retrieve. "compact" returns the last 100 points,
                    "full" returns all available data (default is "compact").

    Returns:
    DataFrame: A Pandas DataFrame containing the 1-hour interval stock data.
    """
    # Base URL for Alpha Vantage API
    url = "https://www.alphavantage.co/query"

    # Define the API parameters
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": "60min",  # 1-hour interval
        "apikey": api_key,
        "outputsize": output_size,
        "datatype": "json"
    }

    # Make the API request
    response = requests.get(url, params=params)
    data = response.json()

    time_series_key = "Time Series (60min)"

    if time_series_key in data:
        # Convert the data to a DataFrame
        df = pd.DataFrame.from_dict(data[time_series_key], orient="index")
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        df.index = pd.to_datetime(df.index)
        df = df.astype(float)
        return df
    else:
        print("Error fetching data:", data.get("Error Message", "Unknown error"))
        return None
        
    
def add_cagr_to_stats(stats, cash):
    days = stats['Duration'].days
    years = days / 365.25
    if years > 0:
        cagr = (stats["Equity Final [$]"] / cash) ** (1 / years) - 1
        stats['CAGR (%)'] = cagr * 100
    else:
        stats['CAGR (%)'] = 0
    return stats


def combine_execution_results(instrument_type, tickers, results_path):
    combined_results = pd.DataFrame()
    for filename in os.listdir(results_path):
        file_details = filename.split('_')
        file_details = [x.replace('.csv', '') for x in file_details]
        print(file_details)
    
        if file_details[0] == 'execution':
            ticker = file_details[2]
            if ticker in tickers:
                df_results = pd.read_csv(results_path + 'execution_results_' + ticker + '.csv')
                df_results['ticker'] = ticker
                combined_results = pd.concat([combined_results, df_results])
    
    combined_results.to_csv(results_path + 'combined_execution_results_' + instrument_type + '.csv')
    
    
def combine_optimization_results(instrument_type, tickers, results_path):
    combined_results = pd.DataFrame()
    for filename in os.listdir(results_path):
        file_details = filename.split('_')
        file_details = [x.replace('.csv', '') for x in file_details]
        print(file_details)
    
        if file_details[0] == 'optimization':
            ticker = file_details[2]
            if ticker in tickers:
                df_results = pd.read_csv(results_path + 'optimization_results_' + ticker + '.csv')
                df_results['ticker'] = ticker
                combined_results = pd.concat([combined_results, df_results])
    
    combined_results.to_csv(results_path + 'combined_optimization_results_' + instrument_type + '.csv')
    

def combine_train_test_results(instrument_type, tickers, results_path):
    combined_results = pd.DataFrame()
    for filename in os.listdir(results_path):
        file_details = filename.split('_')
        file_details = [x.replace('.csv', '') for x in file_details]
        print(file_details)
    
        if file_details[0] == 'optimization':
            ticker = file_details[3]
            if ticker in tickers:
                df_results = pd.read_csv(results_path + 'train_test_results_' + ticker + '.csv')
                df_results['ticker'] = ticker
                combined_results = pd.concat([combined_results, df_results])
    
    combined_results.to_csv(results_path + 'combined_train_test_results_' + instrument_type + '.csv')