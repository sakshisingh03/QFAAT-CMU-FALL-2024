import pandas as pd
import requests
import yfinance as yf
import os
from datetime import timedelta, datetime
import pytz
import numpy as np


class GetYahooFinanceData:
    """
    A class to fetch data from Yahoo Finance for both currency and equities.
    """

    def __init__(self):
        """Initialize the class."""
        print("Object initialized")

    def fetch_data(self, start_date, end_date, currency_list, interval='1h'):
        """
        Fetches the data for a list of tickers from Yahoo Finance.

        Parameters:
        start_date (str): The start date for data fetch (e.g., '2023-01-01').
        end_date (str): The end date for data fetch (e.g., '2023-12-31').
        currency_list (list): List of ticker symbols to fetch data for.
        interval (str): Interval for data ('1h' by default).

        Returns:
        dict: A dictionary of dataframes, where the keys are ticker symbols.
        """
        data_dict = {}
        for ticker in currency_list:
            try:
                # Fetching data for each ticker
                data = yf.download(ticker, start=start_date, end=end_date,
                                   interval=interval)

                # Drop unnecessary columns if they exist
                drop_columns = ['Volume', 'Adj Close']
                data = data.drop(columns=[col for col in drop_columns if col
                                          in data.columns])

                data_dict[ticker] = data
                print(f"Data pull complete for: {ticker}")

            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")

        return data_dict

    def get_alphavantage_1hr_data(self, symbol, api_key,
                                  output_size="compact"):
        """
        Fetches 1-hour interval intraday stock data from Alpha Vantage.

        Parameters:
        symbol (str): The stock symbol to fetch data for (e.g., 'AAPL').
        api_key (str): Your Alpha Vantage API key.
        output_size (str): The amount of data to retrieve ("compact" or "full")

        Returns:
        DataFrame: Pandas DataFrame containing the 1-hour interval stocks data.
        """
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": "60min",  # 1-hour interval
            "apikey": api_key,
            "outputsize": output_size,
            "datatype": "json"
        }

        try:
            # Make the API request
            response = requests.get(url, params=params)
            data = response.json()

            time_series_key = "Time Series (60min)"

            if time_series_key in data:
                # Convert the data to a DataFrame
                df = pd.DataFrame.from_dict(data[time_series_key],
                                            orient="index")
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
                print(f"Error fetching data for {symbol}:"
                      f"{data.get('Error Message', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None


def add_cagr_to_stats(stats, cash):
    """
    Adds Compound Annual Growth Rate (CAGR) to the stats dataframe.

    Parameters:
    stats (DataFrame): The stats dataframe containing equity final value.
    cash (float): The initial cash value.

    Returns:
    DataFrame: The updated stats dataframe with CAGR included.
    """
    try:
        days = stats['Duration'].days
        years = days / 365.25
        if years > 0:
            cagr = (stats["Equity Final [$]"] / cash) ** (1 / years) - 1
            stats['CAGR (%)'] = cagr * 100
        else:
            stats['CAGR (%)'] = 0
        return stats
    except Exception as e:
        print(f"Error calculating CAGR: {e}")
        return stats


def combine_execution_results(instrument_type, tickers, results_path):
    """
    Combines execution result files for the specified tickers into one CSV.

    Parameters:
    instrument_type (str): The type of instrument (e.g., 'stocks', 'currency').
    tickers (list): List of tickers to include.
    results_path (str): Path where result files are located.
    """
    combined_results = pd.DataFrame()
    try:
        for filename in os.listdir(results_path):
            file_details = filename.split('_')
            file_details = [x.replace('.csv', '') for x in file_details]

            if file_details[0] == 'execution':
                ticker = file_details[2]
                if ticker in tickers:
                    df_results = pd.read_csv(
                        os.path.join(results_path,
                                     f'execution_results_{ticker}.csv'))
                    df_results['ticker'] = ticker
                    combined_results = pd.concat([combined_results,
                                                  df_results])

        combined_results.to_csv(
            os.path.join(results_path, f'combined_execution_results_'
                         f'{instrument_type}.csv'))
    except Exception as e:
        print(f"Error combining execution results: {e}")


def combine_optimization_results(instrument_type, tickers, results_path):
    """
    Combines optimization result files for the specified tickers into one CSV.

    Parameters:
    instrument_type (str): The type of instrument (e.g., 'stocks', 'currency').
    tickers (list): List of tickers to include.
    results_path (str): Path where result files are located.
    """
    combined_results = pd.DataFrame()
    try:
        for filename in os.listdir(results_path):
            file_details = filename.split('_')
            file_details = [x.replace('.csv', '') for x in file_details]

            if file_details[0] == 'optimization':
                ticker = file_details[2]
                if ticker in tickers:
                    df_results = pd.read_csv(
                        os.path.join(results_path,
                                     f'optimization_results_{ticker}.csv'))
                    df_results['ticker'] = ticker
                    combined_results = pd.concat([combined_results,
                                                  df_results])

        combined_results.\
            to_csv(os.path.join(results_path,
                                f'combined_optimization_results_'
                                f'{instrument_type}.csv'))
    except Exception as e:
        print(f"Error combining optimization results: {e}")


def combine_wfo_results(instrument_type, tickers, results_path):
    """
    Combines WFO result files for the specified tickers into one CSV.

    Parameters:
    instrument_type (str): The type of instrument (e.g., 'stocks', 'currency').
    tickers (list): List of tickers to include.
    results_path (str): Path where result files are located.
    """
    combined_results = pd.DataFrame()
    try:
        for filename in os.listdir(results_path):
            file_details = filename.split('_')
            file_details = [x.replace('.csv', '') for x in file_details]

            if file_details[0] == 'wfo':
                ticker = file_details[2]
                if ticker in tickers:
                    df_results = pd.read_csv(
                        os.path.join(results_path,
                                     f'wfo_results_{ticker}.csv'))
                    df_results['ticker'] = ticker
                    combined_results = pd.concat([combined_results,
                                                  df_results])

        combined_results['start_date'] = combined_results['start_date'].\
            apply(lambda x: datetime.strptime(x[:-6],
                                              '%Y-%m-%d %H:%M:%S').date())
        combined_results['end_date'] = combined_results['end_date'].\
            apply(lambda x: datetime.strptime(x[:-6],
                                              '%Y-%m-%d %H:%M:%S').date())
        combined_results = combined_results.\
            rename(columns={'ret_strat_ann': 'ret_ann',
                            'volatility_strat_ann': 'vol_ann'})

        combined_results.\
            to_csv(os.path.join(results_path,
                                f'combined_wfo_results_'
                                f'{instrument_type}.csv'))
    except Exception as e:
        print(f"Error combining WFO results: {e}")


def combine_train_test_results(instrument_type, tickers, results_path):
    """
    Combines train test result files for the specified tickers into one CSV.

    Parameters:
    instrument_type (str): The type of instrument (e.g., 'stocks', 'currency').
    tickers (list): List of tickers to include.
    results_path (str): Path where result files are located.
    """
    combined_results = pd.DataFrame()
    try:
        for filename in os.listdir(results_path):
            file_details = filename.split('_')
            file_details = [x.replace('.csv', '') for x in file_details]

            if file_details[0] == 'train' and file_details[1] == 'test':
                ticker = file_details[3]
                if ticker in tickers:
                    df_results = pd.read_csv(
                        os.path.join(results_path,
                                     f'train_test_results_{ticker}.csv'))
                    df_results['ticker'] = ticker
                    combined_results = pd.concat([combined_results,
                                                  df_results])

        combined_results.to_csv(
            os.path.join(results_path,
                         f'combined_train_test_results_{instrument_type}.csv'))
    except Exception as e:
        print(f"Error combining train-test results: {e}")


def fetch_data_for_instruments(start_date, end_date, tickers,
                               interval):
    """
    Fetches data for a list of instruments and checks if data exists for each.

    Parameters:
    data_obj (object): The data object used to fetch the data.
    start_date (str): The start date for fetching data.
    end_date (str): The end date for fetching data.
    tickers (list): The list of tickers.
    interval (str): The interval for the data ('1h' for currencies or '1d'
    for equities).

    Returns:
    dict: A dictionary with tickers as keys and data as values.
    """
    try:
        data_obj = GetYahooFinanceData()
        data = data_obj.fetch_data(start_date, end_date, tickers, interval)

        # Check if data exists for each ticker
        for ticker in tickers:
            if ticker not in data or data[ticker].empty:
                raise ValueError(f"No data found for ticker: {ticker}")

        return data
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return {}
    except Exception as e:
        print(f"Error fetching data for instruments: {e}")
        return {}


def calculate_walk_forward_metric(oos_metric, ins_metric):
    """
    Calculates a Walk Forward metric (e.g., WFE)

    Args:
        oos_metric: Out-of-sample metric value (float)
        ins_metric: In-sample metric value (float)

    Returns:
        The calculated Walk Forward metric (float)
    """
    if ins_metric != 0:
        return (oos_metric / ins_metric)
    else:
        return np.nan  # Handle division by zero


def define_walk_forward_iterations(start_date, end_date, in_sample_duration,
                                   out_of_sample_duration, num_iterations):
    """
    Defines a list of dictionaries representing walk-forward iterations

    Args:
        start_date: Overall start date (date object)
        end_date: Overall end date (datetime object)
        in_sample_duration: Length of the in-sample period (DateOffset object)
        out_of_sample_duration: Length of the out-of-sample period (DateOffset
        object)
        num_iterations: Number of walk-forward iterations (int)

    Returns:
        A list of dictionaries where each dictionary represents an iteration
        with in-sample and out-of-sample start and end dates
    """
    iterations = []
    for i in range(num_iterations):
        in_sample_start = start_date + pd.DateOffset(days=0)
        in_sample_end = in_sample_start + \
            (i+1)*in_sample_duration - timedelta(hours=1)
        out_of_sample_start = in_sample_end + timedelta(hours=1)
        out_of_sample_end = out_of_sample_start + \
            out_of_sample_duration - timedelta(hours=1)

        # Make the datetime objects timezone-aware in UTC
        in_sample_start = pytz.utc.localize(in_sample_start)
        in_sample_end = pytz.utc.localize(in_sample_end)
        out_of_sample_start = pytz.utc.localize(out_of_sample_start)
        out_of_sample_end = pytz.utc.localize(out_of_sample_end)

        iterations.append({
            'in_sample': [in_sample_start, in_sample_end],
            'out_of_sample': [out_of_sample_start, out_of_sample_end]
        })
    return iterations


def localize_data(data):
    """
    Localizes the index of a DataFrame to UTC

    Args:
        data: DataFrame with potentially non-localized index

    Returns:
        A DataFrame with its index localized to UTC
    """
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert('UTC')
    return data
