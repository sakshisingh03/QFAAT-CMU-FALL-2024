import pytest
import pandas as pd
from unittest.mock import patch
from utils.utils import (
    GetYahooFinanceData,
    add_cagr_to_stats,
    combine_execution_results,
    combine_optimization_results,
    combine_train_test_results,
    fetch_data_for_instruments
    )


# --- Test for GetYahooFinanceData class ---
@pytest.fixture
def mock_yahoo_finance_data():
    """
    This function sets up a mock object of GetYahooFinanceData class
    to be used in the test functions.
    """
    return GetYahooFinanceData()


# Test for fetching data from Yahoo Finance
@patch('yfinance.download')
def test_fetch_data(mock_download, mock_yahoo_finance_data):
    """
    This test checks if the data is correctly fetched from Yahoo Finance
    for a list of tickers in the specified date range and interval.
    """
    # Mock data returned by yfinance.download
    mock_download.return_value = pd.DataFrame({
        'Open': [1, 2],
        'High': [2, 3],
        'Low': [1, 1.5],
        'Close': [1.5, 2.5],
        'Volume': [100, 200],
        'Adj Close': [1.4, 2.4]
    })

    start_date = '2023-01-01'
    end_date = '2023-12-31'
    currency_list = ['EURUSD=X']

    data = mock_yahoo_finance_data.fetch_data(start_date, end_date,
                                              currency_list, interval='1h')

    # Test if the method returns data for each ticker
    assert 'EURUSD=X' in data
    # Check if the correct columns are returned (Open, High, Low, Close)
    assert data['EURUSD=X'].shape == (2, 4)


# Test for error handling in fetching data
@patch('yfinance.download')
def test_fetch_data_error(mock_download, mock_yahoo_finance_data):
    """
    This test checks how the function handles errors while fetching data.
    It ensures the function returns an empty result if there's an error.
    """
    # Simulate an error in the data fetching
    mock_download.side_effect = Exception("Error fetching data")

    start_date = '2023-01-01'
    end_date = '2023-12-31'
    currency_list = ['EURUSD=X']

    data = mock_yahoo_finance_data.fetch_data(start_date, end_date,
                                              currency_list, interval='1h')

    # Ensure the data is empty and error is handled
    assert 'EURUSD=X' not in data


# --- Test for add_cagr_to_stats function ---
def test_add_cagr_to_stats():
    """
    This test checks if the add_cagr_to_stats function correctly adds
    the Compound Annual Growth Rate (CAGR) to the stats dictionary.
    """
    # Create mock stats data
    stats = {
        'Equity Final [$]': 110000,
        'Equity Peak [$]': 100000,
        'Duration': pd.Timedelta(days=100)
    }
    initial_cash = 100000

    # Use the utility function to calculate CAGR
    updated_stats = add_cagr_to_stats(stats, initial_cash)

    # Check if CAGR is correctly added
    assert 'CAGR (%)' in updated_stats
    assert updated_stats['CAGR (%)'] > 0


# --- Test for combine_execution_results function ---
@patch('os.listdir')
@patch('pandas.read_csv')
def test_combine_execution_results(mock_read_csv, mock_listdir):
    """
    This test checks if the function combines execution results from multiple
    CSV files into a single combined result.
    """
    mock_listdir.return_value = ['execution_results_AAPL.csv',
                                 'execution_results_GOOG.csv']
    mock_read_csv.side_effect = [
        pd.DataFrame({'equity': [1000], 'duration': ['1d']}),
        pd.DataFrame({'equity': [1500], 'duration': ['1d']})
    ]

    instrument_type = 'stocks'
    tickers = ['AAPL', 'GOOG']
    results_path = '/path/to/results'

    # Run the function to combine results
    combine_execution_results(instrument_type, tickers, results_path)

    # Check if the combined file is created
    mock_read_csv.assert_called()
    assert mock_listdir.called


# --- Test for combine_optimization_results function ---
@patch('os.listdir')
@patch('pandas.read_csv')
def test_combine_optimization_results(mock_read_csv, mock_listdir):
    """This test checks if the function combines optimization results from
    multiple CSV files into a single combined result."""
    mock_listdir.return_value = ['optimization_results_AAPL.csv',
                                 'optimization_results_GOOG.csv']
    mock_read_csv.side_effect = [
        pd.DataFrame({'param': ['param1'], 'value': [100]}),
        pd.DataFrame({'param': ['param2'], 'value': [200]})
    ]

    instrument_type = 'stocks'
    tickers = ['AAPL', 'GOOG']
    results_path = '/path/to/results'

    # Run the function to combine optimization results
    combine_optimization_results(instrument_type, tickers, results_path)

    # Check if the combined file is created
    mock_read_csv.assert_called()
    assert mock_listdir.called


# --- Test for combine_train_test_results function ---
@patch('os.listdir')
@patch('pandas.read_csv')
def test_combine_train_test_results(mock_read_csv, mock_listdir):
    """
    This test checks if the function combines train-test results from multiple
    CSV files into a single combined result.
    """
    mock_listdir.return_value = ['train_test_results_AAPL.csv',
                                 'train_test_results_GOOG.csv']
    mock_read_csv.side_effect = [
        pd.DataFrame({'accuracy': [0.95], 'loss': [0.1]}),
        pd.DataFrame({'accuracy': [0.97], 'loss': [0.08]})
    ]

    instrument_type = 'stocks'
    tickers = ['AAPL', 'GOOG']
    results_path = '/path/to/results'

    # Run the function to combine train-test results
    combine_train_test_results(instrument_type, tickers, results_path)

    # Check if the combined file is created
    mock_read_csv.assert_called()
    assert mock_listdir.called


# --- Test for fetch_data_for_instruments function ---
@patch('utils.utils.GetYahooFinanceData.fetch_data')
def test_fetch_data_for_instruments(mock_fetch_data):
    """This test checks if the function correctly fetches data for multiple
    instruments. It ensures that the correct data is returned for each
    instrument."""
    # Mock fetch data to return a sample data dict
    mock_fetch_data.return_value = {
        'EURUSD=X': pd.DataFrame({'Open': [1.1, 1.2], 'Close': [1.2, 1.3]})
    }

    start_date = '2023-01-01'
    end_date = '2023-12-31'
    tickers = ['EURUSD=X']
    interval = '1h'

    data = fetch_data_for_instruments(start_date, end_date, tickers, interval)

    # Assert if data exists for the ticker
    assert 'EURUSD=X' in data
    assert data['EURUSD=X'].shape == (2, 2)  # Open and Close columns


# --- Test for error handling in fetch_data_for_instruments ---
@patch('utils.utils.GetYahooFinanceData.fetch_data')
def test_fetch_data_for_instruments_error(mock_fetch_data):
    """
    This test checks how the function handles errors while fetching data.
    It ensures the function returns an empty result if there's an error.
    """
    # Simulate an error
    mock_fetch_data.side_effect = Exception("Error fetching data")

    start_date = '2023-01-01'
    end_date = '2023-12-31'
    tickers = ['EURUSD=X']
    interval = '1h'

    data = fetch_data_for_instruments(start_date, end_date, tickers, interval)

    # Ensure empty data is returned on error
    assert data == {}
