import pytest
import pandas as pd
import numpy as np
import pytz
import os
from datetime import datetime
from unittest.mock import patch
from utils.utils import (
    GetYahooFinanceData,
    add_cagr_to_stats,
    combine_execution_results,
    combine_optimization_results,
    combine_train_test_results,
    fetch_data_for_instruments,
    calculate_walk_forward_metric,
    define_walk_forward_iterations,
    localize_data,
    combine_wfo_results
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


def test_calculate_walk_forward_metric():
    """
    Tests the calculate_walk_forward_metric function.
    """
    # Test with valid in-sample metric
    assert calculate_walk_forward_metric(10, 5) == 2.0

    # Test with zero in-sample metric
    assert np.isnan(calculate_walk_forward_metric(10, 0))

    # Test with negative values
    assert calculate_walk_forward_metric(-10, -5) == 2.0

    # Test with zero out-of-sample metric
    assert calculate_walk_forward_metric(0, 5) == 0.0


def test_define_walk_forward_iterations():
    """
    Tests the define_walk_forward_iterations function.
    """
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    in_sample_duration = pd.DateOffset(days=30)
    out_of_sample_duration = pd.DateOffset(days=15)
    num_iterations = 3

    iterations = define_walk_forward_iterations(
        start_date, end_date, in_sample_duration,
        out_of_sample_duration, num_iterations)

    # Check the number of iterations
    assert len(iterations) == num_iterations

    # Check the structure of the first iteration
    first_iteration = iterations[0]
    assert 'in_sample' in first_iteration and 'out_of_sample' \
        in first_iteration

    # Check that the dates are localized to UTC
    assert first_iteration['in_sample'][0].tzinfo == pytz.UTC
    assert first_iteration['out_of_sample'][0].tzinfo == pytz.UTC


def test_localize_data():
    """
    Tests the localize_data function.
    """
    # Create a DataFrame with a naive datetime index
    naive_index = pd.date_range(start='2024-01-01', periods=5, freq='D')
    data = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=naive_index)

    # Localize the data
    localized_data = localize_data(data)

    # Check that the index is localized to UTC
    assert localized_data.index.tzinfo == pytz.UTC

    # Test with already localized index
    tz_aware_index = pd.date_range(start='2024-01-01',
                                   periods=5, freq='D', tz='UTC')
    data = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, index=tz_aware_index)

    localized_data = localize_data(data)
    assert localized_data.index.tzinfo == pytz.UTC


# Mock data and setup
@pytest.fixture
def mock_results_path(tmp_path):
    """Creates a temporary directory with mock WFO result files."""
    results_path = tmp_path / "results"
    results_path.mkdir()

    # Create mock WFO result files
    tickers = ["AAPL", "MSFT"]
    for ticker in tickers:
        file_path = results_path / f"wfo_results_{ticker}.csv"
        df = pd.DataFrame({
            "start_date": ["2024-01-01 00:00:00+00:00"],
            "end_date": ["2024-01-31 00:00:00+00:00"],
            "ret_strat_ann": [0.1],
            "volatility_strat_ann": [0.2],
        })
        df.to_csv(file_path, index=False)

    return str(results_path)


def test_combine_wfo_results(mock_results_path):
    """Tests the combine_wfo_results function."""

    instrument_type = "stocks"
    tickers = ["AAPL", "MSFT"]
    results_path = mock_results_path

    # Call the function
    combine_wfo_results(instrument_type, tickers, results_path)

    # Check the combined file exists
    combined_file_path = os.path.\
        join(results_path, f"combined_wfo_results_{instrument_type}.csv")
    assert os.path.exists(combined_file_path)

    # Validate the content of the combined file
    combined_df = pd.read_csv(combined_file_path)
    assert not combined_df.empty, "Combined results file is empty."

    # Check the columns
    expected_columns = ["start_date", "end_date", "ret_ann", "vol_ann",
                        "ticker"]
    assert all(col in combined_df.columns for col in expected_columns)

    # Check data integrity
    assert len(combined_df[combined_df["ticker"] == "AAPL"]) == 1
    assert len(combined_df[combined_df["ticker"] == "MSFT"]) == 1
