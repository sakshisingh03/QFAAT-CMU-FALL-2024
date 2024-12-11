import pytest
import pandas as pd
from lib_macd.macd_modules import (
    make_macd_indicators,
    MACDStrategy,
    execute_macd_strategy,
    optimize_macd_strategy
)
from backtesting import Backtest


def create_sample_data():
    """
    Creates sample stock market data for testing.

    Returns:
        pd.DataFrame: DataFrame with 'Open', 'High', 'Low', 'Close' prices
        and dates.
    """
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    }

    # Generate a date range (10 days from '2024-01-01')
    dates = pd.date_range(start='2024-01-01', periods=len(data['Open']),
                          freq='D')

    # Create a DataFrame with the dates as index
    df = pd.DataFrame(data, index=dates)

    return df


def test_make_macd_indicators():
    """
    Tests the function that makes MACD indicators for the data.

    Checks if MACD, Signal, and Hist columns are added correctly.
    """
    df = create_sample_data()

    # Calculate MACD indicators
    result = make_macd_indicators(df)

    # Check if 'MACD', 'Signal', and 'Hist' columns exist
    assert 'MACD' in result.columns
    assert 'Signal' in result.columns
    assert 'Hist' in result.columns

    # Check if result length matches the input data length
    assert len(result) == len(df)

    # Check if values in columns are numeric
    assert pd.api.types.is_numeric_dtype(result['MACD'])
    assert pd.api.types.is_numeric_dtype(result['Signal'])
    assert pd.api.types.is_numeric_dtype(result['Hist'])


def test_macd_strategy_logic():
    """
    Tests the logic of the MACD trading strategy.

    Checks if the backtest runs without errors and returns valid trades.
    """
    df = create_sample_data()

    # Set up a mock Backtest instance
    MACDStrategy.data = df
    bt = Backtest(df, MACDStrategy, cash=10**5, commission=0.001)

    # Run the backtest
    stats = bt.run()

    # Check if the number of trades is valid
    assert stats['# Trades'] >= 0


def test_execute_macd_strategy():
    """
    Tests the function that runs the MACD strategy.

    Checks if valid stats are returned after executing the strategy.
    """
    df = create_sample_data()

    # Execute the MACD strategy
    stats = execute_macd_strategy(df, MACDStrategy)

    # Check if expected keys are in the stats
    assert 'Return [%]' in stats
    assert 'Sharpe Ratio' in stats


def test_optimize_macd_strategy():
    """
    Tests the function that optimizes the MACD strategy.

    Checks if the optimization produces valid results and heatmap.
    """
    df = create_sample_data()

    # Define ranges for fast, slow, and signal periods
    fast_periods = [12, 14, 16]
    slow_periods = [26, 28, 30]
    signal_periods = [9, 10, 11]
    metric = 'Sharpe Ratio'

    # Run the optimization
    stats, heatmap = optimize_macd_strategy(
        df, MACDStrategy, fast_periods, slow_periods, signal_periods, metric
    )

    # Check if optimization results contain expected stats
    assert 'Return [%]' in stats
    assert 'Sharpe Ratio' in stats
    assert heatmap is not None  # Ensure heatmap was returned


def test_invalid_data_input():
    """
    Tests if an error is raised when input data is missing required columns.

    Specifically checks for missing 'Close' column.
    """
    # Create invalid data (no 'Close' column)
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    }
    df = pd.DataFrame(data)

    # Test that an error is raised when 'Close' is missing
    with pytest.raises(KeyError):
        make_macd_indicators(df)
