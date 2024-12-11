import pytest
import pandas as pd
from backtesting import Backtest
from lib_keltner_channels.keltner_modules import (
    make_keltner_indicators,
    KeltnerStrategy,
    execute_keltner_strategy,
    optimize_keltner_strategy
)
from utils import utils


# Function to create sample data for testing
def create_sample_data():
    """
    Creates a sample DataFrame with basic stock market data for testing.

    Returns:
        pd.DataFrame: DataFrame with 'Open', 'High', 'Low', 'Close'  and dates.
    """
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    }

    # Generate a date range (10 days starting from '2024-01-01')
    dates = pd.date_range(start='2024-01-01',
                          periods=len(data['Open']), freq='D')

    # Create DataFrame with date as index
    df = pd.DataFrame(data, index=dates)

    return df


# Test Case 1: Test make_keltner_indicators function
def test_make_keltner_indicators():
    """
    Tests the make_keltner_indicators function to ensure it adds correct cols
    and the values are numeric.
    """
    df = create_sample_data()

    # Calculate Keltner indicators
    result = make_keltner_indicators(df)

    # Check if the necessary columns are added
    assert 'ATR' in result.columns
    assert 'Middle' in result.columns
    assert 'Upper' in result.columns
    assert 'Lower' in result.columns

    # Check if the length of the result matches the input length
    assert len(result) == len(df)

    # Check that the values are numeric
    assert pd.api.types.is_numeric_dtype(result['ATR'])
    assert pd.api.types.is_numeric_dtype(result['Middle'])
    assert pd.api.types.is_numeric_dtype(result['Upper'])
    assert pd.api.types.is_numeric_dtype(result['Lower'])


def test_keltner_strategy_logic():
    """
    Tests the logic of the Keltner strategy using the Backtest class.
    """
    df = create_sample_data()

    # Create Backtest instance for the Keltner strategy
    KeltnerStrategy.data = df
    bt = Backtest(df, KeltnerStrategy, cash=10**5, commission=0.001)

    # Run the backtest
    stats = bt.run()

    # Check if the backtest runs without error
    assert stats['# Trades'] >= 0


def test_execute_keltner_strategy():
    """Tests the execute_keltner_strategy function to ensure it returns
    expected stats."""
    df = create_sample_data()

    # Execute Keltner strategy
    stats = execute_keltner_strategy(df, KeltnerStrategy)

    # Check if stats are returned and have expected keys
    assert 'Return [%]' in stats
    assert 'Sharpe Ratio' in stats


def test_optimize_keltner_strategy():
    """
    Tests the optimize_keltner_strategy function to check if optimization runs
    correctly and returns valid stats and heatmap.
    """
    df = create_sample_data()

    # Define optimization ranges for periods and ATR factors
    periods = [20, 22, 24]
    atr_factors = [1.5, 1.6, 1.8]
    metric = 'Sharpe Ratio'

    # Run the optimization
    stats, heatmap = optimize_keltner_strategy(
        df, KeltnerStrategy, periods, atr_factors, metric
    )

    # Check if optimization results contain expected stats
    assert 'Return [%]' in stats
    assert 'Sharpe Ratio' in stats
    assert heatmap is not None  # Ensure heatmap was returned


def test_invalid_data_input():
    """Tests if an error is raised when input data is missing a required
    column ('Close')."""
    # Create invalid data (missing 'Close' column)
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    }
    df = pd.DataFrame(data)

    # Test that an error is raised when 'Close' column is missing
    with pytest.raises(KeyError):
        make_keltner_indicators(df)


def test_add_cagr_to_stats():
    """Tests if the add_cagr_to_stats function correctly adds CAGR to the
    stats dictionary."""
    # Create mock stats dictionary
    stats = {'Equity Final [$]': 110000, 'Equity Peak [$]': 100000,
             'Duration': pd.Timedelta(days=100)}
    initial_cash = 100000

    # Using utils function to calculate CAGR
    updated_stats = utils.add_cagr_to_stats(stats, initial_cash)

    # Check if 'CAGR' is added to the stats
    assert 'CAGR (%)' in updated_stats
    assert updated_stats['CAGR (%)'] > 0
