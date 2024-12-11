import pytest
import pandas as pd
from backtesting import Backtest
from lib_bollinger_bands.bollinger_modules import (
    make_bollinger_indicators,
    BollingerStrategy,
    execute_bollinger_strategy,
    optimize_bollinger_strategy
)


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


# Test Case 1: Test make_bollinger_indicators function
def test_make_bollinger_indicators():
    """Tests the make_bollinger_indicators function to ensure it adds correct
    columnsand the values are numeric."""
    df = create_sample_data()

    # Calculate Bollinger Bands indicators
    result = make_bollinger_indicators(df)

    # Check if the necessary columns are added
    assert 'EMA' in result.columns
    assert 'STD' in result.columns
    assert 'Upper Band' in result.columns
    assert 'Lower Band' in result.columns
    assert 'BBW' in result.columns
    assert 'Signal' in result.columns

    # Check if the length of the result matches the input length
    assert len(result) == len(df)

    # Check that the values are numeric
    assert pd.api.types.is_numeric_dtype(result['EMA'])
    assert pd.api.types.is_numeric_dtype(result['STD'])
    assert pd.api.types.is_numeric_dtype(result['Upper Band'])
    assert pd.api.types.is_numeric_dtype(result['Lower Band'])
    assert pd.api.types.is_numeric_dtype(result['BBW'])
    assert pd.api.types.is_numeric_dtype(result['Signal'])


# Test Case 2: Test BollingerStrategy logic
def test_bollinger_strategy_logic():
    """
    Tests the logic of the Bollinger strategy using the Backtest class.
    """
    df = create_sample_data()

    # Create Backtest instance for the Bollinger strategy
    BollingerStrategy.data = df
    bt = Backtest(df, BollingerStrategy, cash=10**5, commission=0.001)

    # Run the backtest
    stats = bt.run()

    # Check if the backtest runs without error
    assert stats['# Trades'] >= 0


# Test Case 3: Test execute_bollinger_strategy function
def test_execute_bollinger_strategy():
    """Tests the execute_bollinger_strategy function to ensure it returns
    expected stats."""
    df = create_sample_data()

    # Execute Bollinger strategy
    stats = execute_bollinger_strategy(df, BollingerStrategy)

    # Check if stats are returned and have expected keys
    assert 'Return [%]' in stats
    assert 'Sharpe Ratio' in stats


# Test Case 4: Test optimize_bollinger_strategy function
def test_optimize_bollinger_strategy():
    """Tests the optimize_bollinger_strategy function to check if optimization
    runs correctly and returns valid stats and heatmap."""
    df = create_sample_data()

    # Define optimization ranges for periods and ATR factors
    periods = [20, 22, 24]
    atr_factors = [1.5, 1.6, 1.8]
    metric = 'Sharpe Ratio'

    # Run the optimization
    stats, heatmap = optimize_bollinger_strategy(
        df, BollingerStrategy, periods, atr_factors, metric
    )

    # Check if optimization results contain expected stats
    assert 'Return [%]' in stats
    assert 'Sharpe Ratio' in stats
    assert heatmap is not None  # Ensure heatmap was returned


# Test Case 5: Test invalid data input (missing 'Close' column)
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
        make_bollinger_indicators(df)
