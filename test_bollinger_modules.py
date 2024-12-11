import pytest
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib
from lib_bollinger_bands.bollinger_modules import make_bollinger_indicators, BollingerStrategy, execute_bollinger_strategy, optimize_bollinger_strategy

# Sample data for testing
def create_sample_data():
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'Close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    }
    return pd.DataFrame(data)

# Test the `make_bollinger_indicators` function
def test_make_bollinger_indicators():
    df = create_sample_data()

    # Call the function to calculate Bollinger indicators
    df_with_indicators = make_bollinger_indicators(df)

    # Test if the Bollinger Bands and BBW columns are created
    assert 'Upper Band' in df_with_indicators.columns
    assert 'Lower Band' in df_with_indicators.columns
    assert 'BBW' in df_with_indicators.columns
    assert 'Signal' in df_with_indicators.columns

    # Test the Signal generation (ensuring buy and sell signals are created)
    assert df_with_indicators['Signal'].iloc[0] == 0  # No signal at first point
    assert df_with_indicators['Signal'].iloc[1] == 0  # No signal at second point

# Test BollingerStrategy initialization and `next` method
def test_bollinger_strategy():
    df = create_sample_data()

    # Create a backtest instance with the BollingerStrategy
    BollingerStrategy.data = df
    bt = Backtest(df, BollingerStrategy, cash=10**5, commission=0.001)
    stats = bt.run()

    # Test if the strategy is initialized correctly
    assert isinstance(bt.strategy, BollingerStrategy)
    assert 'Buy' in stats  # Example of checking stats

# # Test `execute_bollinger_strategy`
# def test_execute_bollinger_strategy():
#     df = create_sample_data()

#     # Execute the strategy with the function
#     stats = execute_bollinger_strategy(df, BollingerStrategy, period=20, atr_factor=2)

#     # Test if the stats are returned and are in the expected format
#     assert 'Buy' in stats  # Example check on the result
#     assert isinstance(stats, dict)  # Ensure stats is a dictionary
#     assert 'cagr' in stats  # Check if CAGR is present in stats

# # Test `optimize_bollinger_strategy`
# def test_optimize_bollinger_strategy():
#     df = create_sample_data()
#     periods = [10, 20, 30]  # Example periods to test
#     atr_factors = [1.5, 2, 2.5]  # Example ATR factors to test
#     metric = 'SharpeRatio'  # Example metric to maximize during optimization

#     # Optimize the strategy
#     stats, heatmap = optimize_bollinger_strategy(df, BollingerStrategy, periods, atr_factors, metric)

#     # Check if stats and heatmap are returned correctly
#     assert isinstance(stats, dict)
#     assert 'SharpeRatio' in stats
#     assert isinstance(heatmap, np.ndarray)  # Ensure heatmap is a numpy array

# # Test the integration with mock data
# def test_integration_with_mock_data():
#     # Mock DataFrame with close prices for a backtest
#     data = create_sample_data()

#     # Run backtest
#     bt = Backtest(data, BollingerStrategy, cash=10**5, commission=0.001)
#     stats = bt.run()

#     # Verify the stats are reasonable (e.g., presence of metrics)
#     assert 'Buy' in stats
#     assert 'Total Return' in stats
