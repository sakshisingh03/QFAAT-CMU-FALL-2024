"""
This script implements a MACD strategy using the backtesting library.
It includes functions for:
- Calculating MACD indicators
- Defining a custom MACD-based strategy
- Running the strategy with user-defined parameters
- Optimizing strategy parameters and returning performance metrics.
"""

import talib
from backtesting import Backtest, Strategy
from utils import utils


def make_macd_indicators(data, fast_period=12, slow_period=26,
                         signal_period=9):
    """
    Calculate the MACD indicators (MACD, Signal,
    and Histogram) for a given dataset.

    Parameters:
        data (pd.DataFrame): The input data containing 'Close' prices.
        fast_period (int): The fast period for the MACD calculation
        (default is 12).
        slow_period (int): The slow period for the MACD calculation
        (default is 26).
        signal_period (int): The signal period for the MACD calculation
        (default is 9).

    Returns:
        pd.DataFrame: The data with added 'MACD', 'Signal', and 'Hist' columns.
    """
    try:
        data = data.copy()
        data['MACD'], data['Signal'], data['Hist'] = talib.MACD(
            data['Close'],
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
    except Exception as e:
        print(f"Error calculating MACD indicators: {e}")
        raise
    return data


class MACDStrategy(Strategy):
    """A custom strategy class that uses the MACD indicators to make buy
    and sell decisions."""
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        """
        Initializes the MACD indicators for the strategy.
        """
        self.data = make_macd_indicators(
            self.data,
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period
        )
        self.macd = self.I(lambda: self.data['MACD'])
        self.signal = self.I(lambda: self.data['Signal'])

    def next(self):
        """
        Defines the buy and sell logic based on MACD crossovers.
        """
        # Buy when MACD crosses above the Signal line
        if self.macd[-1] > self.signal[-1] and \
                self.macd[-2] <= self.signal[-2]:
            self.buy()
        # Sell when MACD crosses below the Signal line
        elif self.macd[-1] < self.signal[-1] and \
                self.macd[-2] >= self.signal[-2]:
            self.position.close()


def execute_macd_strategy(data, MACDStrategy, fast_period=12, slow_period=26,
                          signal_period=9, cash=10**5, commission=0.001,
                          exclusive_orders=True):
    """
    Executes the MACD strategy on the provided data.

    Parameters:
        data (pd.DataFrame): The input price data.
        MACDStrategy (Strategy): The custom MACD strategy class.
        fast_period (int): The fast period for the MACD calculation.
        slow_period (int): The slow period for the MACD calculation.
        signal_period (int): The signal period for the MACD calculation.
        cash (float): The initial cash for the backtest.
        commission (float): The commission for trades.
        exclusive_orders (bool): Whether to allow only one order at a time.

    Returns:
        pd.Series: Performance statistics of the strategy.
    """
    try:
        strategy = MACDStrategy
        strategy.fast_period = fast_period
        strategy.slow_period = slow_period
        strategy.signal_period = signal_period
        strategy.data = data
        bt = Backtest(data, strategy, cash=cash, commission=commission,
                      exclusive_orders=exclusive_orders)
        stats = bt.run()
        stats = utils.add_cagr_to_stats(stats, cash)
    except Exception as e:
        print(f"Error executing MACD strategy: {e}")
        raise
    return stats


def optimize_macd_strategy(data, MACDStrategy, fast_periods, slow_periods,
                           signal_periods, metric):
    """
    Optimizes the MACD strategy by adjusting the MACD periods and returns
    performance statistics and heatmap.

    Parameters:
        data (pd.DataFrame): The input price data.
        MACDStrategy (Strategy): The custom MACD strategy class.
        fast_periods (list): The range of fast periods to test.
        slow_periods (list): The range of slow periods to test.
        signal_periods (list): The range of signal periods to test.
        metric (str): The performance metric to maximize ('Return',
        'SharpeRatio', etc.).

    Returns:
        pd.Series, pd.DataFrame: Performance statistics and heatmap of results.
    """
    try:
        strategy = MACDStrategy
        strategy.data = data
        cash = 10**5
        bt = Backtest(data, strategy, cash=cash, commission=0.000,
                      exclusive_orders=True)
        stats, heatmap = bt.optimize(
            fast_period=fast_periods,
            slow_period=slow_periods,
            signal_period=signal_periods,
            maximize=metric,
            method='grid',
            max_tries=100,
            return_heatmap=True
        )
        stats = utils.add_cagr_to_stats(stats, cash)
    except Exception as e:
        print(f"Error optimizing MACD strategy: {e}")
        raise
    return stats, heatmap
