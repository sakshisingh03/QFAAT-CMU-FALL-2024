"""This script implements Keltner Channels-based trading strategy,
including backtesting, optimization, and signal generation."""

import talib
from backtesting import Backtest, Strategy
from utils import utils


def make_keltner_indicators(data, period=20, atr_factor=2):
    """
    Calculate Keltner Channels for a given dataset.

    Parameters:
    data (pd.DataFrame): The dataset with 'High', 'Low', and 'Close' columns.
    period (int): The period for calculating the moving average and ATR.
    atr_factor (float): The multiplier for the ATR to define the upper and
    lower channels.

    Returns:
    pd.DataFrame: The dataset with Keltner Channel indicators ('Middle',
    'Upper', 'Lower', 'ATR').
    """
    try:
        data = data.copy()
        # Calculate ATR
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'],
                                timeperiod=period)
        # Calculate middle band using EMA
        data['Middle'] = talib.EMA((data['Close'] + data['High'] +
                                    data['Low'])/3, timeperiod=period)
        # Calculate upper and lower bands
        data['Upper'] = data['Middle'] + atr_factor * data['ATR']
        data['Lower'] = data['Middle'] - atr_factor * data['ATR']
        return data
    except Exception as e:
        raise ValueError(f"Error in calculating Keltner indicators: {e}")


class KeltnerStrategy(Strategy):
    """Keltner Channels trading strategy, generates buy and sell signals based
    on price crossing the channels."""
    period = 20
    atr_factor = 1.6

    def init(self):
        """
        Initialize Keltner Channel values for strategy.
        """
        try:
            # Generate Keltner indicators and store them
            self.data = make_keltner_indicators(self.data, period=self.period,
                                                atr_factor=self.atr_factor)
            self.close = self.I(lambda: self.data['Close'])
            self.middle = self.I(lambda: self.data['Middle'])
            self.upper = self.I(lambda: self.data['Upper'])
            self.lower = self.I(lambda: self.data['Lower'])
        except Exception as e:
            raise ValueError(f"Error in initializing Keltner strategy: {e}")

    def next(self):
        """
        Define buy and sell logic based on Keltner Channels.
        """
        try:
            if self.close[-1] > self.middle[-1]:
                # Close position if price is above middle band
                self.position.close()
            elif self.close[-1] < self.lower[-1]:
                self.buy()  # Buy signal when price is below the lower band
        except Exception as e:
            raise ValueError(f"Error in strategy logic: {e}")


def execute_macd_strategy(data, MACDStrategy, fast_period=12, slow_period=26,
                          signal_period=9, cash=10**5, commission=0.001,
                          exclusive_orders=True):
    """
    Execute a backtest using the MACD strategy with given parameters.

    Parameters:
    data (pd.DataFrame): The historical data.
    MACDStrategy (Strategy): The strategy class to use.
    fast_period (int): The fast period for MACD.
    slow_period (int): The slow period for MACD.
    signal_period (int): The signal period for MACD.
    cash (float): The initial cash for the strategy.
    commission (float): The commission rate for trading.
    exclusive_orders (bool): Whether to allow exclusive orders.

    Returns:
    pd.DataFrame: The backtest results with performance metrics.
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
        return stats
    except Exception as e:
        raise ValueError(f"Error executing MACD strategy: {e}")


def optimize_macd_strategy(data, MACDStrategy, fast_periods, slow_periods,
                           signal_periods, metric):
    """
    Optimize the MACD strategy using grid search to find the best parameters.

    Parameters:
    data (pd.DataFrame): The historical data.
    MACDStrategy (Strategy): The strategy class to optimize.
    fast_periods (list): List of fast periods to test.
    slow_periods (list): List of slow periods to test.
    signal_periods (list): List of signal periods to test.
    metric (str): The metric to maximize during optimization.

    Returns:
    pd.DataFrame, np.ndarray: The optimized strategy results and heatmap.
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
        return stats, heatmap
    except Exception as e:
        raise ValueError(f"Error optimizing MACD strategy: {e}")


def execute_keltner_strategy(data, KeltnerStrategy, period=20, atr_factor=2,
                             cash=10**5, commission=0.001,
                             exclusive_orders=True):
    """
    Execute a backtest using the Keltner strategy with given parameters.

    Parameters:
    data (pd.DataFrame): The historical data.
    KeltnerStrategy (Strategy): The strategy class to use.
    period (int): The period for Keltner channels.
    atr_factor (float): The multiplier for ATR.
    cash (float): The initial cash for the strategy.
    commission (float): The commission rate for trading.
    exclusive_orders (bool): Whether to allow exclusive orders.

    Returns:
    pd.DataFrame: The backtest results with performance metrics.
    """
    try:
        strategy = KeltnerStrategy
        strategy.period = period
        strategy.atr_factor = atr_factor
        strategy.data = data
        bt = Backtest(data, strategy, cash=cash, commission=commission,
                      exclusive_orders=exclusive_orders)
        stats = bt.run()
        stats = utils.add_cagr_to_stats(stats, cash)
        return stats
    except Exception as e:
        raise ValueError(f"Error executing Keltner strategy: {e}")


def optimize_keltner_strategy(data, KeltnerStrategy, periods, atr_factors,
                              metric):
    """
    Optimize the Keltner strategy using grid search to find best parameters.

    Parameters:
    data (pd.DataFrame): The historical data.
    KeltnerStrategy (Strategy): The strategy class to optimize.
    periods (list): List of periods to test.
    atr_factors (list): List of ATR factors to test.
    metric (str): The metric to maximize during optimization.

    Returns:
    pd.DataFrame, np.ndarray: The optimized strategy results and heatmap.
    """
    try:
        strategy = KeltnerStrategy
        strategy.data = data
        cash = 10**5
        bt = Backtest(data, strategy, cash=cash, commission=0.000,
                      exclusive_orders=True)
        stats, heatmap = bt.optimize(
            period=periods,
            atr_factor=atr_factors,
            maximize=metric,
            method='grid',
            max_tries=100,
            return_heatmap=True
        )
        stats = utils.add_cagr_to_stats(stats, cash)
        return stats, heatmap
    except Exception as e:
        raise ValueError(f"Error optimizing Keltner strategy: {e}")
