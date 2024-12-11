"""This script implements a Bollinger Bands-based trading strategy,
including backtesting, optimization, and signal generation."""


import pandas as pd
from backtesting import Backtest, Strategy
from utils import utils


def make_bollinger_indicators(df, period=20, atr_factor=2):
    """
    Calculate Bollinger Bands, Bollinger Band Width (BBW), and generate signals
    based on low volatility (BBW).

    A Buy signal is generated when BBW is below threshold and price breaks
    above the upper band.
    A Sell signal is generated when BBW is below threshold and price breaks
    below the lower band.

    Parameters:
    df (pd.DataFrame): The dataframe containing the price data.
    period (int): The period for calculating Bollinger Bands (default 20).
    atr_factor (float): The multiplier for the standard deviation to calculate
    the bands (default 2).

    Returns:
    pd.DataFrame: The dataframe with the calculated Bollinger Bands, BBW, and
    Buy/Sell signals.
    """
    try:
        df = df.copy()

        # Calculate Bollinger Bands
        df['EMA'] = df['Close'].ewm(span=period, adjust=True).mean()
        df['STD'] = df['Close'].rolling(window=period).std()
        df['Upper Band'] = df['EMA'] + (df['STD'] * atr_factor)
        df['Lower Band'] = df['EMA'] - (df['STD'] * atr_factor)

        # Calculate Bollinger Band Width (BBW)
        df['BBW'] = (df['Upper Band'] - df['Lower Band']) / df['EMA']

        # Define a threshold for low BBW (suggested < 5% for low volatility)
        threshold = 0.04

        # Initialize Signal column
        df['Signal'] = 0

        # Generate buy/sell signals
        df.loc[(df['BBW'] < threshold) & (df['Close'] > df['Upper Band']),
               'Signal'] = 1  # Buy signal
        df.loc[(df['BBW'] < threshold) & (df['Close'] < df['Lower Band']),
               'Signal'] = -1  # Sell signal

        return df
    except Exception as e:
        print(f"Error in make_bollinger_indicators: {e}")
        return df


class BollingerStrategy(Strategy):
    """A Bollinger Bands-based strategy class that implements buy and sell
    signals based on the Bollinger Band Width (BBW) and price breaking the
    bands."""

    period = 20  # Period for Bollinger Bands calculation
    atr_factor = 2  # ATR factor for the bands

    def init(self):
        """Initialize Bollinger Bands indicators and signals."""
        try:
            # Initialize Bollinger Bands indicators
            self.data = make_bollinger_indicators(
                self.data, period=self.period, atr_factor=self.atr_factor)
            self.signal = self.I(lambda: self.data['Signal'])
            self.close = self.I(lambda: self.data['Close'])
            self.bbw = self.I(lambda: self.data['BBW'])
        except Exception as e:
            print(f"Error in BollingerStrategy initialization: {e}")

    def next(self):
        """Define strategy logic based on generated signals."""
        try:
            if self.signal[-1] == 1:  # Buy signal
                self.buy()
            elif self.signal[-1] == -1:  # Sell signal
                self.position.close()
        except Exception as e:
            print(f"Error in BollingerStrategy next step: {e}")


def execute_bollinger_strategy(data, BollingerStrategy, period=20,
                               atr_factor=2, cash=10**5, commission=0.001,
                               exclusive_orders=True):
    """
    Executes the Bollinger Bands strategy on the provided data and returns
    performance stats.

    Parameters:
    data (pd.DataFrame): The historical data for backtesting.
    BollingerStrategy (class): The strategy class to use for backtesting.
    period (int): The period for the Bollinger Bands (default 20).
    atr_factor (float): The ATR factor (default 2).
    cash (float): The initial cash for the backtest (default 10^5).
    commission (float): The commission per trade (default 0.001).
    exclusive_orders (bool): Whether orders should be exclusive (default True).

    Returns:
    pd.DataFrame: The performance statistics from the backtest.
    """
    try:
        strategy = BollingerStrategy
        strategy.period = period
        strategy.atr_factor = atr_factor
        strategy.data = data

        # Set up and run the backtest
        bt = Backtest(data, strategy, cash=cash, commission=commission,
                      exclusive_orders=exclusive_orders)
        stats = bt.run()

        # Add CAGR to stats
        stats = utils.add_cagr_to_stats(stats, cash)
        return stats
    except Exception as e:
        print(f"Error in execute_bollinger_strategy: {e}")
        return pd.DataFrame()


def optimize_bollinger_strategy(data, BollingerStrategy, periods, atr_factors,
                                metric):
    """
    Optimizes the Bollinger Bands strategy by testing different periods and
    ATR factors and returns the best performance based on the specified metric.

    Parameters:
    data (pd.DataFrame): The historical data for backtesting.
    BollingerStrategy (class): The strategy class to optimize.
    periods (list): A list of periods to test for the Bollinger Bands.
    atr_factors (list): A list of ATR factors to test.
    metric (str): The performance metric to optimize (e.g., 'SharpeRatio').

    Returns:
    pd.DataFrame: The optimized performance statistics.
    pd.DataFrame: The heatmap of optimization results.
    """
    try:
        strategy = BollingerStrategy
        strategy.data = data
        cash = 10**5

        # Set up and run the optimization
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

        # Add CAGR to stats
        stats = utils.add_cagr_to_stats(stats, cash)
        return stats, heatmap
    except Exception as e:
        print(f"Error in optimize_bollinger_strategy: {e}")
        return pd.DataFrame(), pd.DataFrame()
