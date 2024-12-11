import talib
import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
import os
from utils import utils

def make_keltner_indicators(data, period=20, atr_factor=2):
    """Calculate Keltner Channels for a given dataset."""
    data = data.copy()
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=period)
    data['Middle'] = talib.EMA((data['Close'] + data['High'] + data['Low']) / 3, timeperiod=period)
    data['Upper'] = data['Middle'] + atr_factor * data['ATR']
    data['Lower'] = data['Middle'] - atr_factor * data['ATR']
    return data


class KeltnerStrategy(Strategy):
    period = 20
    atr_factor = 1.6

    def init(self):
        """Initialize Keltner Channel values for strategy."""
        self.data = make_keltner_indicators(self.data, period=self.period, atr_factor=self.atr_factor)
        self.close = self.I(lambda: self.data['Close'])
        self.middle = self.I(lambda: self.data['Middle'])
        self.upper = self.I(lambda: self.data['Upper'])
        self.lower = self.I(lambda: self.data['Lower'])

    def next(self):
        """Define buy and sell logic based on Keltner Channels."""
        if self.close[-1] > self.middle[-1]:
            self.position.close()
        elif self.close[-1] < self.lower[-1]:
            self.buy()
            
    

def execute_macd_strategy(data, MACDStrategy, fast_period=12, slow_period=26, signal_period=9, 
                          cash=10**5, commission=0.001, exclusive_orders=True):
    strategy = MACDStrategy
    strategy.fast_period = fast_period
    strategy.slow_period = slow_period
    strategy.signal_period = signal_period
    strategy.data = data
    bt = Backtest(data, strategy, cash=cash, commission=commission, exclusive_orders=exclusive_orders)
    stats = bt.run()
    stats = utils.add_cagr_to_stats(stats, cash)
    return stats

def optimize_macd_strategy(data, MACDStrategy, fast_periods, slow_periods, signal_periods, metric):
    strategy = MACDStrategy
    strategy.data = data
    cash = 10**5
    bt = Backtest(data, strategy, cash=cash, commission=0.000, exclusive_orders=True)
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

def execute_keltner_strategy(data, KeltnerStrategy, period=20, atr_factor=2, cash=10**5, commission=0.001, exclusive_orders=True):
    strategy = KeltnerStrategy
    strategy.period = period
    strategy.atr_factor = atr_factor
    strategy.data = data
    bt = Backtest(data, strategy, cash=cash, commission=commission, exclusive_orders=exclusive_orders)
    stats = bt.run()
    stats = utils.add_cagr_to_stats(stats, cash)
    return stats

def optimize_keltner_strategy(data, KeltnerStrategy, periods, atr_factors, metric):
    strategy = KeltnerStrategy
    strategy.data = data
    cash = 10**5
    bt = Backtest(data, strategy, cash=cash, commission=0.000, exclusive_orders=True)
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
