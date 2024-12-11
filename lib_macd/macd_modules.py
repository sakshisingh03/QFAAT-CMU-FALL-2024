import talib
import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
import os
from utils import utils

def make_macd_indicators(data, fast_period=12, slow_period=26, signal_period=9):
    data = data.copy()
    data['MACD'], data['Signal'], data['Hist'] = talib.MACD(
        data['Close'], 
        fastperiod=fast_period, 
        slowperiod=slow_period, 
        signalperiod=signal_period
    )
    return data

class MACDStrategy(Strategy):
    fast_period = 12
    slow_period = 26
    signal_period = 9

    def init(self):
        self.data = make_macd_indicators(
            self.data, 
            fast_period=self.fast_period, 
            slow_period=self.slow_period, 
            signal_period=self.signal_period
        )
        self.macd = self.I(lambda: self.data['MACD'])
        self.signal = self.I(lambda: self.data['Signal'])

    def next(self):
        # Buy when MACD crosses above the Signal line
        if self.macd[-1] > self.signal[-1] and self.macd[-2] <= self.signal[-2]:
            self.buy()
        # Sell when MACD crosses below the Signal line
        elif self.macd[-1] < self.signal[-1] and self.macd[-2] >= self.signal[-2]:
            self.position.close()    


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