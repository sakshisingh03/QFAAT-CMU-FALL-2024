import pandas as pd
import talib
from backtesting import Strategy

def make_bollinger_indicators(df, period=20, atr_factor=2):
    """
    Calculate Bollinger Bands, Bollinger Band Width (BBW), and generate signals based on low volatility (BBW).
    Buy signal when BBW is below threshold and price breaks above upper band.
    Sell signal when BBW is below threshold and price breaks below lower band.
    """
    df = df.copy()
    
    # Calculate Bollinger Bands
    df['SMA'] = df['Close'].ewm(span=period, adjust=False).mean()  # Exponential Moving Average (SMA)
    df['STD'] = df['Close'].rolling(window=period).std()  # Standard Deviation
    df['Upper Band'] = df['SMA'] + (df['STD'] * atr_factor)
    df['Lower Band'] = df['SMA'] - (df['STD'] * atr_factor)
    
    # Calculate Bollinger Band Width (BBW)
    df['BBW'] = (df['Upper Band'] - df['Lower Band']) / df['SMA']
    
    # Define a threshold for low BBW (suggested < 5% for low volatility)
    threshold = 0.05
    
    # Strategy: Buy when BBW is below threshold and price breaks above the upper band
    df['Signal'] = 0  # Initialize Signal column
    
    # Generate buy/sell signals
    df.loc[(df['BBW'] < threshold) & (df['Close'] > df['Upper Band']), 'Signal'] = 1  # Buy signal
    df.loc[(df['BBW'] < threshold) & (df['Close'] < df['Lower Band']), 'Signal'] = -1  # Sell signal

    return df


class BollingerStrategy(Strategy):
    period = 20
    atr_factor = 2

    def init(self):
        """Initialize Bollinger Bands indicators."""
        self.data = make_bollinger_indicators(self.data, period=self.period, atr_factor=self.atr_factor)
        self.signal = self.I(lambda: self.data['Signal'])
        self.close = self.I(lambda: self.data['Close'])
        self.bbw = self.I(lambda: self.data['BBW'])

    def next(self):
        """Define strategy logic based on the generated signals."""
        if self.signal[-1] == 1:  # Buy signal
            self.buy()
        elif self.signal[-1] == -1:  # Sell signal
            self.position.close()

def execute_bollinger_strategy(data, BollingerStrategy, period=20, atr_factor=2, cash=10**5, commission=0.001, exclusive_orders=True):
    strategy = BollingerStrategy
    strategy.period = period
    strategy.atr_factor = atr_factor
    strategy.data = data
    bt = Backtest(data, strategy, cash=cash, commission=commission, exclusive_orders=exclusive_orders)
    stats = bt.run()
    stats = utils.add_cagr_to_stats(stats, cash)
    return stats

def optimize_bollinger_strategy(data, BollingerStrategy, periods, atr_factors, metric):
    strategy = BollingerStrategy
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
