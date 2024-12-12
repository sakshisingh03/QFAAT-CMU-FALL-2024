# QFAAT-CMU-FALL-2024
This repository contains code for executing, optimizing, backtesting, and validating various trading strategies. 

# Algorithmic Trading Strategies Repository

## Overview

This repository provides a framework to backtest, optimize, and train/test algorithmic trading strategies using the `backtesting.py` library. It includes:

- Pre-built strategies: Keltner Channels, [Strategy 2], and [Strategy 3].
- Scripts to fetch data, execute strategies, optimize parameters, split data into training and testing periods, and perform walk forward optimization.
- Support for equity (daily data) and currency (hourly data).
- Functionality to fetch data from Yahoo Finance or Alpha Vantage.

---

## Features

1. **Execute Strategy**: Run a trading strategy with specified parameters.
2. **Optimize Parameters**: Optimize strategy parameters to identify the best combination.
3. **Train-Test Split**: Train and test strategies on different time periods.
4. **Walk Forward Optimization**: Test strategy over increasing time range and check its performance.
5. **Instrument Types**:
   - **Equity**: Fetches daily data from Yahoo Finance.
   - **Currency**: Fetches hourly data from Yahoo Finance or Alpha Vantage.

---

## Setup

### Requirements

- Python 3.8+
- Libraries:
  - `backtesting.py`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `yfinance`
  - `alpha_vantage`
  - `talib`

Install the required packages with:

```bash
pip install -r requirements.txt


├── QFAAT-CMU-FALL-2024/
|   ├── configs
|   |  ├── configs.py
│   ├── utils
|   |  ├── utils.py
│   ├── lib_bollinger_bands
|   |  ├── bollinger_modules.py
|   ├── lib_keltner_channels
|   |  ├── keltner_modules.py
|   ├── lib_macd
|   |  ├── macd_modules.py
├── outputs/
│   ├── macd
│   ├── keltner_channels
│   └── bollinger_bands
├── run_bollinger_bands.py
├── run_keltner_channels.py
├── run_macd.py
├── test_keltner_modules.py
├── test_bollinger_modules.py
├── test_macd_modules.py
├── test_utils.py
├── requirements.txt
└── README.md
```