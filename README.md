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
│   ├── configs
│   │  ├── configs.py
│   ├── utils
│   │  ├── utils.py
│   ├── lib_bollinger_bands
│   │  ├── bollinger_modules.py
│   ├── lib_keltner_channels
│   │  ├── keltner_modules.py
│   ├── lib_macd
│   │  ├── macd_modules.py
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

## Key Features

- **Strategy Implementation**:
  - Keltner Channels
  - Bollinger Bands
  - MACD
- **Asset Classes**:
  - Equities
  - Currencies
- **Backtesting Framework**:
  - Robust backtesting capabilities to evaluate strategy performance.
- **Parameter Optimization**:
  - Exploration of optimal parameter settings for each strategy.
- **Performance Analysis**:
  - Detailed performance metrics, including Sharpe ratio, maximum drawdown, and return on investment.

## How to Use

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```
2. **Run a Strategy:**
To run a specific strategy for a particular asset class, use the following command:
```bash
python run_<strategy_name>.py <purpose> <instrument_type>
```
Replace <strategy_name> with bollinger_bands, keltner_channels, or macd.

Replace <'purpose'> with one of the following:
- execute: Execute the strategy on historical data for given parameters in code.
- optimize: Optimize the strategy's parameters and report the optimized parameters.
- train_test_optimize: To slice the data in train and test. Optimize on train data and evaluate on test data based on the parameters obtained on optimization.
- wfo: Run robustness check using Walk Forward Optimization. 


Replace <instrument_type> with equity or currency.

Example usage:

```bash
python run_bollinger_bands.py execute equity
```

The results and plots are saved under the outputs folder, within the folder name of the corresponding strategy. 

To execute the strategy with a parameter of your choice, change the parameters assigned under the line 'if purpose == "execute"'. All the parameters required for the particular strategies are given here. 

**Additional Notes**

- Ensure you have necessary data sources (e.g., Yahoo Finance API) configured.
- Adjust parameters and configuration files as needed for specific strategies and asset classes.
- For more advanced usage and customization, refer to the code documentation and configuration files.

**Contributions**
We welcome contributions to improve the framework and add new strategies. Please feel free to fork the repository and submit pull requests.

**License**
This project is licensed under the MIT License.