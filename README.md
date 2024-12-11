# QFAAT-CMU-FALL-2024
This repository contains code for executing, optimizing, backtesting, and validating various trading strategies. 

# Algorithmic Trading Strategies Repository

## Overview

This repository provides a framework to backtest, optimize, and train/test algorithmic trading strategies using the `backtesting.py` library. It includes:

- Pre-built strategies: Keltner Channels, [Strategy 2], and [Strategy 3].
- Scripts to fetch data, execute strategies, optimize parameters, and split data into training and testing periods.
- Support for equity (daily data) and currency (hourly data).
- Functionality to fetch data from Yahoo Finance or Alpha Vantage.

---

## Features

1. **Execute Strategy**: Run a trading strategy with specified parameters.
2. **Optimize Parameters**: Optimize strategy parameters to identify the best combination.
3. **Train-Test Split**: Train and test strategies on different time periods.
4. **Instrument Types**:
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

Install the required packages with:

```bash
pip install -r requirements.txt


├── strategies/
│   ├── keltner_strategy.py
│   ├── strategy2.py
│   └── strategy3.py
├── scripts/
│   ├── run_keltner.py
│   ├── run_strategy2.py
│   └── run_strategy3.py
├── data/
│   └── (Downloaded datasets saved here)
├── README.md
├── requirements.txt
└── utils/
    ├── fetch_data.py
    ├── preprocess.py
    └── utils.py
```