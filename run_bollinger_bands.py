"""This script executes, optimizes, or performs a train-test split
optimization on a Bollinger bands trading strategy for specified instruments"""

import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib_bollinger_bands import bollinger_modules as bm
from configs import configs
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import numpy as np


def execute_bollinger(ticker, data, period, atr_factor):
    """
    Executes the Bollinger Bands strategy on the provided data, generating
    reports and equity curves.

    Args:
        ticker (str): The ticker symbol for the financial instrument.
        data (pd.DataFrame): The historical data for the ticker.
        period (int): The period for the Bollinger Bands.
        atr_factor (float): The ATR factor for the strategy.

    Raises:
        ValueError: If data or stats are not available or incorrect.
    """
    try:
        report = []
        stats = bm.execute_bollinger_strategy(
            data, bm.BollingerStrategy, period=period, atr_factor=atr_factor
        )

        report.append({
            'start_date': stats['Start'],
            'end_date': stats['End'],
            'return_strat': stats['Return [%]'],
            'max_drawdown': stats['Max. Drawdown [%]'],
            'ret_strat_ann': stats['Return (Ann.) [%]'],
            'profit_factor': stats['Profit Factor'],
            'volatility_strat_ann': stats['Volatility (Ann.) [%]'],
            'sharpe_ratio': stats['Sharpe Ratio'],
            'return_bh': stats['Buy & Hold Return [%]'],
            'cagr': stats['CAGR (%)'],
            'periods': period,
            'atr_factors': atr_factor
        })

        df_report = pd.DataFrame(report)
        output_file = f'outputs/bollinger_bands/execution_results_{ticker}.csv'
        df_report.to_csv(output_file, index=False)

        # Plotting the equity curve
        plt.figure(figsize=(15, 6))
        plt.plot(stats._equity_curve.index,
                 (stats._equity_curve['Equity'])/1e6, label='Equity Line')
        plt.title(f'Equity curve for: {ticker}')
        plt.xlabel('Years')
        plt.ylabel('Equity (Mn$)')
        plt.legend()
        output_file = \
            f'outputs/bollinger_bands/executed_equity_curve_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    except ValueError as e:
        logging.error(f"Error in execute_bollinger: {e}")
        raise


def optimize_bollinger(ticker, data, periods_range, atr_factors_range):
    """
    Optimizes the Bollinger Bands strategy based on the Sharpe Ratio,
    generates reports and heatmaps.

    Args:
        ticker (str): The ticker symbol for the financial instrument.
        data (pd.DataFrame): The historical data for the ticker.
        periods_range (range): The range of periods to consider for
        optimization.
        atr_factors_range (list): The list of ATR factors to consider
        for optimization.

    Raises:
        ValueError: If data or stats are not available or incorrect.
    """
    try:
        report = []
        metric = "Sharpe Ratio"
        stats, heatmap = bm.optimize_bollinger_strategy(
            data, bm.BollingerStrategy, periods_range,
            atr_factors_range, metric)

        report.append({
            'start_date': stats['Start'],
            'end_date': stats['End'],
            'return_strat': stats['Return [%]'],
            'max_drawdown': stats['Max. Drawdown [%]'],
            'ret_strat_ann': stats['Return (Ann.) [%]'],
            'profit_factor': stats['Profit Factor'],
            'volatility_strat_ann': stats['Volatility (Ann.) [%]'],
            'sharpe_ratio': stats['Sharpe Ratio'],
            'return_bh': stats['Buy & Hold Return [%]'],
            'cagr': stats['CAGR (%)'],
            'periods': stats._strategy.period,
            'atr_factors': stats._strategy.atr_factor
        })

        df_report = pd.DataFrame(report)
        output_file = \
            f'outputs/bollinger_bands/optimization_results_{ticker}.csv'
        df_report.to_csv(output_file, index=False)

        # Plotting the optimized equity curve
        plt.figure(figsize=(15, 6))
        plt.plot(stats._equity_curve.index,
                 (stats._equity_curve['Equity'])/1e6, label='Equity Line')
        plt.title(f'Equity curve for: {ticker}')
        plt.xlabel('Years')
        plt.ylabel('Equity (Mn$)')
        plt.legend()
        output_file = \
            f'outputs/bollinger_bands/optimized_equity_curve_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        # Create and plot the heatmap
        temp_df = pd.DataFrame(heatmap).reset_index()
        heatmap_data = temp_df.pivot_table(
            index="period",
            columns="atr_factor",
            values="Sharpe Ratio"
        )

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            heatmap_data,
            fmt=".2f",
            cmap="coolwarm",
            cbar_kws={'label': 'Metric Value'}
        )

        plt.title("Heatmap of Metric Value by Periods and ATR Factors")
        plt.xlabel("ATR Factors")
        plt.ylabel("Periods")
        plt.tight_layout()
        output_file = f'outputs/bollinger_bands/optimized_heatmap_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    except ValueError as e:
        logging.error(f"Error in optimize_bollinger: {e}")
        raise


def train_test_split_optimize_bollinger(ticker, data, periods_range,
                                        atr_factors_range):
    """
    Optimizes the Bollinger Bands strategy using a train-test split approach.

    Args:
        ticker (str): The ticker symbol for the financial instrument.
        data (pd.DataFrame): The historical data for the ticker.
        periods_range (range): The range of periods to consider for
        optimization.
        atr_factors_range (list): The list of ATR factors to consider
        for optimization.

    Raises:
        ValueError: If data or stats are not available or incorrect.
    """
    try:
        report = []
        metric = "Sharpe Ratio"

        # Split the data into train and test sets
        split_index = int(len(data) * (2 / 3))
        train_data = data[:split_index]
        test_data = data[split_index:]

        # Optimize on the training data
        train_stats, train_heatmap = bm.optimize_bollinger_strategy(
            train_data, bm.BollingerStrategy, periods_range,
            atr_factors_range, metric)

        # Extract optimized parameters
        optimized_period = train_stats._strategy.period
        optimized_atr_factor = train_stats._strategy.atr_factor

        # Execute strategy on the test data
        test_stats = bm.execute_bollinger_strategy(
            test_data, bm.BollingerStrategy, period=optimized_period,
            atr_factor=optimized_atr_factor
        )

        report.append({
            'start_date': test_stats['Start'],
            'end_date': test_stats['End'],
            'return_strat': test_stats['Return [%]'],
            'max_drawdown': test_stats['Max. Drawdown [%]'],
            'ret_strat_ann': test_stats['Return (Ann.) [%]'],
            'profit_factor': test_stats['Profit Factor'],
            'volatility_strat_ann': test_stats['Volatility (Ann.) [%]'],
            'sharpe_ratio': test_stats['Sharpe Ratio'],
            'return_bh': test_stats['Buy & Hold Return [%]'],
            'cagr': test_stats['CAGR (%)'],
            'periods': optimized_period,
            'atr_factors': optimized_atr_factor
        })

        # Save test results
        df_report = pd.DataFrame(report)
        output_file = \
            f'outputs/bollinger_bands/train_test_results_{ticker}.csv'
        df_report.to_csv(output_file, index=False)

        # Plot test equity curve
        plt.figure(figsize=(15, 6))
        plt.plot(test_stats._equity_curve.index,
                 test_stats._equity_curve['Equity'] / 1e6, label='Equity Line')
        plt.title(f'Test Equity Curve for: {ticker}')
        plt.xlabel('Years')
        plt.ylabel('Equity (Mn$)')
        plt.legend()
        output_file = f'outputs/bollinger_bands/test_equity_curve_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        # Create heatmap for training optimization
        temp_df = pd.DataFrame(train_heatmap).reset_index()
        heatmap_data = temp_df.\
            pivot_table(index="period", columns="atr_factor",
                        values="Sharpe Ratio")

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, fmt=".2f", cmap="coolwarm",
                    cbar_kws={'label': 'Sharpe Ratio'})
        plt.title(f"Training Heatmap for: {ticker}")
        plt.xlabel("ATR Factors")
        plt.ylabel("Periods")
        plt.tight_layout()
        output_file = f'outputs/bollinger_bands/train_heatmap_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    except ValueError as e:
        logging.error(f"Error in train_test_split_optimize_bollinger: {e}")
        raise


def perform_wfo_bollinger(ticker, df_prices, iterations, periods, atr_factors):
    report = []
    metric = 'Sharpe Ratio'
    
    # Iterate over the list of iterations
    for iter in tqdm(iterations):
        # Filter the data to only include the relevant dates
        df_is = df_prices[(df_prices.index >= iter['in_sample'][0]) & (df_prices.index <= iter['in_sample'][1])]
        df_oos = df_prices[(df_prices.index >= iter['out_of_sample'][0]) & (df_prices.index <= iter['out_of_sample'][1])]

        # Calculate the optimal parameters using the in-sample data
        stats_is, heatmap = bm.optimize_bollinger_strategy(
            df_is, bm.BollingerStrategy, periods,
            atr_factors, metric)

        # Run the backtest for the out-of-sample data using the optimal parameters
        period = stats_is._strategy.period
        atr_factor = stats_is._strategy.atr_factor

        stats_oos = bm.execute_bollinger_strategy(
            df_oos, bm.BollingerStrategy, period=period,
            atr_factor=atr_factor
        )

        wfe = utils.calculate_walk_forward_metric(stats_oos['Sharpe Ratio'], stats_is['Sharpe Ratio'])

        # Append relevant metrics to a list of results
        report.append({
            'start_date': stats_oos['Start'],
            'end_date': stats_oos['End'],
            'return_strat': stats_oos['Return [%]'],
            'max_drawdown': stats_oos['Max. Drawdown [%]'],
            'ret_strat_ann': stats_oos['Return (Ann.) [%]'],
            'profit_factor': stats_oos['Profit Factor'],
            'volatility_strat_ann': stats_oos['Volatility (Ann.) [%]'],
            'is_sharpe_ratio': stats_is['Sharpe Ratio'],
            'oos_sharpe_ratio': stats_oos['Sharpe Ratio'],
            'return_bh': stats_oos['Buy & Hold Return [%]'],
            'WFE': wfe,
            'period': period,
            'atr_factor': atr_factor
        })

    df_report = pd.DataFrame(report)

    output_file = f'outputs/bollinger_bands/wfo_results_{ticker}.csv'
    df_report.to_csv(output_file, index=False)
    
    

if __name__ == "__main__":
    """Main script to execute or optimize the Bollinger Bands strategy based on
    provided arguments"""
    if len(sys.argv) != 3:
        logging.error("Usage: run_bollinger.py <purpose> <instrument_type>")
        print("Usage: run_bollinger_bands.py <purpose> <instrument_type>")
        sys.exit(1)
    try:
        # Fetch data for equities/currencies based on instrument_type
        purpose = sys.argv[1]
        instrument_type = sys.argv[2]
        start_date = configs.config[instrument_type]['start_date']
        end_date = configs.config[instrument_type]['end_date']
        tickers = configs.config[instrument_type]['tickers']

        interval = '1h' if instrument_type == 'currency' else '1d'

        # Fetch data and check if data is available
        data = utils.fetch_data_for_instruments(start_date, end_date,
                                                tickers, interval)

        # Execute based on the purpose argument
        if purpose == "execute":
            period = 20
            atr_factor = 1.1
            for ticker in tickers:
                execute_bollinger(ticker, data[ticker], period, atr_factor)

            # Combine execution results in one CSV
            utils.combine_execution_results(instrument_type, tickers,
                                            'outputs/bollinger_bands/')

        elif purpose == "optimize":
            periods_range = np.arange(5, 50, 1).tolist()
            atr_factors_range = np.arange(1, 10, 0.1).tolist()
            
            for ticker in tickers:
                optimize_bollinger(ticker, data[ticker], periods_range,
                                   atr_factors_range)

            # Combine optimization results in one CSV
            utils.combine_optimization_results(instrument_type, tickers,
                                               'outputs/bollinger_bands/')

        elif purpose == "train_test_optimize":
            periods_range = range(10, 30)
            atr_factors_range = [1.1, 1.5, 2.0]
            for ticker in tickers:
                train_test_split_optimize_bollinger(
                    ticker, data[ticker], periods_range, atr_factors_range)

            # Combine train-test results in one CSV
            utils.combine_train_test_results(instrument_type, tickers,
                                             'outputs/bollinger_bands/')
        elif purpose == "wfo":
            periods = np.arange(5, 50, 1).tolist()
            atr_factors = np.arange(1, 10, 0.1).tolist()

            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

            in_sample_duration = pd.DateOffset(months=6)
            out_of_sample_duration = pd.DateOffset(months=6)

            num_iterations = 3

            iterations = utils.define_walk_forward_iterations(start_date, end_date, in_sample_duration, out_of_sample_duration, num_iterations)

            for ticker in tickers:
                data_wfo = utils.localize_data(data[ticker])
                perform_wfo_bollinger(ticker, data_wfo, iterations, periods, atr_factors)
                
            # Combine optimization results in one csv
            utils.combine_wfo_results(instrument_type, tickers,
                                               'outputs/bollinger_bands/')

        else:
            print("No valid purpose to run")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
