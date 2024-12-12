"""
This script executes, optimizes, or performs a train-test split optimization
on a Keltner Channel trading strategy for specified instruments.
"""

import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib_keltner_channels import keltner_modules as km
from configs import configs
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def execute_keltner(ticker, data, period, atr_factor):
    """
    Executes the Keltner Channel strategy and saves the results.

    Args:
        ticker (str): The ticker symbol for the instrument.
        data (pd.DataFrame): Historical data for the instrument.
        period (int): Lookback period for the Keltner Channel.
        atr_factor (float): ATR factor for the channel width.
    """
    try:
        report = []
        stats = km.execute_keltner_strategy(
            data, km.KeltnerStrategy, period=period, atr_factor=atr_factor
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
            'period': period,
            'atr_factor': atr_factor
        })

        df_report = pd.DataFrame(report)
        output_file = \
            f'outputs/keltner_channels/execution_results_{ticker}.csv'
        df_report.to_csv(output_file, index=False)

        plt.figure(figsize=(15, 6))
        plt.plot(stats._equity_curve.index,
                 stats._equity_curve['Equity']/1e6, label='Equity Line')
        plt.title(f'Equity curve for: {ticker}')
        plt.xlabel('Years')
        plt.ylabel('Equity (Mn$)')
        plt.legend()
        output_file = \
            f'outputs/keltner_channels/executed_equity_curve_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    except Exception as e:
        logging.error(f"Error executing Keltner strategy for {ticker}: {e}")


def optimize_keltner(ticker, data, periods, atr_factors):
    """
    Optimizes the Keltner Channel strategy and saves the results.

    Args:
        ticker (str): The ticker symbol for the instrument.
        data (pd.DataFrame): Historical data for the instrument.
        periods (list): List of lookback periods to test.
        atr_factors (list): List of ATR factors to test.
    """
    try:
        report = []
        metric = "Sharpe Ratio"
        stats, heatmap = km.optimize_keltner_strategy(
            data, km.KeltnerStrategy, periods, atr_factors, metric)

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
            'period': stats._strategy.period,
            'atr_factor': stats._strategy.atr_factor
        })

        df_report = pd.DataFrame(report)
        output_file = \
            f'outputs/keltner_channels/optimization_results_{ticker}.csv'
        df_report.to_csv(output_file, index=False)

        plt.figure(figsize=(15, 6))
        plt.plot(stats._equity_curve.index,
                 stats._equity_curve['Equity']/1e6, label='Equity Line')
        plt.title(f'Equity curve for: {ticker}')
        plt.xlabel('Years')
        plt.ylabel('Equity (Mn$)')
        plt.legend()
        output_file = \
            f'outputs/keltner_channels/optimized_equity_curve_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        temp_df = pd.DataFrame(heatmap).reset_index()
        heatmap_data = temp_df.\
            pivot_table(index="period", columns="atr_factor",
                        values="Sharpe Ratio")

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, fmt=".2f", cmap="coolwarm",
                    cbar_kws={'label': 'Metric Value'})
        plt.title(f"Heatmap of Metric Value by Period and ATR for: {ticker}")
        plt.xlabel("ATR Factor")
        plt.ylabel("Period")
        plt.tight_layout()
        output_file = \
            f'outputs/keltner_channels/optimized_heatmap_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    except Exception as e:
        logging.error(f"Error optimizing Keltner strategy for {ticker}: {e}")


def train_test_split_optimize_keltner(ticker, data, periods, atr_factors):
    """
    Split data into train and test, optimize Keltner strategy, and save results.

    Args:
        ticker (str): The ticker symbol for the financial instrument.
        data (DataFrame): The market data for the instrument.
        fast_periods (range): The range of fast periods to optimize.
        slow_periods (range): The range of slow periods to optimize.
        signal_periods (range): The range of signal periods to optimize.
    """
    try:
        report = []
        metric = "Sharpe Ratio"

        # Split the data into train and test sets
        split_index = int(len(data) * (2 / 3))
        train_data = data[:split_index]
        test_data = data[split_index:]

        # Optimize on the training data
        train_stats, train_heatmap = km.optimize_keltner_strategy(
            train_data, km.KeltnerStrategy, periods, atr_factors, metric)

        # Extract optimized parameters
        optimized_period = train_stats._strategy.period
        optimized_atr_factors = train_stats._strategy.atr_factors

        # Execute strategy on the test data
        test_stats = km.execute_keltner_strategy(
            test_data, km.KeltnerStrategy, period=period, atr_factor=atr_factor
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
            'period': optimized_period,
            'atr_factors': optimized_atr_factors
        })

        # Save test results
        df_report = pd.DataFrame(report)
        output_file = f'outputs/keltner_channels/train_test_results_{ticker}.csv'
        df_report.to_csv(output_file, index=False)

        # Plot test equity curve
        plt.figure(figsize=(15, 6))
        plt.plot(test_stats._equity_curve.index,
                 test_stats._equity_curve['Equity']/1e6, label='Equity Line')
        plt.title(f'Test Equity Curve for: {ticker}')
        plt.xlabel('Years')
        plt.ylabel('Equity (Mn$)')
        plt.legend()
        output_file = f'outputs/keltner_channels/test_equity_curve_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        # Create heatmap for training optimization
        temp_df = pd.DataFrame(train_heatmap).reset_index()
        heatmap_data = temp_df.\
            pivot_table(index="fast_period", columns="slow_period",
                        values="Sharpe Ratio")

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, fmt=".2f", cmap="coolwarm",
                    cbar_kws={'label': 'Sharpe Ratio'})
        plt.title(f"Training Heatmap for: {ticker}")
        plt.xlabel("Slow Period")
        plt.ylabel("Fast Period")
        plt.tight_layout()
        output_file = f'outputs/keltner_channels/train_heatmap_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    except Exception as e:
        logging.\
            error(f"Error in train_test_split_optimize_keltner for {ticker}: {e}")
        print(f"Error in train_test_split_optimize_keltner for {ticker}: {e}")
        

def perform_wfo_keltner(ticker, df_prices, iterations, periods, atr_factors):
    report = []
    metric = 'Sharpe Ratio'
    
    print(iterations)

    # Iterate over the list of iterations
    for iter in tqdm(iterations):
        # Filter the data to only include the relevant dates
        df_is = df_prices[(df_prices.index >= iter['in_sample'][0]) & (df_prices.index <= iter['in_sample'][1])]
        df_oos = df_prices[(df_prices.index >= iter['out_of_sample'][0]) & (df_prices.index <= iter['out_of_sample'][1])]

        # Calculate the optimal parameters using the in-sample data
        stats_is, heatmap = km.optimize_keltner_strategy(
            df_is, km.KeltnerStrategy, periods, atr_factors, metric)

        # Run the backtest for the out-of-sample data using the optimal parameters
        period = stats_is._strategy.period
        atr_factor = stats_is._strategy.atr_factor

        stats_oos = km.execute_keltner_strategy(
            df_oos, km.KeltnerStrategy, period=period, atr_factor=atr_factor
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

    output_file = f'outputs/keltner_channels/wfo_results_{ticker}.csv'
    df_report.to_csv(output_file, index=False)
    
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: run_keltner.py <purpose> <instrument_type>")
        sys.exit(1)

    try:
        purpose = sys.argv[1]
        instrument_type = sys.argv[2]
        start_date = configs.config[instrument_type]['start_date']
        end_date = configs.config[instrument_type]['end_date']
        tickers = configs.config[instrument_type]['tickers']
        
        interval = '1h' if instrument_type == 'currency' else '1d'

        # Fetch data and check if data is available
        data = utils.fetch_data_for_instruments(start_date, end_date,
                                                tickers, interval)

        if purpose == "execute":
            period = 20
            atr_factor = 2
            for ticker in tickers:
                execute_keltner(ticker, data[ticker], period=period,
                                atr_factor=atr_factor)
            utils.combine_execution_results(instrument_type, tickers,
                                            'outputs/keltner_channels/')

        elif purpose == "optimize":
            periods = np.arange(5, 50, 1).tolist()
            atr_factors = np.arange(1, 10, 0.1).tolist()
            for ticker in tickers:
                optimize_keltner(ticker, data[ticker], periods, atr_factors)
            utils.combine_optimization_results(instrument_type, tickers,
                                               'outputs/keltner_channels/')
        elif purpose == "train_test_optimize":
            periods = [10, 20, 30, 40]
            atr_factors = [1.0, 1.5, 2.0, 2.5]
            for ticker in tickers:
                train_test_split_optimize_keltner(
                    ticker, data[ticker], periods, atr_factors)

            # Combine train test results in one csv
            utils.combine_train_test_results(instrument_type, tickers,
                                             'outputs/keltner_channels/')
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
                perform_wfo_keltner(ticker, data_wfo, iterations, periods, atr_factors)
                
            # Combine optimization results in one csv
            utils.combine_wfo_results(instrument_type, tickers,
                                               'outputs/keltner_channels/')
        else:
            logging.error("No valid purpose provided")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
