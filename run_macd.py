"""This script executes, optimizes, or performs a train-test split
optimization on a MACD trading strategy for specified instruments"""

import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib_macd import macd_modules as mm
from configs import configs
import seaborn as sns


def execute_macd(ticker, data, fast_period, slow_period, signal_period):
    """
    Execute the MACD strategy and save the results as CSV and plots.

    Args:
        ticker (str): The ticker symbol for the financial instrument.
        data (DataFrame): The market data for the instrument.
        fast_period (int): The fast period for the MACD strategy.
        slow_period (int): The slow period for the MACD strategy.
        signal_period (int): The signal period for the MACD strategy.
    """
    try:
        report = []
        stats = mm.execute_macd_strategy(
            data, mm.MACDStrategy, fast_period=fast_period,
            slow_period=slow_period, signal_period=signal_period)

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
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period
        })

        df_report = pd.DataFrame(report)
        output_file = f'outputs/macd/execution_results_{ticker}.csv'
        df_report.to_csv(output_file, index=False)

        plt.figure(figsize=(15, 6))
        plt.plot(stats._equity_curve.index,
                 (stats._equity_curve['Equity'])/1e6, label='Equity Line')
        plt.title('Equity curve for: ' + ticker)
        plt.xlabel('Years')
        plt.ylabel('Equity (Mn$)')
        plt.legend()
        output_file = f'outputs/macd/executed_equity_curve_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    except Exception as e:
        logging.error(f"Error in execute_macd for {ticker}: {e}")
        print(f"Error in execute_macd for {ticker}: {e}")


def optimize_macd(ticker, data, fast_periods, slow_periods, signal_periods):
    """
    Optimize the MACD strategy and save the results as CSV and plots.

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
        stats, heatmap = mm.optimize_macd_strategy(
            data, mm.MACDStrategy, fast_periods,
            slow_periods, signal_periods, metric)

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
            'fast_period': stats._strategy.fast_period,
            'slow_period': stats._strategy.slow_period,
            'signal_period': stats._strategy.signal_period
        })

        df_report = pd.DataFrame(report)
        output_file = f'outputs/macd/optimization_results_{ticker}.png'
        df_report.to_csv(output_file, index=False)

        plt.figure(figsize=(15, 6))
        plt.plot(stats._equity_curve.index,
                 (stats._equity_curve['Equity'])/1e6, label='Equity Line')
        plt.title('Equity curve for: ' + ticker)
        plt.xlabel('Years')
        plt.ylabel('Equity (Mn$)')
        plt.legend()
        output_file = f'outputs/macd/optimized_equity_curve_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

        # Create the heatmap
        temp_df = pd.DataFrame(heatmap).reset_index()
        heatmap_data = temp_df.\
            pivot_table(index="fast_period", columns="slow_period",
                        values="Sharpe Ratio")

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, fmt=".2f", cmap="coolwarm",
                    cbar_kws={'label': 'Metric Value'})
        plt.title("Heatmap of Metric Value by Fast and Slow Periods")
        plt.xlabel("Slow Period")
        plt.ylabel("Fast Period")
        plt.tight_layout()
        output_file = f'outputs/macd/optimized_heatmap_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    except Exception as e:
        logging.error(f"Error in optimize_macd for {ticker}: {e}")
        print(f"Error in optimize_macd for {ticker}: {e}")


def train_test_split_optimize_macd(ticker, data, fast_periods, slow_periods,
                                   signal_periods):
    """
    Split data into train and test, optimize MACD strategy, and save results.

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
        train_stats, train_heatmap = mm.optimize_macd_strategy(
            train_data, mm.MACDStrategy, fast_periods, slow_periods,
            signal_periods, metric)

        # Extract optimized parameters
        optimized_fast_period = train_stats._strategy.fast_period
        optimized_slow_period = train_stats._strategy.slow_period
        optimized_signal_period = train_stats._strategy.signal_period

        # Execute strategy on the test data
        test_stats = mm.execute_macd_strategy(
            test_data, mm.MACDStrategy,
            fast_period=optimized_fast_period,
            slow_period=optimized_slow_period,
            signal_period=optimized_signal_period
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
            'fast_period': optimized_fast_period,
            'slow_period': optimized_slow_period,
            'signal_period': optimized_signal_period
        })

        # Save test results
        df_report = pd.DataFrame(report)
        output_file = f'outputs/macd/train_test_results_{ticker}.csv'
        df_report.to_csv(output_file, index=False)

        # Plot test equity curve
        plt.figure(figsize=(15, 6))
        plt.plot(test_stats._equity_curve.index,
                 test_stats._equity_curve['Equity']/1e6, label='Equity Line')
        plt.title(f'Test Equity Curve for: {ticker}')
        plt.xlabel('Years')
        plt.ylabel('Equity (Mn$)')
        plt.legend()
        output_file = f'outputs/macd/test_equity_curve_{ticker}.png'
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
        output_file = f'outputs/macd/train_heatmap_{ticker}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    except Exception as e:
        logging.\
            error(f"Error in train_test_split_optimize_macd for {ticker}: {e}")
        print(f"Error in train_test_split_optimize_macd for {ticker}: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: run_macd.py <purpose> <instrument_type>")
        print("Usage: run_macd.py <purpose> <instrument_type>")
        sys.exit(1)

    # fetch data for equities/currencies based on instrument_type
    try:
        purpose = sys.argv[1]
        instrument_type = sys.argv[2]
        start_date = configs.config[instrument_type]['start_date']
        end_date = configs.config[instrument_type]['end_date']
        tickers = configs.config[instrument_type]['tickers']

        data_obj = utils.getYahooFinanceData()
        interval = '1h' if instrument_type == 'currency' else '1d'
        data = data_obj.fetch_data(start_date, end_date, tickers,
                                   interval=interval)

        if purpose == "execute":
            fast_period = 12
            slow_period = 16
            signal_period = 6
            for ticker in tickers:
                execute_macd(
                    ticker, data[ticker], fast_period, slow_period,
                    signal_period)

            # Combine execution results in one csv
            utils.combine_execution_results(instrument_type, tickers,
                                            'outputs/macd/')

        elif purpose == "optimize":
            fast_periods = range(5, 50)
            slow_periods = range(5, 50)
            signal_periods = range(5, 50)
            for ticker in tickers:
                optimize_macd(
                    ticker, data[ticker], fast_periods, slow_periods,
                    signal_periods)

            # Combine optimization results in one csv
            utils.combine_optimization_results(instrument_type, tickers,
                                               'outputs/macd/')

        elif purpose == "train_test_optimize":
            fast_periods = range(5, 50)
            slow_periods = range(5, 50)
            signal_periods = range(5, 50)
            for ticker in tickers:
                train_test_split_optimize_macd(
                    ticker, data[ticker], fast_periods, slow_periods,
                    signal_periods)

            # Combine train test results in one csv
            utils.combine_train_test_results(instrument_type, tickers,
                                             'outputs/macd/')
        else:
            print("No valid purpose to run")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")
        sys.exit(1)
