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
            periods = [10, 20, 30, 40]
            atr_factors = [1.0, 1.5, 2.0, 2.5]
            for ticker in tickers:
                optimize_keltner(ticker, data[ticker], periods, atr_factors)
            utils.combine_optimization_results(instrument_type, tickers,
                                               'outputs/keltner_channels/')

        else:
            logging.error("No valid purpose provided")
            sys.exit(1)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
