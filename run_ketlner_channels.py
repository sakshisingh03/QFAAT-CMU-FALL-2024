import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib_keltner_channels import keltner_modules as km
from configs import configs
import seaborn as sns

def execute_keltner(ticker, data, period, atr_factor):
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
    output_file = f'outputs/keltner_channels/execution_results_{ticker}.csv'
    df_report.to_csv(output_file, index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(stats._equity_curve.index, (stats._equity_curve['Equity'])/1e6,
             label='Equity Line')
    plt.title(f'Equity curve for: {ticker}')
    plt.xlabel('Years')
    plt.ylabel('Equity (Mn$)')
    plt.legend()
    output_file = f'outputs/keltner_channels/executed_equity_curve_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


def optimize_keltner(ticker, data, periods, atr_factors):
    report = []
    metric = "Sharpe Ratio"
    stats, heatmap = km.optimize_keltner_strategy(data, km.KeltnerStrategy, periods, atr_factors, metric)

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
    output_file = f'outputs/keltner_channels/optimization_results_{ticker}.csv'
    df_report.to_csv(output_file, index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(stats._equity_curve.index, (stats._equity_curve['Equity'])/1e6,
             label='Equity Line')
    plt.title(f'Equity curve for: {ticker}')
    plt.xlabel('Years')
    plt.ylabel('Equity (Mn$)')
    plt.legend()
    output_file = f'outputs/keltner_channels/optimized_equity_curve_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # create the heatmap
    temp_df = pd.DataFrame(heatmap).reset_index()
    heatmap_data = temp_df.pivot_table(
        index="period", 
        columns="atr_factor", 
        values="Sharpe Ratio"
    )

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data, 
        fmt=".2f", 
        cmap="coolwarm", 
        cbar_kws={'label': 'Metric Value'}
    )

    plt.title(f"Heatmap of Metric Value by Period and ATR Factor for: {ticker}")
    plt.xlabel("ATR Factor")
    plt.ylabel("Period")
    plt.tight_layout()
    output_file = f'outputs/keltner_channels/optimized_heatmap_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


def train_test_split_optimize_keltner(ticker, data, periods, atr_factors):
    report = []
    metric = "Sharpe Ratio"
    
    # Split the data into train and test sets
    split_index = int(len(data) * (2 / 3))
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    # Optimize on the training data
    train_stats, train_heatmap = km.optimize_keltner_strategy(
        train_data, km.KeltnerStrategy, periods, atr_factors, metric
    )
    
    # Extract optimized parameters
    optimized_period = train_stats._strategy.period
    optimized_atr_factor = train_stats._strategy.atr_factor
    
    # Execute strategy on the test data
    test_stats = km.execute_keltner_strategy(
        test_data, km.KeltnerStrategy, 
        period=optimized_period, 
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
        'period': optimized_period,
        'atr_factor': optimized_atr_factor
    })
    
    # Save test results
    df_report = pd.DataFrame(report)
    output_file = f'outputs/keltner_channels/train_test_results_{ticker}.csv'
    df_report.to_csv(output_file, index=False)
    
    # Plot test equity curve
    plt.figure(figsize=(15, 6))
    plt.plot(test_stats._equity_curve.index, test_stats._equity_curve['Equity'] / 1e6, label='Equity Line')
    plt.title(f'Test Equity Curve for: {ticker}')
    plt.xlabel('Years')
    plt.ylabel('Equity (Mn$)')
    plt.legend()
    output_file = f'outputs/keltner_channels/test_equity_curve_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # Create heatmap for training optimization
    temp_df = pd.DataFrame(train_heatmap).reset_index()
    heatmap_data = temp_df.pivot_table(index="period", columns="atr_factor", values="Sharpe Ratio")

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Sharpe Ratio'})
    plt.title(f"Training Heatmap for: {ticker}")
    plt.xlabel("ATR Factor")
    plt.ylabel("Period")
    plt.tight_layout()
    output_file = f'outputs/keltner_channels/train_heatmap_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: run_keltner.py <purpose> <instrument_type>")
        print("Usage: run_keltner_channels.py <purpose>")
        sys.exit(1)

    # fetch data for equities/currencies based on instrument_type
    purpose = sys.argv[1]
    instrument_type = sys.argv[2]
    start_date = configs.config[instrument_type]['start_date']
    end_date = configs.config[instrument_type]['end_date']
    tickers = configs.config[instrument_type]['tickers']

    data_obj = utils.getYahooFinanceData()
    interval = '1h' if instrument_type == 'currency' else '1d'
    data = data_obj.fetch_data(start_date, end_date, tickers, interval=interval)
        
    if purpose == "execute":
        period = 20
        atr_factor = 2
        for ticker in tickers:
            execute_keltner(ticker, data[ticker], period=period, atr_factor=atr_factor)
        # combine execution results in one csv
        utils.combine_execution_results(instrument_type, tickers, 'outputs/keltner_channels/')
    elif purpose == "optimize":
        periods = [10, 20, 30, 40]
        atr_factors = [1.0, 1.5, 2.0, 2.5]
        for ticker in tickers:
            optimize_keltner(ticker, data[ticker], periods, atr_factors)
        # combine optimization results in one csv
        utils.combine_optimization_results(instrument_type, tickers, 'outputs/keltner_channels/')
    elif purpose == "train_test_optimize":
        periods = [10, 20, 30, 40]
        atr_factors = [1.0, 1.5, 2.0, 2.5]
        for ticker in tickers:
            train_test_split_optimize_keltner(ticker, data[ticker], periods, atr_factors)
        # combine train test results in one csv
        utils.combine_train_test_results(instrument_type, tickers, 'outputs/keltner_channels/')
    else:
        print("No valid purpose to run")
        exit(1)
