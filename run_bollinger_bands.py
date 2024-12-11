import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
from lib_bollinger_bands import bollinger_modules as bm  # Import Bollinger Bands-related modules
from configs import configs
import seaborn as sns

def execute_bollinger(ticker, data, periods, atr_factors):
    report = []
    stats = bm.execute_bollinger_strategy(
            data, periods=periods, atr_factors=atr_factors
        ).run()

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
            'periods': periods,
            'atr_factors': atr_factors
        })

    df_report = pd.DataFrame(report)
    output_file = f'outputs/bollinger_bands/execution_results_{ticker}.csv'
    df_report.to_csv(output_file, index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(stats._equity_curve.index, (stats._equity_curve['Equity'])/1e6,
             label='Equity Line')
    plt.title('Equity curve for: ' + ticker)
    plt.xlabel('Years')
    plt.ylabel('Equity (Mn$)')
    plt.legend()
    output_file = f'outputs/bollinger_bands/executed_equity_curve_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


def optimize_bollinger(ticker, data, periods_range, atr_factors_range):
    report = []
    metric = "Sharpe Ratio"
    stats, heatmap = bm.optimize_bollinger_strategy(
        data, periods_range, atr_factors_range, metric
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
            'periods': stats._strategy.periods,
            'atr_factors': stats._strategy.atr_factors
        })

    df_report = pd.DataFrame(report)
    output_file = f'outputs/bollinger_bands/optimization_results_{ticker}.csv'
    df_report.to_csv(output_file, index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(stats._equity_curve.index, (stats._equity_curve['Equity'])/1e6,
             label='Equity Line')
    plt.title('Equity curve for: ' + ticker)
    plt.xlabel('Years')
    plt.ylabel('Equity (Mn$)')
    plt.legend()
    output_file = f'outputs/bollinger_bands/optimized_equity_curve_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # create the heatmap
    temp_df = pd.DataFrame(heatmap).reset_index()
    heatmap_data = temp_df.pivot_table(
        index="periods", 
        columns="atr_factors", 
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

    plt.title("Heatmap of Metric Value by Periods and ATR Factors")
    plt.xlabel("ATR Factors")
    plt.ylabel("Periods")
    plt.tight_layout()
    output_file = f'outputs/bollinger_bands/optimized_heatmap_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


def train_test_split_optimize_bollinger(ticker, data, periods_range, atr_factors_range):
    report = []
    metric = "Sharpe Ratio"
    
    # Split the data into train and test sets
    split_index = int(len(data) * (2 / 3))
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    # Optimize on the training data
    train_stats, train_heatmap = bm.optimize_bollinger(
        train_data, periods_range, atr_factors_range, metric
    )
    
    # Extract optimized parameters
    optimized_periods = train_stats._strategy.periods
    optimized_atr_factors = train_stats._strategy.atr_factors
    
    # Execute strategy on the test data
    test_stats = bm.execute_bollinger_strategy(
        test_data, periods=optimized_periods, atr_factors=optimized_atr_factors
    ).run()
    
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
        'periods': optimized_periods,
        'atr_factors': optimized_atr_factors
    })
    
    # Save test results
    df_report = pd.DataFrame(report)
    output_file = f'outputs/bollinger_bands/train_test_results_{ticker}.csv'
    df_report.to_csv(output_file, index=False)
    
    # Plot test equity curve
    plt.figure(figsize=(15, 6))
    plt.plot(test_stats._equity_curve.index, test_stats._equity_curve['Equity'] / 1e6, label='Equity Line')
    plt.title(f'Test Equity Curve for: {ticker}')
    plt.xlabel('Years')
    plt.ylabel('Equity (Mn$)')
    plt.legend()
    output_file = f'outputs/bollinger_bands/test_equity_curve_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    # Create heatmap for training optimization
    temp_df = pd.DataFrame(train_heatmap).reset_index()
    heatmap_data = temp_df.pivot_table(index="periods", columns="atr_factors", values="Sharpe Ratio")

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Sharpe Ratio'})
    plt.title(f"Training Heatmap for: {ticker}")
    plt.xlabel("ATR Factors")
    plt.ylabel("Periods")
    plt.tight_layout()
    output_file = f'outputs/bollinger_bands/train_heatmap_{ticker}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: run_bollinger.py <purpose> <instrument_type>")
        print("Usage: run_bollinger_bands.py <purpose>")
        sys.exit(1)

    # fetch data for equities/currencies based on instrument_type
    purpose = sys.argv[1]
    instrument_type = sys.argv[2]
    start_date = configs.config[instrument_type]['start_date']
    end_date = configs.config[instrument_type]['end_date']
    tickers = configs.config[instrument_type]['tickers']

    data_obj = utils.getYahooFinanceData()
    interval = '1h' if instrument_type == 'currency' else '1d'
    data = data_obj.fetch_data(start_date, end_date, tickers, interval = interval)
        
    if purpose == "execute":
        periods = 20
        atr_factors = 1.1
        for ticker in tickers:
            execute_bollinger(ticker, data[ticker], periods, atr_factors)
            
        # combine execution results in one csv
        utils.combine_execution_results(instrument_type, tickers, 'outputs/bollinger_bands/')
    elif purpose == "optimize":
        periods_range = range(10, 30)       
        atr_factors_range = [1.1, 1.5, 2.0]  
        for ticker in tickers:
            optimize_bollinger(ticker, data[ticker], periods_range, atr_factors_range)
            
        # combine optimization results in one csv
        utils.combine_optimization_results(instrument_type, tickers, 'outputs/bollinger_bands/')
    elif purpose == "train_test_optimize":
        periods_range = range(10, 30)
        atr_factors_range = [1.1, 1.5, 2.0]
        for ticker in tickers:
            train_test_split_optimize_bollinger(ticker, data[ticker], periods_range, atr_factors_range)
        
        # combine train test results in one csv
        utils.combine_train_test_results(instrument_type, tickers, 'outputs/bollinger_bands/')
    else:
        print("No valid purpose to run")
        exit(1)
