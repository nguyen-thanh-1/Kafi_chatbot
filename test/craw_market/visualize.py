import pandas as pd
import mplfinance as mpf
import os

def plot_candlestick(csv_file=None):
    """
    Generate a candlestick chart from CSV data.
    """
    if csv_file is None:
        csv_file = os.path.join(os.path.dirname(__file__), "gold_data.csv")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run crawler.py first.")
        return

    try:
        # Load data
        df = pd.read_csv(csv_file)
        
        # Preprocess for mplfinance
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        df.set_index('Date', inplace=True)
        
        # Select OHLCV columns (case-sensitive)
        # yfinance history uses title case
        ohlc_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[ohlc_cols]
        
        print("Generating candlestick chart...")
        output_image = os.path.join(os.path.dirname(__file__), "gold_candlestick.png")
        
        # Plot and save
        mpf.plot(df, type='candle', style='charles',
                 title='Gold Price Candlestick Chart',
                 ylabel='Price (USD)',
                 volume=True,
                 savefig=output_image)
        
        print(f"Candlestick chart saved as {output_image}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_candlestick()
