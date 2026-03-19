import yfinance as yf
import pandas as pd
import os

def crawl_gold_data(symbol="GLD", period="3y", interval="1d"):
    """
    Crawl gold market data (GLD ETF for stability) and save to CSV.
    """
    print(f"Fetching data for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            print("No data found.")
            return
        
        # Data Cleaning: Remove outliers (e.g., changes > 10% in a single day are likely errors)
        data['Pct_Change'] = data['Close'].pct_change().abs()
        data = data[data['Pct_Change'] < 0.1].copy()
        data.drop(columns=['Pct_Change'], inplace=True)
        
        # Reset index to have Date as a column
        data = data.reset_index()
        
        # Save to CSV
        output_file = os.path.join(os.path.dirname(__file__), "gold_data.csv")
        data.to_csv(output_file, index=False)
        print(f"Successfully saved cleaned data to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    crawl_gold_data()
