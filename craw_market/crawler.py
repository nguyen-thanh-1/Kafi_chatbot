import yfinance as yf
import pandas as pd
import os

def crawl_gold_data(symbol="GC=F", period="1y", interval="1d"):
    """
    Crawl gold market data and save to CSV.
    Default symbol: GC=F (Gold Futures)
    """
    print(f"Fetching data for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            print("No data found.")
            return
        
        # Reset index to have Date as a column
        data = data.reset_index()
        
        # Save to CSV
        output_file = os.path.join(os.path.dirname(__file__), "gold_data.csv")
        data.to_csv(output_file, index=False)
        print(f"Successfully saved data to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    crawl_gold_data()
