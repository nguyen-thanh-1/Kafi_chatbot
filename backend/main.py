from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import json

app = FastAPI()

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Trading API is running"}

@app.get("/api/market-data")
def get_market_data():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "craw_market", "gold_data.csv")
    
    if not os.path.exists(csv_path):
        return {"error": "Data not found. Please run the crawler first."}
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Ensure 'Date' is datetime and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Format for Lightweight Charts: { time: timestamp_in_seconds, open, high, low, close }
        chart_data = []
        for _, row in df.iterrows():
            chart_data.append({
                "time": int(row['Date'].timestamp()),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": float(row['Volume'])
            })
            
        return chart_data
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
