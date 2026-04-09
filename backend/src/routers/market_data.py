from fastapi import APIRouter
import os
import pandas as pd

router = APIRouter(prefix="/api", tags=["market"])


@router.get("/market-data")
def get_market_data():
    """
    Return candlestick data for the chart.
    Falls back to an empty list if the CSV isn't available.
    """
    # Repo root: backend/src/routers -> backend/src -> backend -> repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    csv_path = os.path.join(repo_root, "gold_data.csv")

    if not os.path.exists(csv_path):
        return []

    try:
        df = pd.read_csv(csv_path)
        if "Date" not in df.columns:
            return []

        # gold_data.csv contains timezone offsets; normalize to UTC to avoid mixed-tz issues.
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date")

        chart_data = []
        for _, row in df.iterrows():
            chart_data.append(
                {
                    "time": int(row["Date"].timestamp()),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                }
            )

        return chart_data
    except Exception:
        return []
