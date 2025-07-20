# main.py
import os
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prophet import Prophet

# --- App Setup ---
app = FastAPI()

# IMPORTANT: This allows your frontend to talk to your backend
# In production, restrict this to your frontend's actual URL
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Keys & Endpoints ---
# Store keys as environment variables, not in the code!
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
AV_ENDPOINT = "https://www.alphavantage.co/query"

@app.get("/")
async def read_root():
    return {"message": "Market Analyzer API is running. Go to /analyze/{ticker} to get data."}

# --- Main Analysis Endpoint ---
@app.get("/analyze/{ticker}")
async def analyze_ticker(ticker: str):
    # 1. Fetch Data
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": ALPHA_VANTAGE_KEY,
    }
    try:
        response = requests.get(AV_ENDPOINT, params=params)
        response.raise_for_status() # Raises an error for bad responses (4xx or 5xx)
        data = response.json()["Time Series (Daily)"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error fetching data from provider: {e}")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Could not find data for ticker: {ticker}. Invalid ticker or API issue.")

    # 2. Process Data with Pandas
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.rename(columns={
        "1. open": "open", "2. high": "high", 
        "3. low": "low", "4. close": "close", 
        "5. adjusted close": "adj_close", "6. volume": "volume"
    })
    df = df.apply(pd.to_numeric)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index(ascending=True) # Sort from oldest to newest

    # 3. Calculate Technical Indicators
    # This automatically adds columns like 'RSI_14', 'MACD_12_26_9' to the DataFrame
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    
    # 4. Create & Run Forecast
    # Prophet requires columns named 'ds' (datestamp) and 'y' (value)
    prophet_df = df.reset_index().rename(columns={"index": "ds", "close": "y"})
    
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=90) # Forecast 90 days ahead
    forecast = model.predict(future)

    # 5. Return Everything
    return {
        "historical_data": df.reset_index().to_dict(orient="records"),
        "forecast_data": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient="records")
    }