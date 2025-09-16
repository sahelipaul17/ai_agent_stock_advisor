import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sklearn.ensemble import RandomForestRegressor
from fastapi import FastAPI
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Gemini Client (via OpenAI SDK)
client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# FastAPI app
app = FastAPI(title="AI Stock Advisor with Gemini & Custom Tools")

MODEL_FILE = "rf_stock_model.joblib"


# -------- Utility: NSE/BSE Fallback --------
def resolve_ticker(ticker: str):
    """Try NSE (.NS), then BSE (.BO), then raw ticker"""
    suffixes = ["", ".NS", ".BO"] if "." not in ticker else [""]
    for suffix in suffixes:
        candidate = ticker + suffix if suffix else ticker
        t = yf.Ticker(candidate)
        try:
            df = t.history(period="1mo", interval="1d")
            if not df.empty:
                return candidate
        except Exception:
            continue
    return ticker  # fallback to raw


# -------- Tools --------
def fetch_stock_data(ticker: str, period="12mo", interval="1d"):
    ticker = resolve_ticker(ticker)
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return []
        df.reset_index(inplace=True)
        return df.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


def calculate_indicators(ticker: str, period="12mo", interval="1d"):
    ticker = resolve_ticker(ticker)
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, threads=False, timeout=30
        )
        if df.empty:
            return {"error": f"No data for {ticker}"}

        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["Momentum"] = df["Close"].diff()

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        latest = df.dropna().iloc[-1]
        return {
            "RSI": round(latest["RSI"], 2),
            "SMA_20": round(latest["SMA_20"], 2),
            "Momentum": round(latest["Momentum"], 2),
        }
    except Exception as e:
        return {"error": str(e)}


def get_model_advice(ticker: str):
    ticker = resolve_ticker(ticker)
    if not os.path.exists(MODEL_FILE):
        return {"error": "Model not trained. Please train first."}
    try:
        model = joblib.load(MODEL_FILE)
        df = yf.download(ticker, period="1y", progress=False).rename(columns={"Adj Close": "Adj_Close"})
        if df.empty:
            return {"error": f"No data for {ticker}"}

        df["sma_5"] = SMAIndicator(df["Adj_Close"], 5).sma_indicator()
        df["sma_20"] = SMAIndicator(df["Adj_Close"], 20).sma_indicator()
        df["rsi_14"] = RSIIndicator(df["Adj_Close"], 14).rsi()
        df["mom_5"] = df["Adj_Close"] / df["sma_5"] - 1
        df = df.dropna()

        latest = df.tail(1)[["Adj_Close", "sma_5", "sma_20", "rsi_14", "mom_5"]]
        pred = model.predict(latest)[0]
        return {"ticker": ticker, "predicted_return": float(pred)}
    except Exception as e:
        return {"error": str(e)}


# -------- Routes --------
@app.get("/")
def root():
    return {"message": "Stock Advisor AI Agent running 🚀"}


@app.get("/advise/{ticker}")
def advise(ticker: str):
    """
    AI Stock Advisor with Gemini + Custom Tools
    """
    # Tool results
    stock_data = fetch_stock_data(ticker)
    indicators = calculate_indicators(ticker)
    model_advice = get_model_advice(ticker)

    # Validate results
    if not stock_data:
        return {"error": f"No stock data found for {ticker}"}
    if isinstance(indicators, dict) and "error" in indicators:
        return {"error": indicators["error"]}
    if isinstance(model_advice, dict) and "error" in model_advice:
        return {"error": model_advice["error"]}

    # Prompts
    system_prompt = """
    You are a stock advisor AI agent. You have access to 3 tools:
    1) Stock data (recent OHLCV)
    2) Technical indicators (RSI, SMA, momentum)
    3) Model prediction (expected next-day return)

    Your job: Analyze the tools' results and give a recommendation:
    - BUY, HOLD, or SELL
    - Explain in simple terms using the indicators + model prediction
    - Include one risk management suggestion
    """

    user_prompt = f"""
    Stock: {ticker}
    Latest Data: {stock_data[-1]}
    Indicators: {indicators}
    Model Advice: {model_advice}
    """

    # Gemini call
    response = client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=400,
    )

    return {
        "ticker": ticker,
        "tools": {
            "stock_data": stock_data[-1],
            "indicators": indicators,
            "model_advice": model_advice,
        },
        "agent_reply": response.choices[0].message.content
    }
