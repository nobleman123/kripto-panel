import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from ai_engine import analyze_with_gemini
from trend_cloud import generate_specter_signal

# ----------------------------
# 🌐 AYARLAR
# ----------------------------
st.set_page_config(page_title="Kripto Sinyal Paneli", layout="wide")

GEMINI_API_KEY = "AIzaSyBn9dEGq1cTQqwUq6bnoQSEgmbaiJiRnks"
BASE_URL = "https://api.mexc.com/api/v3/klines"

# ----------------------------
# 📦 Yardımcı Fonksiyonlar
# ----------------------------
def get_mexc_data(symbol="BTCUSDT", interval="1h", limit=200):
    """MEXC borsasından OHLCV verisi çeker"""
    try:
        url = f"{BASE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        st.error(f"Veri alınamadı: {e}")
        return pd.DataFrame()

# ----------------------------
# 📊 Grafik Fonksiyonu
# ----------------------------
def create_specter_chart(df, specter_df, title="Specter Signal Chart"):
    """Fiyat, RSI ve sinyalleri içeren profesyonel Plotly grafiği oluşturur"""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        specs=[[{"secondary_y": True}], [{}]],
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05
    )

    # Fiyat grafiği
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price"
    ), row=1, col=1, secondary_y=False)

    # Sinyal noktaları
    if "signal" in specter_df.columns:
        long_signals = specter_df[specter_df["signal"] == "BUY"]
        short_signals = specter_df[specter_df["signal"] == "SELL"]

        fig.add_trace(go.Scatter(
            x=long_signals["time"], y=long_signals["close"],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="lime"),
            name="BUY"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=short_signals["time"], y=short_signals["close"],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="red"),
            name="SELL"
        ), row=1, col=1)

    # RSI veya Hacim grafiği
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["time"], y=df["rsi"],
            line=dict(color="orange", width=1),
            name="RSI"
        ), row=2, col=1)
    elif "volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df["time"], y=df["volume"], name="Volume"
        ), row=2, col=1)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800,
        showlegend=True
    )
    return fig

# ----------------------------
# 🤖 İndikatör Analizi
# ----------------------------
def add_indicators(df, strategy="Scalp"):
    """Zaman dilimine göre indikatör parametreleri ekler"""
    if strategy == "Scalp":
        short_ma, long_ma, rsi_len = 5, 20, 7
    elif strategy == "Swing":
        short_ma, long_ma, rsi_len = 10, 50, 14
    else:  # Long-term
        short_ma, long_ma, rsi_len = 20, 100, 21

    df["ema_short"] = ta.ema(df["close"], length=short_ma)
    df["ema_long"] = ta.ema(df["close"], length=long_ma)
    df["rsi"] = ta.rsi(df["close"], length=rsi_len)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["trend"] = np.where(df["ema_short"] > df["ema_long"], "Bull", "Bear")
    return df

# ----------------------------
# 🧠 Sinyal Üretimi
# ----------------------------
def generate_signal(df, strategy="Scalp"):
    """EMA, RSI ve trend bazlı sinyal üretimi"""
    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal = "HOLD"
    reason = "Piyasa nötr durumda."

    if strategy == "Scalp":
        if last["ema_short"] > last["ema_long"] and last["rsi"] < 70:
            signal, reason = "BUY", "Scalp alım fırsatı: EMA kesişimi ve RSI uygun."
        elif last["ema_short"] < last["ema_long"] and last["rsi"] > 30:
            signal, reason = "SELL", "Scalp satış fırsatı: EMA kesişimi ve RSI yüksek."
    else:
        if last["trend"] == "Bull" and last["rsi"] < 60:
            signal, reason = "BUY", "Trend yukarı ve RSI dengede."
        elif last["trend"] == "Bear" and last["rsi"] > 40:
            signal, reason = "SELL", "Trend aşağı ve RSI aşırı yüksek."

    return signal, reason

# ----------------------------
# 🚀 ARAYÜZ
# ----------------------------
st.title("🧠 Kripto Sinyal Analiz Paneli (MEXC + Gemini AI)")

symbol = st.sidebar.text_input("Sembol", "BTCUSDT")
interval = st.sidebar.selectbox("Zaman Dilimi", ["1m", "5m", "15m", "1h", "4h", "1d"])
strategy = st.sidebar.selectbox("Strateji Tipi", ["Scalp", "Swing", "Long-term"])
limit = st.sidebar.slider("Veri Limit", 100, 1000, 200)
analyze_btn = st.sidebar.button("Analiz Et")

if analyze_btn:
    with st.spinner("Veriler alınıyor ve analiz ediliyor..."):
        df = get_mexc_data(symbol, interval, limit)
        if not df.empty:
            df = add_indicators(df, strategy)
            signal, reason = generate_signal(df, strategy)

            ai_comment = analyze_with_gemini(
                df.tail(50).to_dict(),
                api_key=GEMINI_API_KEY,
                strategy=strategy,
                symbol=symbol,
                timeframe=interval,
                signal=signal
            )

            # Görsel grafik oluştur
            specter_df = generate_specter_signal(df)
            fig = create_specter_chart(df, specter_df, f"{symbol} - {strategy} ({interval})")

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📈 Sinyal Sonucu")
            st.success(f"**{signal}** → {reason}")

            st.subheader("🧩 AI Yorum (Gemini)")
            st.info(ai_comment if ai_comment else "AI yorum alınamadı.")
        else:
            st.error("Veri alınamadı, sembol veya zaman dilimini kontrol edin.")

st.caption("© 2025 SinanLab - MEXC & Gemini destekli kripto sinyal analizi.")
