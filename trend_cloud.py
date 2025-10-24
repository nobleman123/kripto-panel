import pandas as pd
import numpy as np
import pandas_ta as ta

# ----------------------------
# 🔮 Sinyal Üretici Fonksiyon
# ----------------------------
def generate_specter_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fiyat hareketlerinden ve indikatörlerden yararlanarak
    net al/sat sinyalleri oluşturur.
    """
    df = df.copy()

    # EMA ve RSI eklentileri
    df["ema_fast"] = ta.ema(df["close"], length=9)
    df["ema_slow"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=14)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # Trend yönü
    df["trend"] = np.where(df["ema_fast"] > df["ema_slow"], "Bull", "Bear")

    # Momentum gücü
    df["momentum"] = df["close"] - df["ema_slow"]

    # Sinyal alanı
    df["signal"] = "HOLD"

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # AL SİNYALİ (trend yukarı, RSI dengede)
        if (
            prev["ema_fast"] < prev["ema_slow"]
            and curr["ema_fast"] > curr["ema_slow"]
            and curr["rsi"] < 70
            and curr["momentum"] > 0
        ):
            df.loc[df.index[i], "signal"] = "BUY"

        # SAT SİNYALİ (trend aşağı, RSI yüksek)
        elif (
            prev["ema_fast"] > prev["ema_slow"]
            and curr["ema_fast"] < curr["ema_slow"]
            and curr["rsi"] > 30
            and curr["momentum"] < 0
        ):
            df.loc[df.index[i], "signal"] = "SELL"

    # Fiyat sütunları
    df["close"] = df["close"].astype(float)
    df["time"] = pd.to_datetime(df["time"])

    # Sadece gerekli sütunları döndür
    return df[["time", "close", "signal", "rsi", "trend"]]
