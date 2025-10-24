# ai_engine.py
# v3 - İndikatör/Skorlama + Hacim Dönüş + Strateji Kombinasyonu

import math
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import os
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta

# Gemini AI kütüphanesi
# ... (Import aynı kaldı) ...
try: import google.generativeai as genai; GEMINI_AVAILABLE = True
except ImportError: GEMINI_AVAILABLE = False; logging.warning("Gemini kütüphanesi...")

RECORDS_FILE = Path("prediction_records.json")
EMA_TREND_LENGTH = 200

# --------------- TEMEL MATEMATİK VE NORMALİZASYON (Aynı kaldı) ---------------
# ... (logistic, normalize) ...
def logistic(x): # ...
    pass
def normalize(v, lo, hi): # ...
    pass

# --------------- İNDİKATÖR HESAPLAMA (Yeni indikatörler eklendi) ---------------
def nw_smooth(series, bandwidth=8): # ... (Aynı kaldı) ...
    pass

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Teknik indikatörleri hesaplar (SuperTrend, SSL, ADX eklendi)."""
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy(); required_input_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_input_cols): return pd.DataFrame()

    # EMA'lar
    for length in [20, 50, EMA_TREND_LENGTH]:
        try: df[f'ema{length}'] = ta.ema(df['close'], length=length)
        except Exception as e: df[f'ema{length}'] = np.nan
    # MACD
    try: macd = ta.macd(df['close']); df['macd_hist'] = macd.iloc[:,1] if isinstance(macd, pd.DataFrame) and macd.shape[1]>=2 else np.nan
    except Exception as e: df['macd_hist'] = np.nan
    # RSI
    try: df['rsi14'] = ta.rsi(df['close'], length=14)
    except Exception as e: df['rsi14'] = np.nan
    # Bollinger Bands
    try: bb = ta.bbands(df['close']); df['bb_lower'] = bb.iloc[:,0]; df['bb_mid'] = bb.iloc[:,1]; df['bb_upper'] = bb.iloc[:,2]
    except Exception as e: df[['bb_lower','bb_mid','bb_upper']] = np.nan
    # ATR
    try: df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    except Exception as e: df['atr14'] = np.nan
    # Volume SMA ve Oscillator
    try:
        df['vol_sma20'] = ta.sma(df['volume'], length=20) # Hacim teyidi için SMA
        df['vol_ma_short'] = df['vol_sma20']; df['vol_ma_long'] = ta.sma(df['volume'], length=50)
        df['vol_osc'] = (df['vol_ma_short'] - df['vol_ma_long']) / (df['vol_ma_long'].replace(0, 1e-9) + 1e-9)
    except Exception as e: df[['vol_sma20','vol_ma_short','vol_ma_long','vol_osc']] = np.nan
    # NW Slope
    try:
        if len(df['close'].dropna()) > 10: sm = nw_smooth(df['close'].values, bandwidth=8); df['nw_smooth'] = sm; df['nw_slope'] = pd.Series(sm).diff().fillna(0)
        else: df['nw_smooth'] = np.nan; df['nw_slope'] = np.nan
    except Exception as e: df[['nw_smooth','nw_slope']] = np.nan

    # --- YENİ İNDİKATÖRLER ---
    # SuperTrend (Varsayılan Ayarlar 7, 3)
    try:
         st = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3)
         if isinstance(st, pd.DataFrame) and 'SUPERTd_7_3.0' in st.columns:
              df['supertrend_direction'] = st['SUPERTd_7_3.0'] # 1 for uptrend, -1 for downtrend
              df['supertrend_line'] = st['SUPERTl_7_3.0'] # Alt/Üst çizgi
         else: df['supertrend_direction'] = 0; df['supertrend_line'] = np.nan
    except Exception as e: logging.debug(f"SuperTrend hatası: {e}"); df['supertrend_direction'] = 0; df['supertrend_line'] = np.nan

    # SSL Channel (Varsayılan Ayarlar 14)
    try:
         # pandas_ta SSL eklentisi varsayılarak (yoksa manuel hesaplamak gerekir)
         # ssl = ta.ssl(df['high'], df['low'], df['close'], length=14) # Örnek kullanım, kütüphane desteği gerekebilir
         # Geçici manuel hesaplama (Basitleştirilmiş):
         sma_high = ta.sma(df['high'], length=14)
         sma_low = ta.sma(df['low'], length=14)
         hlv = np.where(df['close'] > sma_high, 1, np.where(df['close'] < sma_low, -1, 0))
         ssl_up = np.where(hlv == -1, sma_low, sma_high) # Basitleştirilmiş - Gerçek SSL daha karmaşık
         ssl_down = np.where(hlv == -1, sma_high, sma_low) # Basitleştirilmiş
         df['ssl_up'] = ssl_up
         df['ssl_down'] = ssl_down
         df['ssl_direction'] = np.where(df['close'] > df['ssl_up'], 1, np.where(df['close'] < df['ssl_down'], -1, 0)) # Yönü belirle
    except Exception as e: logging.debug(f"SSL hatası: {e}"); df[['ssl_up','ssl_down', 'ssl_direction']] = np.nan

    # ADX / DMI (Varsayılan Ayarlar 14)
    try:
         dmi = ta.dmi(df['high'], df['low'], df['close'], length=14)
         if isinstance(dmi, pd.DataFrame):
              df['adx'] = dmi['ADX_14']
              df['dmi_plus'] = dmi['DMP_14'] # Pozitif yönlü hareket
              df['dmi_minus'] = dmi['DMN_14'] # Negatif yönlü hareket
         else: df[['adx','dmi_plus','dmi_minus']] = np.nan
    except Exception as e: logging.debug(f"ADX/DMI hatası: {e}"); df[['adx','dmi_plus','dmi_minus']] = np.nan
    # --- YENİ İNDİKATÖRLER SONU ---


    # Gerekli sütunlar (NaN kontrolü için - yeni eklenenler dahil)
    required_cols = ['close', 'ema20', 'ema50', f'ema{EMA_TREND_LENGTH}', 'macd_hist', 'rsi14',
                     'bb_upper', 'bb_lower', 'atr14', 'vol_osc', 'nw_slope', 'vol_sma20',
                     'supertrend_direction', 'supertrend_line', 'ssl_direction', 'ssl_up', 'ssl_down',
                     'adx', 'dmi_plus', 'dmi_minus']
    original_len = len(df)
    df = df.dropna(subset=required_cols) # Tüm gerekli indikatörler hesaplanmış olmalı
    if len(df) < 3: logging.warning(f"compute_indicators: Hesaplama sonrası yetersiz veri ({len(df)}/{original_len}).")

    return df


# --------------- ALGORİTMA SKORLAMA (Aynı kaldı) ---------------
# ... (label_from_score, score_signals aynı kaldı) ...

# --------------- SEVİYE HESAPLAMA (Aynı kaldı) ---------------
# ... (compute_trade_levels aynı kaldı) ...

# --------------- HACİM TEYİTLİ DÖNÜŞ MOTORU (Aynı kaldı) ---------------
# ... (analyze_volume_reversal fonksiyonu aynı kaldı) ...
def analyze_volume_reversal(df_ind: pd.DataFrame, look_back: int = 20, confirm_in: int = 5, vol_multiplier: float = 1.5, use_ema_filter: bool = True) -> Dict[str, Any]:
    # ... içerik aynı ...
    return {...} # Sonuç sözlüğü

# --------------- YENİ: STRATEJİ KOMBİNASYON MOTORU ---------------
def analyze_strategy_combo(latest: pd.Series, adx_threshold: int = 20) -> Dict[str, Any]:
    """
    Birden fazla indikatörün onayına dayalı strateji sinyali üretir.
    """
    if latest is None or not isinstance(latest, pd.Series):
         return {"signal": "NONE", "confidence": 0, "confirming_indicators": [], "explanation": "Geçersiz giriş verisi"}

    confirmations_buy = []
    confirmations_sell = []
    conditions_met_buy = 0
    conditions_met_sell = 0
    total_conditions = 5 # Kontrol edilen koşul sayısı

    price = latest.get('close')
    if price is None or math.isnan(price): price = 0 # Fiyat yoksa devam etme

    # 1. EMA Cross
    ema20 = latest.get('ema20'); ema50 = latest.get('ema50')
    if ema20 is not None and ema50 is not None and not math.isnan(ema20) and not math.isnan(ema50):
        if ema20 > ema50: conditions_met_buy += 1; confirmations_buy.append("EMA Cross (20>50)")
        if ema20 < ema50: conditions_met_sell += 1; confirmations_sell.append("EMA Cross (20<50)")

    # 2. SuperTrend
    st_dir = latest.get('supertrend_direction'); st_line = latest.get('supertrend_line')
    if st_dir is not None and st_line is not None and not math.isnan(st_line):
        if st_dir == 1 and price > st_line: conditions_met_buy += 1; confirmations_buy.append("SuperTrend Up")
        if st_dir == -1 and price < st_line: conditions_met_sell += 1; confirmations_sell.append("SuperTrend Down")

    # 3. SSL Channel
    ssl_dir = latest.get('ssl_direction'); ssl_up = latest.get('ssl_up'); ssl_down = latest.get('ssl_down')
    if ssl_dir is not None and ssl_up is not None and ssl_down is not None:
         if ssl_dir == 1 and price > ssl_up: conditions_met_buy += 1; confirmations_buy.append("SSL Up")
         if ssl_dir == -1 and price < ssl_down: conditions_met_sell += 1; confirmations_sell.append("SSL Down")

    # 4. MACD Histogram
    macd_h = latest.get('macd_hist')
    if macd_h is not None and not math.isnan(macd_h):
        if macd_h > 0: conditions_met_buy += 1; confirmations_buy.append("MACD Hist > 0")
        if macd_h < 0: conditions_met_sell += 1; confirmations_sell.append("MACD Hist < 0")

    # 5. ADX (Trend Gücü)
    adx = latest.get('adx'); dmi_p = latest.get('dmi_plus'); dmi_n = latest.get('dmi_minus')
    if adx is not None and dmi_p is not None and dmi_n is not None and not math.isnan(adx):
        if adx > adx_threshold:
             # ADX güçlüyse, DMI yönünü kontrol et
             if dmi_p > dmi_n: # Pozitif trend güçlü
                  conditions_met_buy += 1; confirmations_buy.append(f"ADX > {adx_threshold} (DMI+)")
             elif dmi_n > dmi_p: # Negatif trend güçlü
                  conditions_met_sell += 1; confirmations_sell.append(f"ADX > {adx_threshold} (DMI-)")
        # else: ADX düşükse, trend yok sayılır, koşul eklenmez

    # Sinyal Kararı
    signal = "NONE"
    confidence = 0
    confirming_indicators = []
    explanation = "Yeterli onay yok."

    # Tüm koşullar sağlandıysa sinyal ver
    if conditions_met_buy == total_conditions:
        signal = "BUY"
        confidence = 100 # Veya koşul sayısına göre ayarlanabilir
        confirming_indicators = confirmations_buy
        explanation = f"Tüm ({total_conditions}) AL koşulu onaylandı: {', '.join(confirming_indicators)}"
    elif conditions_met_sell == total_conditions:
        signal = "SELL"
        confidence = 100
        confirming_indicators = confirmations_sell
        explanation = f"Tüm ({total_conditions}) SAT koşulu onaylandı: {', '.join(confirming_indicators)}"
    # İsteğe bağlı: Daha az koşulla daha düşük güvenli sinyal
    # elif conditions_met_buy >= 4: signal = "BUY"; confidence = 75; ...
    # elif conditions_met_sell >= 4: signal = "SELL"; confidence = 75; ...


    return {
        "signal": signal,
        "confidence": confidence, # Şimdilik ya 0 ya 100
        "confirming_indicators": confirming_indicators,
        "explanation": explanation
    }


# --------------- GEMINI AI ANALİZ (Aynı kaldı) ---------------
# ... (get_gemini_analysis fonksiyonu aynı kaldı) ...
def get_gemini_analysis(indicators: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    # ...
    return {"signal": "ERROR", ...} # Hata durumu


# --------------- ANA TAHMİN FONKSİYONU (Aynı kaldı) ---------------
# ... (get_ai_prediction fonksiyonu aynı kaldı) ...
def get_ai_prediction(indicators: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    # ...
    return get_heuristic_analysis(indicators)

# --------------- KAYIT FONKSİYONLARI (Aynı kaldı) ---------------
# ... (load_records, save_record, clear_records aynı kaldı) ...
