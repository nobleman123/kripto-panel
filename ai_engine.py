# ai_engine.py
# v4 - İndikatör/Skorlama + Hacim Dönüş + Strateji Kombinasyonu + Specter Trend

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
try: import google.generativeai as genai; GEMINI_AVAILABLE = True
except ImportError: GEMINI_AVAILABLE = False; logging.warning("Gemini kütüphanesi...")

RECORDS_FILE = Path("prediction_records.json")
EMA_TREND_LENGTH = 200
SPECTER_ATR_LENGTH = 200 # Specter için ATR periyodu

# --------------- TEMEL MATEMATİK VE NORMALİZASYON (Aynı kaldı) ---------------
# ... (logistic, normalize) ...
def logistic(x): # ...
    pass
def normalize(v, lo, hi): # ...
    pass

# --------------- İNDİKATÖR HESAPLAMA (Specter için MA/ATR eklendi) ---------------
def nw_smooth(series, bandwidth=8): # ... (Aynı kaldı) ...
    pass

def compute_indicators(df: pd.DataFrame, ma_type: str = 'EMA', ma_length: int = 21) -> pd.DataFrame:
    """Teknik indikatörleri hesaplar (Specter için MA ve ATR eklendi)."""
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy(); required_input_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_input_cols): return pd.DataFrame()

    # MA Hesaplama Fonksiyonu (Seçilebilir Tip İçin)
    def calculate_ma(series, length, ma_type_str):
        try:
            if ma_type_str.upper() == 'SMA': return ta.sma(series, length=length)
            elif ma_type_str.upper() == 'EMA': return ta.ema(series, length=length)
            elif ma_type_str.upper() == 'SMMA': return ta.rma(series, length=length) # SMMA = RMA
            elif ma_type_str.upper() == 'WMA': return ta.wma(series, length=length)
            elif ma_type_str.upper() == 'VWMA': return ta.vwma(series, df['volume'], length=length) # VWMA hacim gerektirir
            else: return ta.ema(series, length=length) # Varsayılan EMA
        except Exception as e:
            logging.debug(f"{ma_type_str}({length}) hesaplama hatası: {e}")
            return pd.Series(np.nan, index=df.index) # Hata durumunda NaN serisi döndür


    # --- Specter Trend için MA'lar ---
    df['specter_ma_short'] = calculate_ma(df['close'], ma_length, ma_type)
    df['specter_ma_long'] = calculate_ma(df['close'], ma_length * 2, ma_type)
    # --- Specter Trend için ATR ---
    try: df[f'atr{SPECTER_ATR_LENGTH}'] = ta.atr(df['high'], df['low'], df['close'], length=SPECTER_ATR_LENGTH)
    except Exception as e: logging.debug(f"ATR{SPECTER_ATR_LENGTH} hatası: {e}"); df[f'atr{SPECTER_ATR_LENGTH}'] = np.nan

    # --- Diğer İndikatörler (Önceki gibi) ---
    # EMA'lar (Genel Skorlama için hala gerekli olabilir)
    for length in [20, 50, EMA_TREND_LENGTH]: df[f'ema{length}'] = calculate_ma(df['close'], length, 'EMA') # EMA olarak sabit
    # MACD, RSI, BBands, Vol SMA/Osc, NW Slope, SuperTrend, SSL, ADX/DMI
    # ... (Bu indikatörlerin hesaplamaları önceki yanıttaki gibi aynı kalacak) ...
    try: macd = ta.macd(df['close']); df['macd_hist'] = macd.iloc[:,1] #...
    except: df['macd_hist'] = np.nan
    try: df['rsi14'] = ta.rsi(df['close'], length=14) #...
    except: df['rsi14'] = np.nan
    try: bb = ta.bbands(df['close']); df['bb_lower']=bb.iloc[:,0]; df['bb_mid']=bb.iloc[:,1]; df['bb_upper']=bb.iloc[:,2] #...
    except: df[['bb_lower','bb_mid','bb_upper']] = np.nan
    # ... (Volume, NW Slope, SuperTrend, SSL, ADX/DMI hesaplamaları) ...


    # Gerekli sütunlar (NaN kontrolü için - Specter dahil)
    required_cols = [
        'close', 'ema20', 'ema50', f'ema{EMA_TREND_LENGTH}', 'macd_hist', 'rsi14',
        'bb_upper', 'bb_lower', 'atr14', 'vol_osc', 'nw_slope', 'vol_sma20',
        'supertrend_direction', 'supertrend_line', 'ssl_direction', 'ssl_up', 'ssl_down',
        'adx', 'dmi_plus', 'dmi_minus',
        'specter_ma_short', 'specter_ma_long', f'atr{SPECTER_ATR_LENGTH}' # Specter eklendi
    ]
    original_len = len(df)
    df = df.dropna(subset=required_cols) # Tüm gerekli indikatörler hesaplanmış olmalı
    if len(df) < 3: logging.warning(f"compute_indicators: Yetersiz veri ({len(df)}/{original_len}).")
    return df


# --------------- ALGORİTMA SKORLAMA (Aynı kaldı) ---------------
# ... (label_from_score, score_signals) ...
def label_from_score(score, thresholds): #...
    pass
def score_signals(latest: pd.Series, prev: pd.Series, funding: Dict[str, float], weights: Dict[str, int]) -> (int, Dict[str, int], list):
    # ... (Puanlama mantığı aynı) ...
    pass

# --------------- SEVİYE HESAPLAMA (Aynı kaldı) ---------------
# ... (compute_trade_levels) ...
def compute_trade_levels(price: float, atr: float, direction: str = 'LONG', **kwargs): #...
    pass

# --------------- HACİM TEYİTLİ DÖNÜŞ MOTORU (Aynı kaldı) ---------------
# ... (analyze_volume_reversal) ...
def analyze_volume_reversal(df_ind: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    # ... (İçerik aynı) ...
    pass

# --------------- STRATEJİ KOMBİNASYON MOTORU (Aynı kaldı) ---------------
# ... (analyze_strategy_combo) ...
def analyze_strategy_combo(latest: pd.Series, **kwargs) -> Dict[str, Any]:
    # ... (İçerik aynı) ...
    pass

# --------------- YENİ: SPECTER TREND ANALİZ MOTORU ---------------
def analyze_specter_trend(df_ind: pd.DataFrame, retest_cooldown: int = 5) -> Dict[str, Any]:
    """Specter Trend Cloud mantığını uygular: Trend + ATR Offset + Retest."""
    result = {"trend": "NEUTRAL", "retest_signal": "NONE", "retest_price": None, "status": "Veri Yok"}
    required_cols_specter = ['close', 'low', 'high', 'specter_ma_short', 'specter_ma_long', f'atr{SPECTER_ATR_LENGTH}']

    if df_ind is None or df_ind.empty or len(df_ind) < max(SPECTER_ATR_LENGTH, 10): # Yeterli ATR ve MA verisi lazım
        result["status"] = "Yetersiz Veri"
        return result
    if not all(col in df_ind.columns and not df_ind[col].isnull().all() for col in required_cols_specter):
        missing = [col for col in required_cols_specter if col not in df_ind.columns or df_ind[col].isnull().all()]
        result["status"] = f"Eksik Sütunlar: {missing}"
        logging.warning(f"analyze_specter_trend eksik sütunlar: {missing}")
        return result

    try:
        # Son kapanan mumun verilerini al
        latest = df_ind.iloc[-2] # Mevcut mumu değil, son kapananı kullan
        if pd.isna(latest['specter_ma_short']) or pd.isna(latest['specter_ma_long']) or pd.isna(latest[f'atr{SPECTER_ATR_LENGTH}']):
             result["status"] = "Hesaplama Bekleniyor"
             return result


        # 1. Trend Belirle
        trend = "BULLISH" if latest['specter_ma_short'] > latest['specter_ma_long'] else "BEARISH"
        result['trend'] = trend

        # 2. ATR ile MA'ları Kaydır (Offset)
        atr_shift = latest[f'atr{SPECTER_ATR_LENGTH}']
        if trend == "BULLISH":
            offset_ma_short = latest['specter_ma_short'] - atr_shift # Aşağı kaydır
            offset_ma_long = latest['specter_ma_long'] - atr_shift
        else: # BEARISH
            offset_ma_short = latest['specter_ma_short'] + atr_shift # Yukarı kaydır
            offset_ma_long = latest['specter_ma_long'] + atr_shift
        result['status'] = f"Trend: {trend}" # Durumu güncelle

        # 3. Retest Ara
        # Son 'retest_cooldown + 2' muma bak (en sondaki hariç)
        retest_window = df_ind.iloc[-(retest_cooldown + 2):-1]
        if len(retest_window) < 2: return result # Yeterli pencere yoksa çık

        current_retest_candle = retest_window.iloc[-1] # Retest için son kapanan muma bak
        previous_retest_candle = retest_window.iloc[-2] # Bir önceki mum (crossover için)

        retest_signal = "NONE"
        retest_price = None

        # Bullish Trend: Fiyat (low) kaydırılmış kısa MA'nın ALTINA inip SONRA ÜSTÜNE çıkarsa (crossover)
        if trend == "BULLISH":
            ma_short_offset_current = current_retest_candle.get(f'specter_ma_short', np.nan) - current_retest_candle.get(f'atr{SPECTER_ATR_LENGTH}', 0)
            ma_short_offset_prev = previous_retest_candle.get(f'specter_ma_short', np.nan) - previous_retest_candle.get(f'atr{SPECTER_ATR_LENGTH}', 0)
            low_current = current_retest_candle.get('low', np.nan)
            low_prev = previous_retest_candle.get('low', np.nan)

            # Basitleştirilmiş retest: Fiyat offset MA'ya değerse (veya altına inerse)
            if not pd.isna(low_current) and not pd.isna(ma_short_offset_current) and low_current <= ma_short_offset_current:
                 # Cooldown kontrolü: Son 'cooldown' mumda başka retest var mı?
                 recent_retests = False
                 for k in range(1, min(retest_cooldown, len(retest_window)-1)):
                      past_candle = retest_window.iloc[-(k+1)]
                      past_ma_offset = past_candle.get(f'specter_ma_short', np.nan) - past_candle.get(f'atr{SPECTER_ATR_LENGTH}', 0)
                      past_low = past_candle.get('low', np.nan)
                      if not pd.isna(past_low) and not pd.isna(past_ma_offset) and past_low <= past_ma_offset:
                           recent_retests = True; break
                 if not recent_retests:
                      retest_signal = "BUY"
                      retest_price = current_retest_candle['close'] # Retest mumunun kapanış fiyatı
                      result['status'] = f"Trend: {trend} - BUY Retest Tespit Edildi!"

        # Bearish Trend: Fiyat (high) kaydırılmış kısa MA'nın ÜSTÜNE çıkıp SONRA ALTINA inerse (crossunder)
        elif trend == "BEARISH":
            ma_short_offset_current = current_retest_candle.get(f'specter_ma_short', np.nan) + current_retest_candle.get(f'atr{SPECTER_ATR_LENGTH}', 0)
            ma_short_offset_prev = previous_retest_candle.get(f'specter_ma_short', np.nan) + previous_retest_candle.get(f'atr{SPECTER_ATR_LENGTH}', 0)
            high_current = current_retest_candle.get('high', np.nan)
            high_prev = previous_retest_candle.get('high', np.nan)

            # Basitleştirilmiş retest: Fiyat offset MA'ya değerse (veya üstüne çıkarsa)
            if not pd.isna(high_current) and not pd.isna(ma_short_offset_current) and high_current >= ma_short_offset_current:
                 # Cooldown kontrolü
                 recent_retests = False
                 for k in range(1, min(retest_cooldown, len(retest_window)-1)):
                      past_candle = retest_window.iloc[-(k+1)]
                      past_ma_offset = past_candle.get(f'specter_ma_short', np.nan) + past_candle.get(f'atr{SPECTER_ATR_LENGTH}', 0)
                      past_high = past_candle.get('high', np.nan)
                      if not pd.isna(past_high) and not pd.isna(past_ma_offset) and past_high >= past_ma_offset:
                           recent_retests = True; break
                 if not recent_retests:
                      retest_signal = "SELL"
                      retest_price = current_retest_candle['close']
                      result['status'] = f"Trend: {trend} - SELL Retest Tespit Edildi!"


        result["retest_signal"] = retest_signal
        result["retest_price"] = f"{retest_price:.5f}" if retest_price is not None else None

        return result

    except Exception as e:
         logging.error(f"analyze_specter_trend hatası: {e}", exc_info=True)
         result["status"] = f"Hesaplama Hatası: {e}"
         return result


# --------------- GEMINI AI ANALİZ (Aynı kaldı) ---------------
# ... (get_gemini_analysis fonksiyonu aynı kaldı) ...
def get_gemini_analysis(indicators: Dict[str, Any], api_key: str) -> Dict[str, Any]: #...
    pass

# --------------- ANA TAHMİN FONKSİYONU (Aynı kaldı) ---------------
# ... (get_ai_prediction fonksiyonu aynı kaldı) ...
def get_ai_prediction(indicators: Dict[str, Any], api_key: str = None) -> Dict[str, Any]: #...
    pass

# --------------- KAYIT FONKSİYONLARI (Aynı kaldı) ---------------
# ... (load_records, save_record, clear_records aynı kaldı) ...
def load_records(): #...
    pass
def save_record(record: Dict[str, Any]): #...
    pass
def clear_records(): #...
    pass
