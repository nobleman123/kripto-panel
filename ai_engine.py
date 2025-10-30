# ai_engine.py
# v6.2 - Circular Import Hatası Düzeltmesi

import streamlit as st
from datetime import timedelta
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import json
import math
from typing import Dict, Any, Tuple, List

# Gemini AI kütüphanesi
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini kütüphanesi (google-generativeai) yüklü değil.")

# --- SABİTLER ---
# --- HATA BURADAYDI ---
# EMA_TREND_LENGTH = ai_engine.EMA_TREND_LENGTH 
# --- DÜZELTİLMİŞ HALİ ---
EMA_TREND_LENGTH = 200 # Sabit değeri doğrudan burada tanımla
# --- DÜZELTME SONU ---

# --- İndikatör Açıklamaları (Tooltip için) ---
INDICATOR_DESCRIPTIONS = {
    "RSI": "...",
    # ... (Diğer açıklamalar aynı) ...
}

# --------------- 1. İNDİKATÖR HESAPLAMA (8 Ana İndikatör) ---------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Belirtilen 8 ana indikatörü ve yardımcıları hesaplar."""
    if df is None or df.empty or len(df) < 50:
        logging.warning("compute_indicators: Yetersiz veri.")
        return pd.DataFrame()
    
    df = df.copy()
    try:
        # 1. RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        # 2. MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if isinstance(macd, pd.DataFrame):
            df['macd_hist'] = macd['MACDh_12_26_9'] # Histogram
            df['macd_line'] = macd['MACD_12_26_9'] # MACD Çizgisi
            df['macd_signal'] = macd['MACDs_12_26_9'] # Sinyal Çizgisi

        # 3. EMA Cross
        df['ema_short'] = ta.ema(df['close'], length=10)
        df['ema_long'] = ta.ema(df['close'], length=30)
        
        # 4. Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if isinstance(bb, pd.DataFrame):
            df['bbp'] = bb['BBP_20_2.0'] # %B (Puanlama için daha kolay)

        # 5. Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if isinstance(stoch, pd.DataFrame):
            df['stoch_k'] = stoch['STOCHk_14_3_3'] # %K
            df['stoch_d'] = stoch['STOCHd_14_3_3'] # %D

        # 6. ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if isinstance(adx, pd.DataFrame):
            df['adx'] = adx['ADX_14']
            df['dmi_plus'] = adx['DMP_14']
            df['dmi_minus'] = adx['DMN_14']
            
        # 7. Volume Spike
        df['vol_sma'] = ta.sma(df['volume'], length=20)
        # Hacim anlık > ortalamanın 2 katıysa 1, değilse 0
        df['volume_spike'] = (df['volume'] > (df['vol_sma'] * 2)) 

        # 8. ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        # ATR'yi fiyatın yüzdesi olarak hesapla (normalize etmek için)
        df['atr_percent'] = (df['atr'] / df['close']) * 100

        # Gerekli sütunlarda NaN olanları kaldır
        required_cols = ['rsi', 'macd_hist', 'ema_short', 'ema_long', 'bbp', 'stoch_k', 'stoch_d', 'adx', 'volume_spike', 'atr_percent']
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        
        if len(df) < 2:
             logging.warning("compute_indicators: İndikatör hesaplaması sonrası veri kalmadı.")
             return pd.DataFrame()

        return df
    
    except Exception as e:
        logging.error(f"compute_indicators hatası: {e}", exc_info=True)
        return pd.DataFrame()


# --------------- 2. PUANLAMA SİSTEMİ (8 İndikatör) ---------------
def score_signals(latest: pd.Series, prev: pd.Series, weights: Dict[str, int], timeframe: str, scalp_tfs: List[str]) -> Tuple[int, str, Dict[str, float]]:
    """
    Belirtilen 8 indikatöre ve zaman dilimine (Scalp/Swing) göre puanlama yapar.
    """
    # ... (Önceki yanıttaki 8 indikatörlü puanlama mantığının aynısı) ...
    total_score = 50; contributions = {}
    is_scalp = timeframe in scalp_tfs
    # Ağırlık ayarı...
    # Puanlama (RSI, Stoch, MACD, EMA, BB, ADX, Volume)...
    final_score = int(max(0, min(100, total_score)))
    # Etiket...
    label = "TUT" #...
    return final_score, label, contributions


# --------------- 3. GEMINI ANALİZİ (Karşılaştırmalı) ---------------
@st.cache_data(ttl=timedelta(minutes=10))
def get_gemini_analysis(indicators: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Gemini AI'dan analiz ve karşılaştırmalı puan alır."""
    # ... (Önceki yanıttaki Gemini prompt ve parse etme mantığının aynısı) ...
    if not GEMINI_AVAILABLE: return {"error": "Gemini kütüphanesi yüklü değil."}
    # ...
    try:
        # ... (API çağrısı) ...
        # ... (JSON parse etme) ...
        return ai_plan
    except Exception as e:
        return {"error": str(e)}


# --------------- 4. KAYIT FONKSİYONLARI (Basit) ---------------
def load_tracked_signals():
    """Session state'den takip edilen sinyalleri yükler (şimdikilik)."""
    return st.session_state.get('tracked_signals', [])

def save_tracked_signal(signal_data):
    """Sinyali session state'e kaydeder."""
    st.session_state.tracked_signals.append(signal_data)
    return True
