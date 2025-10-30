# ai_engine.py
# v6.1 - NameError Düzeltmesi (Import eklendi)

import streamlit as st                # <-- HATA DÜZELTMESİ: Eklendi
from datetime import timedelta      # <-- HATA DÜZELTMESİ: Eklendi
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
        # ... (Tüm indikatör hesaplamaları aynı kaldı) ...
        # 1. RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        # 2. MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if isinstance(macd, pd.DataFrame):
            df['macd_hist'] = macd['MACDh_12_26_9']
            df['macd_line'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
        # 3. EMA Cross
        df['ema_short'] = ta.ema(df['close'], length=10)
        df['ema_long'] = ta.ema(df['close'], length=30)
        # 4. Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if isinstance(bb, pd.DataFrame):
            df['bbp'] = bb['BBP_20_2.0'] # %B
        # 5. Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
        if isinstance(stoch, pd.DataFrame):
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
        # 6. ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if isinstance(adx, pd.DataFrame):
            df['adx'] = adx['ADX_14']
            df['dmi_plus'] = adx['DMP_14']
            df['dmi_minus'] = adx['DMN_14']
        # 7. Volume Spike
        df['vol_sma'] = ta.sma(df['volume'], length=20)
        df['volume_spike'] = (df['volume'] > (df['vol_sma'] * 2))
        # 8. ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_percent'] = (df['atr'] / df['close']) * 100

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
    # ... (Tüm puanlama mantığı aynı kaldı) ...
    total_score = 50; contributions = {}
    is_scalp = timeframe in scalp_tfs
    
    # Ağırlık ayarı
    rsi_w = weights.get('rsi_weight', 25) * 1.5 if is_scalp else weights.get('rsi_weight', 25) * 0.8
    stoch_w = weights.get('stoch_weight', 20) * 1.5 if is_scalp else weights.get('stoch_weight', 20) * 0.8
    macd_w = weights.get('macd_weight', 20) * 0.8 if is_scalp else weights.get('macd_weight', 20) * 1.2
    ema_w = weights.get('ema_cross_weight', 15)
    bb_w = weights.get('bb_weight', 10)
    adx_w = weights.get('adx_weight', 10)
    vol_w = weights.get('volume_weight', 15)

    # Puanlama
    # 1. RSI
    rsi = latest.get('rsi', 50); rsi_score = 0
    if rsi < 30: rsi_score = rsi_w; elif rsi > 70: rsi_score = -rsi_w
    elif rsi < 45: rsi_score = rsi_w * 0.5; elif rsi > 55: rsi_score = -rsi_w * 0.5
    total_score += rsi_score; contributions['RSI'] = rsi_score
    # 2. Stochastic
    stoch_k = latest.get('stoch_k', 50); prev_stoch_k = prev.get('stoch_k', 50)
    stoch_d = latest.get('stoch_d', 50); prev_stoch_d = prev.get('stoch_d', 50)
    stoch_score = 0
    if stoch_k < 20 and stoch_d < 20 and stoch_k > prev_stoch_k: stoch_score = stoch_w
    elif stoch_k > 80 and stoch_d > 80 and stoch_k < prev_stoch_k: stoch_score = -stoch_w
    total_score += stoch_score; contributions['Stochastic'] = stoch_score
    # 3. MACD
    macd_hist = latest.get('macd_hist', 0); prev_macd_hist = prev.get('macd_hist', 0)
    macd_score = 0
    if macd_hist > 0 and prev_macd_hist < 0: macd_score = macd_w
    elif macd_hist < 0 and prev_macd_hist > 0: macd_score = -macd_w
    elif macd_hist > 0: macd_score = macd_w * 0.3
    elif macd_hist < 0: macd_score = -macd_w * 0.3
    total_score += macd_score; contributions['MACD'] = macd_score
    # 4. EMA Cross
    ema_short = latest.get('ema_short', 0); ema_long = latest.get('ema_long', 0)
    ema_score = 0
    if ema_short > ema_long: ema_score = ema_w * 0.5
    elif ema_short < ema_long: ema_score = -ema_w * 0.5
    total_score += ema_score; contributions['EMA Cross'] = ema_score
    # 5. Bollinger Bands (%B)
    bbp = latest.get('bbp', 0.5); bb_score = 0
    if bbp < 0.05: bb_score = bb_w
    elif bbp > 0.95: bb_score = -bb_w
    total_score += bb_score; contributions['Bollinger'] = bb_score
    # 6. ADX
    adx = latest.get('adx', 0); dmi_p = latest.get('dmi_plus', 0); dmi_n = latest.get('dmi_minus', 0)
    adx_score = 0
    if adx > 25:
        if dmi_p > dmi_n: adx_score = adx_w
        else: adx_score = -adx_w
    total_score += adx_score; contributions['ADX/DMI'] = adx_score
    # 7. Volume Spike
    vol_spike = latest.get('volume_spike', False); vol_score = 0
    if vol_spike:
        if (rsi_score + macd_score) > 0: vol_score = vol_w
        elif (rsi_score + macd_score) < 0: vol_score = -vol_w
    total_score += vol_score; contributions['Hacim Artışı'] = vol_score
    # 8. ATR
    contributions['ATR %'] = latest.get('atr_percent', 0)

    final_score = int(max(0, min(100, total_score)))
    # Etiket
    label = "TUT"
    if final_score > 85: label = "GÜÇLÜ AL"
    elif final_score > 60: label = "AL"
    elif final_score < 15: label = "GÜÇLÜ SAT"
    elif final_score < 40: label = "SAT"

    return final_score, label, contributions


# --------------- 3. GEMINI ANALİZİ (Karşılaştırmalı) ---------------
@st.cache_data(ttl=timedelta(minutes=10)) # Gemini yanıtlarını 10dk cache'le
def get_gemini_analysis(indicators: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Gemini AI'dan analiz ve karşılaştırmalı puan alır."""
    
    if not GEMINI_AVAILABLE:
        return {"error": "Gemini kütüphanesi (google-generativeai) yüklü değil."}
    if not api_key:
        return {"error": "API anahtarı girilmedi."}
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        data_context = json.dumps(indicators, indent=2, default=str)
        algo_score = indicators.get('algo_score', 50)
        scan_mode = indicators.get('scan_mode', 'Normal')

        prompt = f"""
        Sen bir usta kripto para teknik analistisin...
        Görevin, aşağıdaki JSON verilerini analiz etmek ve {scan_mode} moduna uygun bir ticaret planı oluşturmaktır...
        İNDİKATÖR VERİLERİ: {data_context}
        ...
        CEVAP FORMATI (SADECE JSON):
        ```json
        {{
          "text": "...",
          "score": 76,
          "comparison": "Uyumlu",
          "trade_plan": {{...}}
        }}
        ```
        """
        
        response = model.generate_content(prompt, request_options={'timeout': 120})
        
        try:
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                 cleaned_response = response.text[json_start:json_end]
                 ai_plan = json.loads(cleaned_response)
            else:
                 cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
                 ai_plan = json.loads(cleaned_response)
            return ai_plan

        except (json.JSONDecodeError, AttributeError) as json_e:
             logging.error(f"Gemini yanıtı JSON ayrıştırma hatası: {json_e}\nYanıt: {response.text}")
             return {"text": f"AI yanıtı alınamadı (JSON Hatası): {response.text[:100]}...", "score": 50, "comparison": "Bilinmiyor"}
            
    except Exception as e:
        logging.error(f"Gemini API hatası: {e}", exc_info=True)
        return {"error": str(e)}


# --------------- 4. KAYIT FONKSİYONLARI (Basit) ---------------
def load_tracked_signals():
    """Session state'den takip edilen sinyalleri yükler (şimdilik)."""
    return st.session_state.get('tracked_signals', [])

def save_tracked_signal(signal_data):
    """Sinyali session state'e kaydeder."""
    st.session_state.tracked_signals.append(signal_data)
    return True
