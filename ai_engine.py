# ai_engine.py
# Profesyonel Hibrit Analiz Motoru + İndikatör/Skorlama Mantığı

import math
import json
from pathlib import Path
from typing import Dict, Any
import os
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta # İndikatörler için

# Gemini AI kütüphanesi
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini kütüphanesi (google-generativeai) yüklü değil. Yalnızca heuristic mod kullanılabilir.")

RECORDS_FILE = Path("prediction_records.json")

# --------------- TEMEL MATEMATİK VE NORMALİZASYON ---------------
def logistic(x):
    # ... (Aynı kaldı) ...
    try: return 1.0 / (1.0 + math.exp(-x))
    except OverflowError: return 0.0 if x < 0 else 1.0

def normalize(v, lo, hi):
    # ... (Aynı kaldı) ...
    if v is None: return 0.0
    try: v = float(v)
    except Exception: return 0.0
    if hi == lo: return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

# --------------- İNDİKATÖR HESAPLAMA (app.py'den taşındı) ---------------
def nw_smooth(series, bandwidth=8):
    # ... (Aynı kaldı) ...
    y = np.asarray(series)
    n = len(y)
    if n == 0: return np.array([])
    sm = np.zeros(n)
    for i in range(n):
        distances = np.arange(n) - i
        bw = max(1, bandwidth)
        weights = np.exp(-0.5 * (distances / bw)**2)
        sm[i] = np.sum(weights * y) / (np.sum(weights) + 1e-12)
    return sm

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Verilen DataFrame'e teknik indikatörleri hesaplar ve ekler."""
    if df is None or df.empty:
        logging.warning("compute_indicators: Boş DataFrame alındı.")
        return pd.DataFrame()
    
    df = df.copy()
    
    # Gerekli sütunların varlığını kontrol et
    required_input_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_input_cols):
        logging.error(f"compute_indicators: Gerekli sütunlar eksik: {required_input_cols}")
        return pd.DataFrame() # Eksik sütun varsa boş döndür

    # EMA
    try: df['ema20'] = ta.ema(df['close'], length=20)
    except Exception as e: logging.warning(f"EMA20 hesaplama hatası: {e}"); df['ema20'] = np.nan
    try: df['ema50'] = ta.ema(df['close'], length=50)
    except Exception as e: logging.warning(f"EMA50 hesaplama hatası: {e}"); df['ema50'] = np.nan
    try: df['ema200'] = ta.ema(df['close'], length=200)
    except Exception as e: logging.warning(f"EMA200 hesaplama hatası: {e}"); df['ema200'] = np.nan
    
    # MACD
    try:
        macd = ta.macd(df['close'])
        if isinstance(macd, pd.DataFrame) and macd.shape[1]>=2: df['macd_hist'] = macd.iloc[:,1]
        else: df['macd_hist'] = np.nan
    except Exception as e: logging.warning(f"MACD hesaplama hatası: {e}"); df['macd_hist'] = np.nan
    
    # RSI
    try: df['rsi14'] = ta.rsi(df['close'], length=14)
    except Exception as e: logging.warning(f"RSI hesaplama hatası: {e}"); df['rsi14'] = np.nan
    
    # Bollinger Bands
    try:
        bb = ta.bbands(df['close'])
        if isinstance(bb, pd.DataFrame) and bb.shape[1]>=3:
            df['bb_lower'] = bb.iloc[:,0]; df['bb_mid'] = bb.iloc[:,1]; df['bb_upper'] = bb.iloc[:,2]
        else: df[['bb_lower','bb_mid','bb_upper']] = np.nan
    except Exception as e: logging.warning(f"BBands hesaplama hatası: {e}"); df[['bb_lower','bb_mid','bb_upper']] = np.nan
    
    # ADX (Kullanılmıyor ama hesaplanabilir)
    # try:
    #     adx = ta.adx(df['high'], df['low'], df['close'])
    #     df['adx14'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx.columns else np.nan
    # except Exception as e: logging.warning(f"ADX hesaplama hatası: {e}"); df['adx14'] = np.nan
    
    # ATR
    try: df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    except Exception as e: logging.warning(f"ATR hesaplama hatası: {e}"); df['atr14'] = np.nan
    
    # Volume Oscillator
    try:
        df['vol_ma_short'] = ta.sma(df['volume'], length=20)
        df['vol_ma_long'] = ta.sma(df['volume'], length=50)
        # 0'a bölme hatasını önle
        df['vol_osc'] = (df['vol_ma_short'] - df['vol_ma_long']) / (df['vol_ma_long'].replace(0, 1e-9) + 1e-9)
    except Exception as e: logging.warning(f"Volume Oscillator hesaplama hatası: {e}"); df['vol_osc'] = np.nan
    
    # Nadaraya-Watson Slope
    try:
        # Yeterli veri varsa hesapla
        if len(df['close'].dropna()) > 10: # En az 10 veri noktası olsun
             sm = nw_smooth(df['close'].values, bandwidth=8)
             if len(sm) == len(df):
                 df['nw_smooth'] = sm
                 df['nw_slope'] = pd.Series(sm).diff().fillna(0)
             else: df['nw_smooth'] = np.nan; df['nw_slope'] = np.nan
        else: df['nw_smooth'] = np.nan; df['nw_slope'] = np.nan
    except Exception as e: logging.warning(f"NW Slope hesaplama hatası: {e}"); df['nw_smooth'] = np.nan; df['nw_slope'] = np.nan
    
    # Hesaplama sonrası NaN kontrolü
    required_cols = ['close', 'ema20', 'ema50', 'ema200', 'macd_hist', 'rsi14', 'bb_upper', 'bb_lower', 'atr14', 'vol_osc', 'nw_slope']
    # Sadece hesaplanan sütunlardaki NaN değerleri kontrol edelim
    df_calculated = df.dropna(subset=required_cols)
    if len(df_calculated) < 3: # En az 3 satır kalmalı
         logging.warning(f"compute_indicators: Hesaplama sonrası yetersiz veri ({len(df_calculated)} satır).")
         # return pd.DataFrame() # Boş döndürmek yerine NaN içeren df'i döndür ki en azından ham veri kalsın
    
    return df


# --------------- ALGORİTMA SKORLAMA (app.py'den taşındı) ---------------
def label_from_score(score, thresholds):
    # ... (Aynı kaldı) ...
    strong_buy_t, buy_t, sell_t, strong_sell_t = thresholds
    if score is None or not isinstance(score, (int, float)) or math.isnan(score): return "NO DATA"
    if score >= strong_buy_t: return "GÜÇLÜ AL"
    if score >= buy_t: return "AL"
    if score <= strong_sell_t: return "GÜÇLÜ SAT"
    if score <= sell_t: return "SAT"
    return "TUT"

def score_signals(latest: pd.Series, prev: pd.Series, funding: Dict[str, float], weights: Dict[str, int]) -> (int, Dict[str, int], list):
    """Verilen son iki veri noktası ve ağırlıklara göre puanlama yapar."""
    per = {}; reasons = []; total = 0
    
    # Veri kontrolleri
    if latest is None or prev is None or not isinstance(latest, pd.Series) or not isinstance(prev, pd.Series):
         logging.warning("score_signals: Geçersiz 'latest' veya 'prev' verisi.")
         return 0, {}, ["Geçersiz giriş verisi"]

    def get_safe_float(series, key, default=0.0):
        val = series.get(key)
        if val is None or math.isnan(val): return default
        try: return float(val)
        except (ValueError, TypeError): return default

    # EMA
    try:
        w = weights.get('ema', 0)
        ema20 = get_safe_float(latest, 'ema20')
        ema50 = get_safe_float(latest, 'ema50')
        ema200 = get_safe_float(latest, 'ema200')
        contrib = 0
        if ema20 > ema50 > ema200: contrib = +w; reasons.append("EMA bullish")
        elif ema20 < ema50 < ema200: contrib = -w; reasons.append("EMA bearish")
        per['ema'] = contrib; total += contrib
    except Exception as e: logging.debug(f"EMA Skorlama Hatası: {e}"); per['ema']=0
    # MACD
    try:
        w = weights.get('macd', 0)
        p_h = get_safe_float(prev, 'macd_hist'); l_h = get_safe_float(latest, 'macd_hist')
        contrib = 0
        if p_h < 0 and l_h > 0: contrib = w; reasons.append("MACD cross bullish")
        elif p_h > 0 and l_h < 0: contrib = -w; reasons.append("MACD cross bearish")
        per['macd'] = contrib; total += contrib
    except Exception as e: logging.debug(f"MACD Skorlama Hatası: {e}"); per['macd']=0
    # RSI
    try:
        w = weights.get('rsi', 0); rsi = get_safe_float(latest, 'rsi14', 50)
        contrib = 0
        if rsi < 30: contrib = w; reasons.append("RSI oversold")
        elif rsi > 70: contrib = -w; reasons.append("RSI overbought")
        per['rsi'] = contrib; total += contrib
    except Exception as e: logging.debug(f"RSI Skorlama Hatası: {e}"); per['rsi']=0
    # Bollinger Bands
    try:
        w = weights.get('bb', 0)
        price = get_safe_float(latest, 'close')
        bb_upper = get_safe_float(latest, 'bb_upper', float('inf'))
        bb_lower = get_safe_float(latest, 'bb_lower', 0)
        contrib = 0
        if price > bb_upper: contrib = -w; reasons.append("Above BB upper (Short)")
        elif price < bb_lower: contrib = w; reasons.append("Below BB lower (Long)")
        per['bb'] = contrib; total += contrib
    except Exception as e: logging.debug(f"BB Skorlama Hatası: {e}"); per['bb']=0
    # Volume Oscillator
    try:
        w = weights.get('vol', 0); vol_osc = get_safe_float(latest, 'vol_osc')
        contrib = 0
        if vol_osc > 0.4: contrib = w; reasons.append("Volume spike")
        per['vol'] = contrib; total += contrib
    except Exception as e: logging.debug(f"Vol Skorlama Hatası: {e}"); per['vol']=0
    # NW Slope
    try:
        w = weights.get('nw', 0); nw_s = get_safe_float(latest, 'nw_slope')
        contrib = 0
        if nw_s > 0: contrib = w; reasons.append("NW slope +")
        elif nw_s < 0: contrib = -w; reasons.append("NW slope -")
        per['nw'] = contrib; total += contrib
    except Exception as e: logging.debug(f"NW Skorlama Hatası: {e}"); per['nw']=0
    # Funding Rate
    try:
        w = weights.get('funding', 0); fr = funding.get('fundingRate', 0.0)
        contrib = 0
        if fr > 0.0006: contrib = -w; reasons.append("Funding High (Short)")
        elif fr < -0.0006: contrib = w; reasons.append("Funding Low (Long)")
        per['funding'] = contrib; total += contrib
    except Exception as e: logging.debug(f"Funding Skorlama Hatası: {e}"); per['funding']=0
    
    total = int(max(min(total, 100), -100))
    return total, per, reasons


# --------------- SEVİYE HESAPLAMA ---------------
def compute_trade_levels(price: float, atr: float, direction: str = 'LONG', risk_reward_ratio: float = 2.0, atr_multiplier: float = 1.5):
    # ... (Aynı kaldı) ...
    if not isinstance(price, (int, float)) or price <= 0: return {'entry': None, 'stop_loss': None, 'take_profit': None}
    atr = float(atr) if atr is not None and isinstance(atr, (int, float)) and atr > 0 else price * 0.02
    stop_distance = atr * atr_multiplier
    target_distance = stop_distance * risk_reward_ratio
    if direction == 'LONG': stop_loss = price - stop_distance; take_profit = price + target_distance
    elif direction == 'SHORT': stop_loss = price + stop_distance; take_profit = price - target_distance
    else: return {'entry': price, 'stop_loss': None, 'take_profit': None}
    return {'entry': price, 'stop_loss': max(0.0, stop_loss), 'take_profit': max(0.0, take_profit)}

# --------------- HEURISTIC ANALİZ ---------------
def get_heuristic_analysis(indicators: Dict[str, Any]) -> Dict[str, Any]:
    # ... (Aynı kaldı - app.py'den gelen 'score'u kullanıyor) ...
    score = indicators.get('score', 0)
    signal = "NEUTRAL"; confidence = 0
    if score > 50: signal = "LONG"; confidence = int(normalize(score, 50, 100)*50 + 50)
    elif score > 20: signal = "LONG"; confidence = int(normalize(score, 20, 50)*50)
    elif score < -50: signal = "SHORT"; confidence = int(normalize(abs(score), 50, 100)*50 + 50)
    elif score < -20: signal = "SHORT"; confidence = int(normalize(abs(score), 20, 50)*50)
    else: confidence = int(normalize(abs(score), 0, 20)*30)
    levels = compute_trade_levels(price=indicators.get('price'), atr=indicators.get('atr14'), direction=signal)
    explanation = f"**Algoritma Sinyali: {signal} (Skor: {score}, Güven Yaklaşık: {confidence}%)**\n"
    explanation += f"Bu sinyal, `score_signals` fonksiyonunda tanımlanan ağırlıklara göre hesaplanan `{score}` puanına dayanmaktadır."
    return {"signal": signal, "confidence": confidence, "explanation": explanation, **levels}

# --------------- GEMINI AI ANALİZ ---------------
def get_gemini_analysis(indicators: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    # ... (Aynı kaldı - Gelişmiş Prompt v2) ...
    if not GEMINI_AVAILABLE: raise ImportError("Gemini AI kütüphanesi...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    symbol = indicators.get('symbol', 'Bilinmeyen'); tf = indicators.get('timeframe', 'Bilinmeyen'); scan_mode = indicators.get('scan_mode', 'Normal')
    safe_indicators = {k: v for k, v in indicators.items() if k not in ['api_key']}
    data_context = json.dumps(safe_indicators, indent=2)
    prompt = f"""
    Sen, {symbol} ({tf}) paritesi üzerinde uzmanlaşmış, kantitatif bir kripto para analistisin...
    {data_context}
    ... (Prompt'un geri kalanı önceki cevapta olduğu gibi aynı) ...
    CEVAP FORMATI (SADECE JSON):
    ```json
    {{...}}
    ```
    """
    try:
        response = model.generate_content(prompt, request_options={'timeout': 120})
        logging.info(f"Gemini analizi başarılı: {symbol} - {tf}")
        try:
            json_start = response.text.find('{'); json_end = response.text.rfind('}') + 1
            if json_start != -1 and json_end != -1: cleaned_response = response.text[json_start:json_end]
            else: cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            ai_plan = json.loads(cleaned_response)
        except (json.JSONDecodeError, AttributeError) as json_e:
             logging.error(f"Gemini JSON ayrıştırma hatası ({symbol}, {tf}): {json_e}\nYanıt: {response.text}")
             raise ConnectionError(f"Gemini yanıtı JSON formatında değil: {json_e}")

        ai_plan['explanation'] = f"**Gemini AI ({scan_mode}) Sinyal: {ai_plan.get('signal','N/A')} (Güven: {ai_plan.get('confidence','N/A')}%)**\n{ai_plan.get('explanation','Açıklama alınamadı.')}"
        for key in ['entry', 'stop_loss', 'take_profit']:
            if key in ai_plan and ai_plan[key] is not None:
                try: ai_plan[key] = float(ai_plan[key])
                except (ValueError, TypeError): ai_plan[key] = None
            else: ai_plan[key] = None
        return ai_plan
    except Exception as e:
        raw_response_text = "Yanıt yok"
        if 'response' in locals() and hasattr(response, 'text'): raw_response_text = response.text
        logging.error(f"Gemini API/İşlem Hatası ({symbol}, {tf}): {e}\nYanıt: {raw_response_text}")
        return {"signal": "ERROR", "confidence": 0, "explanation": f"**Gemini AI Hatası:** {e}\nRaw Response:\n{raw_response_text[:500]}...",
                "entry": indicators.get('price'), "stop_loss": None, "take_profit": None}


# --------------- ANA TAHMİN FONKSİYONU ---------------
def get_ai_prediction(indicators: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    # ... (Aynı kaldı - Gemini veya Heuristic seçimi ve seviye tamamlama) ...
    if api_key and GEMINI_AVAILABLE:
        try:
            gemini_result = get_gemini_analysis(indicators, api_key)
            if gemini_result.get("signal") != "ERROR" and \
               (gemini_result.get("stop_loss") is None or gemini_result.get("take_profit") is None):
                logging.info(f"Gemini seviyeleri eksik/hesaplanamadı ({indicators.get('symbol')}, {indicators.get('timeframe')}), heuristic seviyeler kullanılıyor.")
                heuristic_levels = compute_trade_levels(price=indicators.get('price'), atr=indicators.get('atr14'), direction=gemini_result.get("signal", "NEUTRAL"))
                if gemini_result.get("stop_loss") is None: gemini_result["stop_loss"] = heuristic_levels.get("stop_loss")
                if gemini_result.get("take_profit") is None: gemini_result["take_profit"] = heuristic_levels.get("take_profit")
                gemini_result["entry"] = heuristic_levels.get("entry") # Girişi her zaman anlık fiyat olarak ayarla
            return gemini_result
        except Exception as e:
            logging.warning(f"Gemini genel hatası ({indicators.get('symbol')}), heuristic moda geçiliyor: {e}")
            return get_heuristic_analysis(indicators)
    else:
        return get_heuristic_analysis(indicators)

# --------------- KAYIT FONKSİYONLARI (Aynı kaldı) ---------------
# ... (load_records, save_record, clear_records) ...
def load_records():
    if not RECORDS_FILE.exists(): return []
    try:
        with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
            content = f.read();
            if not content: return []
            return json.loads(content)
    except Exception as e: logging.error(f"Kayıtlar yüklenemedi ({RECORDS_FILE}): {e}"); return []
def save_record(record: Dict[str, Any]):
    recs = load_records(); recs.append(record)
    try:
        with open(RECORDS_FILE, 'w', encoding='utf-8') as f: json.dump(recs, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e: logging.error(f"Kayıt kaydedilemedi ({RECORDS_FILE}): {e}"); return False
def clear_records():
    try:
        if RECORDS_FILE.exists(): RECORDS_FILE.unlink(); logging.info("Kayıt dosyası silindi.")
        return True
    except Exception as e: logging.error(f"Kayıtlar silinemedi ({RECORDS_FILE}): {e}"); return False
