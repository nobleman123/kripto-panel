# ai_engine.py
# Profesyonel Hibrit Analiz Motoru + İndikatör/Skorlama + Hacim Teyitli Dönüş

import math
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
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
    logging.warning("Gemini kütüphanesi (google-generativeai) yüklü değil.")

RECORDS_FILE = Path("prediction_records.json")
EMA_TREND_LENGTH = 200 # Trend filtresi için EMA uzunluğu

# --------------- TEMEL MATEMATİK VE NORMALİZASYON ---------------
# ... (logistic, normalize aynı kaldı) ...
def logistic(x):
    try: return 1.0 / (1.0 + math.exp(-x))
    except OverflowError: return 0.0 if x < 0 else 1.0
def normalize(v, lo, hi):
    if v is None: return 0.0
    try: v = float(v); assert not math.isnan(v) # NaN kontrolü
    except: return 0.0
    if hi == lo: return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

# --------------- İNDİKATÖR HESAPLAMA (EMA Trend Eklendi) ---------------
def nw_smooth(series, bandwidth=8):
    # ... (Aynı kaldı) ...
    y = np.asarray(series); n = len(y);
    if n == 0: return np.array([])
    sm = np.zeros(n)
    for i in range(n): distances = np.arange(n) - i; bw = max(1, bandwidth); weights = np.exp(-0.5 * (distances / bw)**2); sm[i] = np.sum(weights * y) / (np.sum(weights) + 1e-12)
    return sm

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Teknik indikatörleri hesaplar."""
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()
    required_input_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_input_cols): return pd.DataFrame()

    # EMA'lar
    for length in [20, 50, EMA_TREND_LENGTH]:
        try: df[f'ema{length}'] = ta.ema(df['close'], length=length)
        except Exception as e: logging.debug(f"EMA{length} hatası: {e}"); df[f'ema{length}'] = np.nan
    # MACD
    try: macd = ta.macd(df['close']); df['macd_hist'] = macd.iloc[:,1] if isinstance(macd, pd.DataFrame) and macd.shape[1]>=2 else np.nan
    except Exception as e: logging.debug(f"MACD hatası: {e}"); df['macd_hist'] = np.nan
    # RSI
    try: df['rsi14'] = ta.rsi(df['close'], length=14)
    except Exception as e: logging.debug(f"RSI hatası: {e}"); df['rsi14'] = np.nan
    # Bollinger Bands
    try: bb = ta.bbands(df['close']); df['bb_lower'] = bb.iloc[:,0]; df['bb_mid'] = bb.iloc[:,1]; df['bb_upper'] = bb.iloc[:,2]
    except Exception as e: logging.debug(f"BBands hatası: {e}"); df[['bb_lower','bb_mid','bb_upper']] = np.nan
    # ATR
    try: df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    except Exception as e: logging.debug(f"ATR hatası: {e}"); df['atr14'] = np.nan
    # Volume SMA ve Oscillator
    try:
        df['vol_sma20'] = ta.sma(df['volume'], length=20) # Hacim teyidi için SMA
        df['vol_ma_short'] = df['vol_sma20'] # Eski adıyla uyumlu
        df['vol_ma_long'] = ta.sma(df['volume'], length=50)
        df['vol_osc'] = (df['vol_ma_short'] - df['vol_ma_long']) / (df['vol_ma_long'].replace(0, 1e-9) + 1e-9)
    except Exception as e: logging.debug(f"Volume hatası: {e}"); df[['vol_sma20','vol_ma_short','vol_ma_long','vol_osc']] = np.nan
    # NW Slope
    try:
        if len(df['close'].dropna()) > 10:
             sm = nw_smooth(df['close'].values, bandwidth=8); df['nw_smooth'] = sm; df['nw_slope'] = pd.Series(sm).diff().fillna(0)
        else: df['nw_smooth'] = np.nan; df['nw_slope'] = np.nan
    except Exception as e: logging.debug(f"NW Slope hatası: {e}"); df[['nw_smooth','nw_slope']] = np.nan

    # Gerekli sütunlar (NaN kontrolü için)
    required_cols = ['close', 'ema20', 'ema50', f'ema{EMA_TREND_LENGTH}', 'macd_hist', 'rsi14',
                     'bb_upper', 'bb_lower', 'atr14', 'vol_osc', 'nw_slope', 'vol_sma20']
    df_calculated = df.dropna(subset=required_cols)
    if len(df_calculated) < 3: logging.warning(f"compute_indicators: Hesaplama sonrası yetersiz veri ({len(df_calculated)} satır).")

    return df


# --------------- ALGORİTMA SKORLAMA (app.py'den taşındı - Aynı kaldı) ---------------
# ... (label_from_score, score_signals fonksiyonları aynı kaldı) ...
def label_from_score(score, thresholds):
    strong_buy_t, buy_t, sell_t, strong_sell_t = thresholds
    if score is None or not isinstance(score, (int, float)) or math.isnan(score): return "NO DATA"
    if score >= strong_buy_t: return "GÜÇLÜ AL" #... (diğer koşullar aynı)
    return "TUT"
def score_signals(latest: pd.Series, prev: pd.Series, funding: Dict[str, float], weights: Dict[str, int]) -> (int, Dict[str, int], list):
    per = {}; reasons = []; total = 0
    # ... (içerik aynı kaldı, EMA, MACD, RSI, BB, Vol, NW, Funding puanlaması) ...
    return total, per, reasons

# --------------- SEVİYE HESAPLAMA (Aynı kaldı) ---------------
# ... (compute_trade_levels fonksiyonu aynı kaldı) ...
def compute_trade_levels(price: float, atr: float, direction: str = 'LONG', risk_reward_ratio: float = 2.0, atr_multiplier: float = 1.5):
    # ... (içerik aynı) ...
    return {'entry': price, 'stop_loss': max(0.0, stop_loss), 'take_profit': max(0.0, take_profit)}


# --------------- YENİ: HACİM TEYİTLİ DÖNÜŞ MOTORU ---------------
def analyze_volume_reversal(df_ind: pd.DataFrame, look_back: int = 20, confirm_in: int = 5, vol_multiplier: float = 1.5, use_ema_filter: bool = True) -> Dict[str, Any]:
    """
    Yüksek hacimli dönüş formasyonlarını analiz eder.
    'Anchor Candle' + Hacim + Ters Yönlü Kırılım mantığını kullanır.
    """
    if df_ind is None or df_ind.empty or len(df_ind) < look_back + confirm_in + 2:
        return {"signal": "NONE", "score": 0, "status": "Yetersiz Veri"}

    # Son mumu ve bir öncekini al (mevcut mum kapanmadan sinyal vermemek için)
    latest = df_ind.iloc[-2] # En son kapanan mum
    df_lookback = df_ind.iloc[-(look_back + 2):-2] # Anchor'ı arayacağımız pencere (latest hariç)

    anchor_candle = None
    setup_type = None # 'Bullish' veya 'Bearish'

    # 1. Anchor Candle Ara
    highest_in_lookback = df_lookback['high'].max()
    lowest_in_lookback = df_lookback['low'].min()
    avg_volume = df_lookback['vol_sma20'].mean() # Ortalama hacim (SMA kullanarak)
    volume_threshold = avg_volume * vol_multiplier

    # Bearish Anchor (Yeni Yüksek + Yüksek Hacim)
    if latest['high'] > highest_in_lookback and latest['volume'] > volume_threshold:
        anchor_candle = latest
        setup_type = 'Bearish' # Yüksek yaptı, düşüş beklenir
        logging.info(f"VR Engine: Bearish Anchor bulundu: Index={latest.name}, High={latest['high']}, Vol={latest['volume']:.0f} > Threshold={volume_threshold:.0f}")


    # Bullish Anchor (Yeni Düşük + Yüksek Hacim)
    elif latest['low'] < lowest_in_lookback and latest['volume'] > volume_threshold:
        anchor_candle = latest
        setup_type = 'Bullish' # Düşük yaptı, yükseliş beklenir
        logging.info(f"VR Engine: Bullish Anchor bulundu: Index={latest.name}, Low={latest['low']}, Vol={latest['volume']:.0f} > Threshold={volume_threshold:.0f}")

    if anchor_candle is None:
        return {"signal": "NONE", "score": 0, "status": "Anchor Aranıyor"}

    # 2. Kurulum (Setup) Aktif - Onay Bekleniyor
    anchor_high = anchor_candle['high']
    anchor_low = anchor_candle['low']
    setup_box_status = f"{setup_type} Kurulum Aktif ({confirm_in} Mum)"

    # Sonraki mumları al (en sondaki hariç, confirm_in sayısı kadar)
    confirmation_window_df = df_ind.iloc[-confirm_in-1:-1] # Son mumu (henüz kapanmamış olabilir) dahil etme

    confirmed_signal = "NONE"
    confirmation_candle = None
    breakout_bar_index = -1

    # 3. Onay Ara (Ters Yönlü Kırılım)
    for i in range(len(confirmation_window_df)):
        candle = confirmation_window_df.iloc[i]
        remaining_bars = confirm_in - (i + 1)
        setup_box_status = f"{setup_type} Kurulum İzleniyor ({remaining_bars} Mum Kaldı)"

        # Bearish Setup -> Bullish Confirmation (Anchor Low'un altına kırmalı)
        if setup_type == 'Bearish' and candle['low'] < anchor_low:
             confirmed_signal = "SELL"
             confirmation_candle = candle
             breakout_bar_index = i
             logging.info(f"VR Engine: Bearish Setup Onaylandı (SELL): Index={candle.name}, Low={candle['low']} < AnchorLow={anchor_low}")
             break # İlk kırılım yeterli

        # Bullish Setup -> Bearish Confirmation (Anchor High'ın üstüne kırmalı)
        elif setup_type == 'Bullish' and candle['high'] > anchor_high:
             confirmed_signal = "BUY"
             confirmation_candle = candle
             breakout_bar_index = i
             logging.info(f"VR Engine: Bullish Setup Onaylandı (BUY): Index={candle.name}, High={candle['high']} > AnchorHigh={anchor_high}")
             break # İlk kırılım yeterli

    if confirmed_signal == "NONE":
         return {"signal": "NONE", "score": 0, "status": setup_box_status}

    # 4. Sinyal Gücü Skoru Hesapla
    score = 0
    # Faktör 1: Temel formasyon (kırılım) oluştu (Bu aşamaya geldiyse zaten oluşmuştur)
    score += 1
    # Faktör 2: Anchor Candle hacmi yüksekti (Bu aşamaya geldiyse zaten yüksekti)
    score += 1
    # Faktör 3: Onay mumunun hacmi de yüksek mi?
    if confirmation_candle['volume'] > volume_threshold:
        score += 1
        logging.info(f"VR Engine: Onay mumu hacmi yüksek. Score +1.")
    # Faktör 4: Sinyal ana trendle uyumlu mu? (EMA Filtresi)
    ema_trend = latest.get(f'ema{EMA_TREND_LENGTH}', np.nan)
    if use_ema_filter and not math.isnan(ema_trend):
        if confirmed_signal == "BUY" and latest['close'] > ema_trend:
            score += 1
            logging.info(f"VR Engine: BUY sinyali EMA{EMA_TREND_LENGTH} üstünde. Score +1.")
        elif confirmed_signal == "SELL" and latest['close'] < ema_trend:
            score += 1
            logging.info(f"VR Engine: SELL sinyali EMA{EMA_TREND_LENGTH} altında. Score +1.")

    final_status = f"{confirmed_signal} Sinyali ({score}/4)"

    return {
        "signal": confirmed_signal,
        "score": score,
        "status": final_status,
        "anchor_time": anchor_candle.name.strftime('%Y-%m-%d %H:%M'),
        "anchor_price_high": anchor_high,
        "anchor_price_low": anchor_low,
        "confirmation_time": confirmation_candle.name.strftime('%Y-%m-%d %H:%M'),
        "confirmation_price": confirmation_candle['close']
        # Seviyeler (giriş/stop/hedef) ana AI motorundan alınacak
    }


# --------------- GEMINI AI ANALİZ (Prompt güncellendi) ---------------
def get_gemini_analysis(indicators: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    # ... (Fonksiyonun başı aynı kaldı: Kütüphane kontrolü, model tanımı) ...
    if not GEMINI_AVAILABLE: raise ImportError("Gemini AI kütüphanesi...")
    genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-pro')
    symbol = indicators.get('symbol', 'Bilinmeyen'); tf = indicators.get('timeframe', 'Bilinmeyen'); scan_mode = indicators.get('scan_mode', 'Normal')
    safe_indicators = {k: v for k, v in indicators.items() if k not in ['api_key']}
    data_context = json.dumps(safe_indicators, indent=2)

    # --- Prompt v3 (Hacim vurgusu) ---
    prompt = f"""
    Sen, {symbol} ({tf}) paritesi üzerinde uzmanlaşmış, kantitatif bir kripto para analistisin.
    Mevcut analiz modun: **{scan_mode}**.

    Görevin, aşağıdaki JSON verilerini analiz ederek bu moda uygun, mantıksal ve bilimsel bir ticaret planı oluşturmaktır. Genel piyasa duyarlılığını da göz önünde bulundur.

    İNDİKATÖR VERİLERİ ({symbol} - {tf}):
    ```json
    {data_context}
    ```

    İNDİKATÖR AÇIKLAMALARI VE ÖNCELİKLER:
    1.  **`nw_slope` (Nadaraya-Watson Eğimi - ANA TREND):** En önemli. Pozitif = yükseliş, negatif = düşüş. **Trendin tersine işlemden kaçın.**
    2.  **`rsi14` ve `macd_hist` (Momentum):** Trend yönündeki momentumu onayla. Aşırı bölgeler (RSI 70/30) veya MACD gücü önemli.
    3.  **`vol_osc` (Hacim Osilatörü):** Pozitif ve YÜKSEK değerler (>0.5 gibi), trendi veya kırılımı GÜÇLÜ şekilde teyit eder. Düşük hacimli hareketlere daha az güven.
    4.  **`bb_upper` / `bb_lower` (Bollinger):** Volatilite ve potansiyel destek/direnç/hedef. Bant dışı hareketler trendle yorumlanmalı.
    5.  **`atr14` (Volatilite):** SADECE Stop Loss mesafesi için.
    6.  **`funding_rate` (Fonlama):** Aşırı değerler geri çekilme olasılığını artırır.
    7.  **`score` (Ön Skor):** Ek teyit, ana belirleyici DEĞİL.

    ANALİZ MODU TALİMATLARI ({scan_mode} Modu):
    - Strateji: `{ 'Kısa vade...' if scan_mode == 'Scalp' else ('Orta/uzun vade...' if scan_mode == 'Swing' else 'Dengeli...')}`
    - Seviyeler: `{ 'daha dar' if scan_mode == 'Scalp' else ('daha geniş' if scan_mode == 'Swing' else 'standart')}` olmalı.

    TALEPLER:
    1.  Verileri ve modu dikkate alarak net SİNYAL: "LONG", "SHORT" veya "NEUTRAL". **Ana trend (`nw_slope`) yönünde olmalı.**
    2.  GÜVEN (0-100). (Yüksek hacim teyidi güveni ARTIRIR).
    3.  Detaylı AÇIKLAMA: Ana trend, momentum, **hacim teyidi**, Bollinger durumu, fonlama etkisi. Confirmation/Contradiction faktörleri. Riskler.
    4.  Profesyonel GİRİŞ (entry), STOP LOSS (stop_loss), HEDEF KÂR (take_profit). (Stop `atr14`'e, Hedef R:R veya BBand'a göre).

    CEVAP FORMATI (SADECE JSON):
    ```json
    {{...}}
    ```
    """
    # ... (Try/except bloğu ve JSON parse etme aynı kaldı) ...
    try:
        response = model.generate_content(prompt, request_options={'timeout': 120})
        # ... (JSON parse etme, formatlama, seviye doğrulama aynı kaldı) ...
        return ai_plan
    except Exception as e:
        # ... (Hata yönetimi aynı kaldı) ...
        return {"signal": "ERROR", ...}


# --------------- ANA TAHMİN FONKSİYONU (Aynı kaldı) ---------------
def get_ai_prediction(indicators: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    # ... (Gemini veya Heuristic seçimi ve seviye tamamlama aynı kaldı) ...
    if api_key and GEMINI_AVAILABLE:
        try:
            gemini_result = get_gemini_analysis(indicators, api_key)
            # ... (Seviye tamamlama mantığı aynı) ...
            return gemini_result
        except Exception as e:
             logging.warning(f"Gemini genel hatası ({indicators.get('symbol')}), heuristic moda geçiliyor: {e}")
             return get_heuristic_analysis(indicators) # Hata durumunda heuristic dön
    else:
        return get_heuristic_analysis(indicators)


# --------------- KAYIT FONKSİYONLARI (Aynı kaldı) ---------------
# ... (load_records, save_record, clear_records aynı kaldı) ...
