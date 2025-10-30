# ai_engine.py
# v6 - 8 İndikatörlü Puanlama + Scalp/Swing Ağırlıklandırma + Gemini Karşılaştırma

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
    "RSI": "Göreceli Güç Endeksi (RSI): Fiyat hareketlerinin hızını ve değişimini ölçen bir momentum osilatörüdür. 30 altı 'aşırı satım', 70 üstü 'aşırı alım' olarak kabul edilir. Scalp işlemlerde yüksek etkilidir.",
    "Stochastic": "Stokastik Osilatör (Stoch): Belirli bir periyottaki kapanış fiyatını, o periyodun fiyat aralığına göre karşılaştırır. RSI'a benzer şekilde momentumu ölçer, 20 altı aşırı satım, 80 üstü aşırı alımdır. Scalp için idealdir.",
    "MACD": "Hareketli Ortalama Yakınsama/Iraksama (MACD): İki hareketli ortalama arasındaki ilişkiyi gösterir. Histogramın sıfır çizgisini kesmesi (crossover) trend değişim sinyali olarak kullanılır. Orta vade için etkilidir.",
    "EMA Cross": "Üssel Hareketli Ortalama Kesişimi (EMA Cross): Kısa vadeli bir EMA'nın (örn: 10) uzun vadeli bir EMA'yı (örn: 30) kesmesi trend yönünü belirler. Kısa > Uzun = AL, Kısa < Uzun = SAT.",
    "Bollinger Bands": "Bollinger Bantları (BB): Fiyatın volatilitesini ölçer. Fiyatın üst banda teması 'aşırı alım', alt banda teması 'aşırı satım' olarak yorumlanabilir. Volatiliteyi gösterir.",
    "ADX": "Ortalama Yönsel Endeks (ADX): Trendin gücünü ölçer (yönünü değil). 25'in üzerindeki bir ADX, güçlü bir trendin varlığına işaret eder. Düşük ADX, piyasanın yatay (chop) olduğunu gösterir.",
    "Volume Spike": "Hacim Artışı: Hacmin, son 20 mumun ortalama hacminden (örn: %100 veya 2 kat) daha fazla olması. Bu, mevcut sinyalin (AL veya SAT) daha güçlü bir teyidi olarak kabul edilir.",
    "ATR": "Ortalama Gerçek Aralık (ATR): Volatiliteyi ölçer. Puanlamada doğrudan kullanılmaz, ancak Stop Loss/Take Profit seviyelerini belirlemek için kritik öneme sahiptir."
}

# --------------- 1. İNDİKATÖR HESAPLAMA (8 Ana İndikatör) ---------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Belirtilen 8 ana indikatörü ve yardımcıları hesaplar."""
    if df is None or df.empty or len(df) < 50: # Yeterli veri gerekli
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
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_mid'] = bb['BBM_20_2.0']
            df['bb_upper'] = bb['BBU_20_2.0']
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
    total_score = 50 # Nötr puan 50'den başlar
    contributions = {} # Puan katkılarını saklamak için

    # --- Ağırlıkları Zaman Dilimine Göre Ayarla ---
    is_scalp = timeframe in scalp_tfs
    
    # Scalp ise RSI ve Stoch ağırlığını artır, MACD'yi azalt
    rsi_w = weights.get('rsi_weight', 25) * 1.5 if is_scalp else weights.get('rsi_weight', 25) * 0.8
    stoch_w = weights.get('stoch_weight', 20) * 1.5 if is_scalp else weights.get('stoch_weight', 20) * 0.8
    macd_w = weights.get('macd_weight', 20) * 0.8 if is_scalp else weights.get('macd_weight', 20) * 1.2
    # Diğerleri sabit
    ema_w = weights.get('ema_cross_weight', 15)
    bb_w = weights.get('bb_weight', 10)
    adx_w = weights.get('adx_weight', 10)
    vol_w = weights.get('volume_weight', 15)

    # 1. RSI
    rsi = latest.get('rsi', 50)
    rsi_score = 0
    if rsi < 30: rsi_score = rsi_w # Aşırı satım
    elif rsi > 70: rsi_score = -rsi_w # Aşırı alım
    elif rsi < 45: rsi_score = rsi_w * 0.5 # Satım bölgesine yakın
    elif rsi > 55: rsi_score = -rsi_w * 0.5 # Alım bölgesine yakın
    total_score += rsi_score; contributions['RSI'] = rsi_score

    # 2. Stochastic
    stoch_k = latest.get('stoch_k', 50); stoch_d = latest.get('stoch_d', 50)
    prev_stoch_k = prev.get('stoch_k', 50); prev_stoch_d = prev.get('stoch_d', 50)
    stoch_score = 0
    if stoch_k < 20 and stoch_d < 20 and stoch_k > prev_stoch_k: # Aşırı satımdan yukarı keserse
        stoch_score = stoch_w
    elif stoch_k > 80 and stoch_d > 80 and stoch_k < prev_stoch_k: # Aşırı alımdan aşağı keserse
        stoch_score = -stoch_w
    total_score += stoch_score; contributions['Stochastic'] = stoch_score

    # 3. MACD
    macd_hist = latest.get('macd_hist', 0); prev_macd_hist = prev.get('macd_hist', 0)
    macd_score = 0
    if macd_hist > 0 and prev_macd_hist < 0: # Bullish Crossover
        macd_score = macd_w
    elif macd_hist < 0 and prev_macd_hist > 0: # Bearish Crossover
        macd_score = -macd_w
    elif macd_hist > 0: # Bullish momentum
        macd_score = macd_w * 0.3
    elif macd_hist < 0: # Bearish momentum
        macd_score = -macd_w * 0.3
    total_score += macd_score; contributions['MACD'] = macd_score

    # 4. EMA Cross
    ema_short = latest.get('ema_short', 0); ema_long = latest.get('ema_long', 0)
    ema_score = 0
    if ema_short > ema_long: ema_score = ema_w * 0.5 # Trend Bullish
    elif ema_short < ema_long: ema_score = -ema_w * 0.5 # Trend Bearish
    total_score += ema_score; contributions['EMA Cross'] = ema_score

    # 5. Bollinger Bands (%B)
    bbp = latest.get('bbp', 0.5) # %B (0-1 arası)
    bb_score = 0
    if bbp < 0.05: bb_score = bb_w # Alt banda değdi/kırdı
    elif bbp > 0.95: bb_score = -bb_w # Üst banda değdi/kırdı
    total_score += bb_score; contributions['Bollinger'] = bb_score
    
    # 6. ADX
    adx = latest.get('adx', 0); dmi_p = latest.get('dmi_plus', 0); dmi_n = latest.get('dmi_minus', 0)
    adx_score = 0
    if adx > 25: # Güçlü trend varsa
        if dmi_p > dmi_n: adx_score = adx_w # Güçlü bullish trend
        else: adx_score = -adx_w # Güçlü bearish trend
    total_score += adx_score; contributions['ADX/DMI'] = adx_score

    # 7. Volume Spike
    vol_spike = latest.get('volume_spike', False)
    vol_score = 0
    if vol_spike:
        # Hacim artışı varsa, mevcut momentumu (MACD/RSI'dan gelen) güçlendir
        if (rsi_score + macd_score) > 0: vol_score = vol_w
        elif (rsi_score + macd_score) < 0: vol_score = -vol_w
    total_score += vol_score; contributions['Hacim Artışı'] = vol_score

    # 8. ATR (Puanlamada kullanılmaz, sadece bilgi)
    contributions['ATR %'] = latest.get('atr_percent', 0) # Bilgi olarak ekle

    # Puanı 0-100 arasına sıkıştır
    final_score = int(max(0, min(100, total_score)))

    # Etiket belirle
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
        
        # AI'a gönderilecek veriyi temizle
        data_context = json.dumps(indicators, indent=2, default=str) # NaN vb. için default=str
        algo_score = indicators.get('algo_score', 50)
        scan_mode = indicators.get('scan_mode', 'Normal')

        prompt = f"""
        Sen bir usta kripto para teknik analistisin. 
        Görevin, aşağıdaki JSON verilerini analiz etmek ve {scan_mode} moduna uygun bir ticaret planı oluşturmaktır.

        İNDİKATÖR VERİLERİ:
        ```json
        {data_context}
        ```

        İNDİKATÖR AÇIKLAMALARI:
        - rsi, stoch_k, macd_hist: Momentum göstergeleri
        - ema_cross_signal: 1 ise Bullish, -1 ise Bearish trend
        - bbp: Bollinger %B (0'a yakın AL, 1'e yakın SAT sinyali olabilir)
        - adx: Trend gücü (25 üzeri güçlü trend)
        - dmi_plus / dmi_minus: Trend yönü
        - volume_spike: True ise sinyali güçlendirir
        - atr_percent: Volatilite (Stop/Target için kullan)
        - algo_score: Benim kendi algoritmamın verdiği puan (0-100 arası).

        TALEPLER:
        1.  **Metinsel Analiz:** Bu verilere dayanarak (özellikle RSI, MACD ve ADX) kısa, 1-2 cümlelik bir piyasa yorumu yap. [{cite: 💬 GEMINI ANALİZİ}]
        2.  **AI Puanı:** Bu verilere göre 0 (Güçlü Sat) ile 100 (Güçlü Al) arasında kendi AI puanını ver. [{cite: 💬 GEMINI ANALİZİ}]
        3.  **Karşılaştırma:** Kendi puanını, benim algoritmamın puanı (`algo_score`: {algo_score}) ile karşılaştır (Uyumlu/Uyumsuz). [{cite: 💬 GEMINI ANALİZİ}]
        4.  **Ticaret Planı:** (Eğer sinyal üretiyorsan) `price` ve `atr_percent` kullanarak `entry`, `stop_loss` ve `take_profit` seviyeleri öner.

        CEVAP FORMATI (SADECE JSON):
        ```json
        {{
          "text": "RSI (45) nötr, ancak MACD (0.12) pozitif momentum gösteriyor. ADX (28) güçlü bir yükseliş trendini onaylıyor...",
          "score": 76,
          "comparison": "Uyumlu",
          "trade_plan": {{
            "signal": "LONG",
            "entry": 45000.0,
            "stop_loss": 44500.0,
            "take_profit": 46000.0
          }}
        }}
        ```
        """
        
        response = model.generate_content(prompt, request_options={'timeout': 120})
        
        # Gemini'den gelen yanıtı temizle ve JSON'a dönüştür
        try:
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                 cleaned_response = response.text[json_start:json_end]
                 ai_plan = json.loads(cleaned_response)
            else:
                 cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
                 ai_plan = json.loads(cleaned_response)
            return ai_plan # Başarılı JSON yanıtı

        except (json.JSONDecodeError, AttributeError) as json_e:
             logging.error(f"Gemini yanıtı JSON ayrıştırma hatası: {json_e}\nYanıt: {response.text}")
             # Hata durumunda bile metni döndürmeye çalış
             return {"text": f"AI yanıtı alınamadı (JSON Hatası): {response.text[:100]}...", "score": 50, "comparison": "Bilinmiyor"}
            
    except Exception as e:
        logging.error(f"Gemini API hatası: {e}", exc_info=True)
        return {"error": str(e)}


# --------------- 4. KAYIT FONKSİYONLARI (Basit) ---------------
def load_tracked_signals():
    """Session state'den takip edilen sinyalleri yükler (şimdilik)."""
    # Kalıcı depolama için burası JSON veya DB okuyacak şekilde değiştirilebilir
    return st.session_state.get('tracked_signals', [])

def save_tracked_signal(signal_data):
    """Sinyali session state'e kaydeder."""
    st.session_state.tracked_signals.append(signal_data)
    return True
