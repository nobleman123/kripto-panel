# ai_engine.py
# Profesyonel Hibrit Analiz Motoru (Heuristic + Gemini AI)

import math
import json
from pathlib import Path
from typing import Dict, Any
import os
import logging

# Gemini AI kütüphanesini içe aktarmayı dene
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini kütüphanesi yüklü değil. 'pip install google-generativeai' ile yükleyin. Yalnızca heuristic mod kullanılabilir.")

RECORDS_FILE = Path("prediction_records.json")

def logistic(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def normalize(v, lo, hi):
    if v is None: return 0.0
    try: v = float(v)
    except Exception: return 0.0
    if hi == lo: return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

def compute_trade_levels(price: float, atr: float, direction: str = 'LONG', risk_reward_ratio: float = 2.0, atr_multiplier: float = 1.5):
    """Hem LONG hem SHORT yönler için giriş, stop ve hedef hesaplar."""
    price = float(price)
    atr = float(atr) if atr is not None and atr > 0 else price * 0.02 # ATR yoksa %2 varsay
    
    stop_distance = atr * atr_multiplier
    target_distance = stop_distance * risk_reward_ratio
    
    if direction == 'LONG':
        stop_loss = price - stop_distance
        take_profit = price + target_distance
    elif direction == 'SHORT':
        stop_loss = price + stop_distance
        take_profit = price - target_distance
    else: # NEUTRAL
        return {'entry': price, 'stop_loss': None, 'take_profit': None}

    return {
        'entry': price,
        'stop_loss': max(0.0, stop_loss),
        'take_profit': max(0.0, take_profit)
    }

def get_heuristic_analysis(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gelişmiş kural-bazlı (heuristic) analiz motoru.
    İndikatör katkılarını puanlar ve bir sinyal üretir.
    """
    long_score = 0
    short_score = 0
    explanation_points = []
    
    # Ağırlıklar (Bu değerleri main_app'ten de alabilirsiniz, şimdilik burada sabit)
    weights = {
        'rsi_extreme': 25,
        'rsi_trend': 15,
        'macd_cross': 30,
        'macd_trend': 15,
        'nw_slope': 30,
        'vol_spike': 10,
        'score_bonus': 20
    }

    # 1. RSI Analizi
    rsi = indicators.get('rsi14', 50.0)
    if rsi < 30:
        long_score += weights['rsi_extreme']
        explanation_points.append(f"RSI({rsi:.1f}) aşırı satım bölgesinde (+{weights['rsi_extreme']}p LONG).")
    elif rsi < 45:
        long_score += weights['rsi_trend']
        explanation_points.append(f"RSI({rsi:.1f}) düşüş trendinde, ancak satıma yakın (+{weights['rsi_trend']}p LONG).")
    elif rsi > 70:
        short_score += weights['rsi_extreme']
        explanation_points.append(f"RSI({rsi:.1f}) aşırı alım bölgesinde (+{weights['rsi_extreme']}p SHORT).")
    elif rsi > 55:
        short_score += weights['rsi_trend']
        explanation_points.append(f"RSI({rsi:.1f}) yükseliş trendinde, ancak alıma yakın (+{weights['rsi_trend']}p SHORT).")

    # 2. MACD Analizi
    macd_hist = indicators.get('macd_hist', 0.0)
    if macd_hist > 0:
        long_score += weights['macd_trend']
        explanation_points.append(f"MACD Hist({macd_hist:.2f}) pozitif, bullish momentum (+{weights['macd_trend']}p LONG).")
    elif macd_hist < 0:
        short_score += weights['macd_trend']
        explanation_points.append(f"MACD Hist({macd_hist:.2f}) negatif, bearish momentum (+{weights['macd_trend']}p SHORT).")

    # 3. NW Slope (Trend) Analizi
    nw_slope = indicators.get('nw_slope', 0.0)
    if nw_slope > 0:
        long_score += weights['nw_slope']
        explanation_points.append(f"Trend Eğimi({nw_slope:.2f}) pozitif, yükseliş trendi (+{weights['nw_slope']}p LONG).")
    elif nw_slope < 0:
        short_score += weights['nw_slope']
        # --- BURASI DÜZELTİLDİ ---
        explanation_points.append(f"Trend Eğimi({nw_slope:.2f}) negatif, düşüş trendi (+{weights['nw_slope']}p SHORT).")

    # 4. Hacim Analizi
    vol_osc = indicators.get('vol_osc', 0.0)
    if vol_osc > 0.5:
        # Yüksek hacim genelde mevcut trendi doğrular
        if long_score > short_score: long_score += weights['vol_spike']
        if short_score > long_score: short_score += weights['vol_spike']
        explanation_points.append(f"Hacim Osilatörü({vol_osc:.2f}) yüksek, trendi doğruluyor (+{weights['vol_spike']}p).")

    # 5. Ana Skor Bonusu (main_app'ten gelen skor)
    score = indicators.get('score', 0)
    if score > 40:
        long_score += weights['score_bonus']
        explanation_points.append(f"Genel Puan({score}) güçlü AL sinyali veriyor (+{weights['score_bonus']}p LONG).")
    elif score < -40:
        short_score += weights['score_bonus']
        explanation_points.append(f"Genel Puan({score}) güçlü SAT sinyali veriyor (+{weights['score_bonus']}p SHORT).")

    # Sinyal Kararı
    signal = "NEUTRAL"
    confidence = 0
    threshold = 20  # Sinyal üretmek için gereken minimum puan farkı

    if long_score > short_score + threshold:
        signal = "LONG"
        confidence = int(normalize(long_score - short_score, threshold, 100) * 100)
    elif short_score > long_score + threshold:
        signal = "SHORT"
        confidence = int(normalize(short_score - long_score, threshold, 100) * 100)
    else:
        confidence = int(normalize(max(long_score, short_score), 0, threshold) * 50) # Düşük güvenli nötr

    # Ticaret Seviyelerini Hesapla
    levels = compute_trade_levels(
        price=indicators.get('price'),
        atr=indicators.get('atr14'),
        direction=signal
    )
    
    explanation = f"**Heuristic Sinyal: {signal} (Güven: {confidence}%)**\n"
    explanation += f"LONG Puanı: {long_score} | SHORT Puanı: {short_score}\n"
    explanation += "**Nedenler:**\n* " + "\n* ".join(explanation_points)

    return {
        "signal": signal,
        "confidence": confidence,
        "explanation": explanation,
        **levels
    }


def get_gemini_analysis(indicators: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Gemini AI kullanarak gelişmiş teknik analiz yapar.
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini AI kütüphanesi (google-generativeai) yüklü değil.")
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Gemini'ye gönderilecek ham veriler
    data_context = json.dumps(indicators, indent=2)
    
    prompt = f"""
    Sen usta bir kripto para vadeli işlem teknik analistisin.
    Görevin, aşağıdaki JSON formatındaki indikatör anlık görüntüsünü analiz etmek ve profesyonel bir ticaret planı oluşturmaktır.

    İNDİKATÖR VERİLERİ:
    ```json
    {data_context}
    ```

    İNDİKATÖR AÇIKLAMALARI:
    - 'price': Mevcut Fiyat
    - 'score': Diğer indikatörlerden gelen ön-hesaplanmış genel puan (-100 SAT, +100 AL)
    - 'rsi14': RSI (14) değeri (30 altı aşırı satım, 70 üstü aşırı alım)
    - 'macd_hist': MACD Histogram değeri (pozitif = bullish, negatif = bearish momentum)
    - 'vol_osc': Hacim Osilatörü (pozitif değerler ortalamanın üstünde hacim artışı)
    - 'atr14': ATR (14) - Volatilite göstergesi
    - 'nw_slope': Nadaraya-Watson (Trend) Eğimi (pozitif = yükseliş, negatif = düşüş trendi)

    TALEPLER:
    1.  Verileri analiz ederek net bir SİNYAL belirle: "LONG", "SHORT" veya "NEUTRAL".
    2.  Bu sinyale olan GÜVENİNİ 0 ile 100 arasında bir puanla belirt.
    3.  Detaylı bir AÇIKLAMA yaz. Hangi indikatörlerin bu kararı en çok etkilediğini, piyasa duyarlılığının ne olduğunu (momentum, trend, volatilite) ve riskleri açıkla.
    4.  'price' ve 'atr14' değerlerini kullanarak bu sinyal için profesyonel bir GİRİŞ (entry), STOP LOSS (stop_loss) ve HEDEF KÂR (take_profit) seviyesi belirle. (Risk/Ödül oranı yaklaşık 1:2 olmalı).

    CEVAP FORMATI:
    SADECE aşağıdaki yapıya sahip bir JSON nesnesi döndür:
    {{
      "signal": "LONG",
      "confidence": 85,
      "explanation": "Detaylı analiz...",
      "entry": 12345.67,
      "stop_loss": 12300.00,
      "take_profit": 12450.00
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        # Gemini'den gelen yanıtı temizle (bazen ```json ... ``` bloğu içinde dönebiliyor)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        ai_plan = json.loads(cleaned_response)
        
        # Gemini'nin açıklamasını formatla
        ai_plan['explanation'] = f"**Gemini AI Sinyal: {ai_plan['signal']} (Güven: {ai_plan['confidence']}%)**\n{ai_plan['explanation']}"
        
        return ai_plan
        
    except Exception as e:
        # Hata durumunda, Gemini'den gelen tam yanıtı logla (eğer varsa)
        raw_response_text = "Yanıt yok"
        if 'response' in locals() and hasattr(response, 'text'):
            raw_response_text = response.text
        logging.error(f"Gemini API Hatası: {e}\nYanıt: {raw_response_text}")
        raise ConnectionError(f"Gemini API ile iletişim kurulamadı veya yanıt ayrıştırılamadı. Hata: {e}")


def get_ai_prediction(indicators: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    """
    Ana analiz fonksiyonu.
    API anahtarı varsa Gemini'yi, yoksa heuristic motoru kullanır.
    """
    if api_key and GEMINI_AVAILABLE:
        try:
            return get_gemini_analysis(indicators, api_key)
        except Exception as e:
            logging.warning(f"Gemini analizi başarısız oldu, heuristic moda geçiliyor: {e}")
            # Gemini başarısız olursa heuristic'e geri dön
            return get_heuristic_analysis(indicators)
    else:
        # API anahtarı yoksa veya kütüphane yüklü değilse
        return get_heuristic_analysis(indicators)

# --- Kayıt Fonksiyonları (Değişiklik Yok) ---

def load_records():
    if not RECORDS_FILE.exists():
        return []
    try:
        with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_record(record: Dict[str, Any]):
    recs = load_records()
    recs.append(record)
    try:
        with open(RECORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def clear_records():
    try:
        if RECORDS_FILE.exists():
            RECORDS_FILE.unlink()
        return True
    except Exception:
        return False
