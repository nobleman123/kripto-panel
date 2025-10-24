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
    
    weights = {
        'rsi_extreme': 25,
        'rsi_trend': 15,
        'macd_trend': 15,
        'nw_slope': 30,
        'vol_spike': 10,
        'score_bonus': 20,
        'bb_reversal': 20  # <-- YENİ EKLENDİ
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
        explanation_points.append(f"Trend Eğimi({nw_slope:.2f}) negatif, düşüş trendi (+{weights['nw_slope']}p SHORT).")

    # 4. Hacim Analizi
    vol_osc = indicators.get('vol_osc', 0.0)
    if vol_osc > 0.5:
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

    # 6. Bollinger Bantları Analizi (Reversal) - YENİ EKLENDİ
    price = indicators.get('price', 0.0)
    bb_upper = indicators.get('bb_upper', 0.0)
    bb_lower = indicators.get('bb_lower', 0.0)
    if price > 0 and bb_upper > 0 and price > bb_upper:
        short_score += weights['bb_reversal']
        explanation_points.append(f"Fiyat({price:.2f}) Üst Bollinger Bandını({bb_upper:.2f}) aştı, geri çekilme (SHORT) beklentisi (+{weights['bb_reversal']}p SHORT).")
    elif price > 0 and bb_lower > 0 and price < bb_lower:
        long_score += weights['bb_reversal']
        explanation_points.append(f"Fiyat({price:.2f}) Alt Bollinger Bandını({bb_lower:.2f}) kırdı, tepki (LONG) beklentisi (+{weights['bb_reversal']}p LONG).")


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
    
    # --- YENİ GELİŞTİRİLMİŞ PROMPT ---
    prompt = f"""
    Sen usta bir kantitatif (quantitative) kripto para analistisin.
    Görevin, aşağıdaki JSON formatındaki indikatör anlık görüntüsünü analiz etmek ve bilimsel bir ticaret planı oluşturmaktır.

    İNDİKATÖR VERİLERİ:
    ```json
    {data_context}
    ```

    İNDİKATÖR AÇIKLAMALARI:
    - 'price': Mevcut Fiyat
    - 'score': Ön-hesaplanmış genel puan (-100 SAT, +100 AL)
    - 'rsi14': RSI (14) (Momentum: 30 altı aşırı satım, 70 üstü aşırı alım)
    - 'macd_hist': MACD Histogram (Momentum: pozitif = bullish, negatif = bearish)
    - 'vol_osc': Hacim Osilatörü (Yüksek pozitif = güçlü trend onayı)
    - 'atr14': ATR (14) (Volatilite - SADECE Stop-loss hesaplaması için kullanılır)
    - 'nw_slope': Nadaraya-Watson (ANA TREND - En önemli gösterge. Pozitif = yükseliş, negatif = düşüş)
    - 'bb_upper' / 'bb_lower': Bollinger Bantları (Fiyatın bu bantların dışına çıkması, bir geri çekilme/reversal sinyali olabilir veya hedeftir)

    TALEPLER:
    1.  Ana trendi (`nw_slope`) ve momentumu (`rsi14`, `macd_hist`) analiz ederek net bir SİNYAL belirle: "LONG", "SHORT" veya "NEUTRAL".
    2.  Trendin tersine (counter-trend) işlem açmaktan kaçın. Sinyalin, ana trend (`nw_slope`) yönünde olduğundan emin ol. (Örn: `nw_slope` pozitif ise, sadece "LONG" veya "NEUTRAL" sinyali ver).
    3.  Bu sinyale olan GÜVENİNİ 0 ile 100 arasında bir puanla belirt.
    4.  Detaylı bir AÇIKLAMA yaz. Kararını (Confirmation) ve karşıt görüşleri (Contradictions) belirterek mantıksal bir gerekçe sun.
    5.  `price`, `atr14` ve `bb_upper`/`bb_lower` değerlerini kullanarak profesyonel bir GİRİŞ (entry), STOP LOSS (stop_loss) ve HEDEF KÂR (take_profit) seviyesi belirle. (Stop `atr14`'e, Hedef karşıt Bollinger Bandına veya 1:2 R/R oranına göre belirlenmeli).

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
        raw_response_text = "Yanıt yok"
        if 'response' in locals() and hasattr(response, 'text'):
            raw_response_text = response.text
        logging.error(f"Gemini API Hatası: {e}\nYanıt: {raw_response_text}")
        # Hata durumunda heuristic'e dönmek yerine hatayı göster
        return get_heuristic_analysis(indicators) # Hata olursa heuristic'e dön

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
            return get_heuristic_analysis(indicators)
    else:
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
