# ai_engine.py
# AI + Heuristic değerlendirme - Strateji ayrımlı ve Gemini destekli

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini (google.generativeai) yüklü değil; heuristic fallback kullanılacak.")

RECORDS_FILE = Path("ai_trading_records.json")

class StrategyProfile:
    """
    Varsayılan strateji profilleri:
    - SCALP: çok kısa timeframeler, agresif risk
    - SWING: orta vadeli
    - LONG: günlük/haftalık, daha büyük hedefler
    """
    PROFILES = {
        "SCALP": {
            "timeframes": ["1m","3m","5m","15m"],
            "ma_base": 8,
            "atr_mult": 1.0,
            "risk_fraction": 0.04,   # %4
            "target_rr": 1.8
        },
        "SWING": {
            "timeframes": ["30m","1h","4h"],
            "ma_base": 20,
            "atr_mult": 1.3,
            "risk_fraction": 0.025,  # %2.5
            "target_rr": 2.5
        },
        "LONG": {
            "timeframes": ["1d","1w"],
            "ma_base": 50,
            "atr_mult": 1.8,
            "risk_fraction": 0.02,   # %2
            "target_rr": 3.5
        }
    }

def calculate_trade_levels(price: float, atr: float, direction: str, profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Strateji profiline göre entry/stop/tp oluşturur.
    direction: 'LONG' veya 'SHORT'
    """
    if price <= 0 or atr <= 0:
        return {'entry': price, 'stop_loss': 0.0, 'take_profit': 0.0, 'rr': 0.0}
    atr_mult = profile.get('atr_mult', 1.0)
    stop_distance = max(atr * atr_mult, price * 0.001)  # minimum stop distance
    target_rr = profile.get('target_rr', 2.0)
    if direction == "LONG":
        entry = price
        stop = price - stop_distance
        tp = price + stop_distance * target_rr
    elif direction == "SHORT":
        entry = price
        stop = price + stop_distance
        tp = price - stop_distance * target_rr
    else:
        return {'entry': price, 'stop_loss': 0.0, 'take_profit': 0.0, 'rr': 0.0}
    rr = abs((tp - entry) / (entry - stop)) if (entry - stop) != 0 else 0
    return {'entry': entry, 'stop_loss': stop, 'take_profit': tp, 'rr': rr}

def get_heuristic_signal(indicators: Dict[str, Any], specter: Dict[str, Any], strategy: str = "SWING") -> Dict[str, Any]:
    """
    Heuristic kural motoru - strategy'ye göre farklı ağırlıklar kullanır.
    Döndürür: signal, confidence, explanation, levels
    """
    profile = StrategyProfile.PROFILES.get(strategy, StrategyProfile.PROFILES['SWING'])
    long_score = 0.0
    short_score = 0.0
    notes = []
    # Trend önemi: daha uzun vadede trend daha etkili
    trend_weight = 40 if strategy == "LONG" else (30 if strategy == "SWING" else 20)
    rsi_weight = 20
    macd_weight = 20
    volume_weight = 20

    trend = specter.get('trend', 'NEUTRAL')
    trend_strength = specter.get('trend_strength', 0) / 100.0

    if trend == "BULLISH":
        long_score += trend_weight * (0.5 + trend_strength)
        notes.append(f"Trend: BULLISH (güç {trend_strength:.2f})")
    elif trend == "BEARISH":
        short_score += trend_weight * (0.5 + trend_strength)
        notes.append(f"Trend: BEARISH (güç {trend_strength:.2f})")

    rsi = indicators.get('rsi', 50)
    if rsi < 35:
        long_score += rsi_weight
        notes.append(f"RSI {rsi:.1f} (aşırı satım)")
    elif rsi > 65:
        short_score += rsi_weight
        notes.append(f"RSI {rsi:.1f} (aşırı alım)")

    macd_hist = indicators.get('macd_histogram', 0)
    if macd_hist > 0:
        long_score += macd_weight
        notes.append(f"MACD histogram pozitif ({macd_hist:.4f})")
    elif macd_hist < 0:
        short_score += macd_weight
        notes.append(f"MACD histogram negatif ({macd_hist:.4f})")

    vol_ratio = indicators.get('volume_ratio', 1.0)
    if vol_ratio > 1.5:
        if long_score > short_score:
            long_score += volume_weight * (min(vol_ratio, 3.0) / 2.0)
            notes.append(f"Hacim artışı onayı x{vol_ratio:.2f}")
        elif short_score > long_score:
            short_score += volume_weight * (min(vol_ratio, 3.0) / 2.0)
            notes.append(f"Hacim artışı onayı x{vol_ratio:.2f}")

    # Retest bonus
    if specter.get('has_bullish_retest'):
        long_score += 15
        notes.append("Bullish retest tespit edildi")
    if specter.get('has_bearish_retest'):
        short_score += 15
        notes.append("Bearish retest tespit edildi")

    # confidence scaling ve karar
    threshold = 25 if strategy == "SCALP" else (20 if strategy == "SWING" else 15)
    signal = "NEUTRAL"
    confidence = 0
    if long_score > short_score + threshold:
        signal = "LONG"
        confidence = int(min(100, (long_score - short_score)))
    elif short_score > long_score + threshold:
        signal = "SHORT"
        confidence = int(min(100, (short_score - long_score)))
    else:
        confidence = int(min(100, max(long_score, short_score) / 2))

    price = indicators.get('price', 0.0)
    atr = specter.get('atr', indicators.get('atr', 0.0))
    levels = calculate_trade_levels(price, atr, signal, profile)

    explanation = f"Heuristic sonuç: {signal} (Güven {confidence}%)\n" + "\n".join(f"- {n}" for n in notes)
    return {
        "signal": signal,
        "confidence": confidence,
        "explanation": explanation,
        "levels": levels,
        "profile": profile
    }

def get_gemini_analysis(indicators: Dict[str, Any], specter: Dict[str, Any], strategy: str, api_key: str) -> Dict[str, Any]:
    """
    Gemini çağrısı: kısa ve net JSON dönecek şekilde prompt oluşturulur.
    Eğer gemini yoksa ImportError fırlatır.
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini kütüphanesi mevcut değil.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    context = {
        "indicators": {
            "price": indicators.get('price'),
            "rsi": indicators.get('rsi'),
            "macd_hist": indicators.get('macd_histogram'),
            "atr_percent": indicators.get('atr_percent'),
            "volume_ratio": indicators.get('volume_ratio')
        },
        "specter": specter,
        "strategy": strategy
    }
    prompt = f"""
    SEN BİR PROFESYONEL KRİPTO ANALİSTİSİN. Verilen verileri kullanarak kısa JSON üret.
    Input:
    {json.dumps(context, ensure_ascii=False)}
    Output JSON formatında:
    {{
      "signal": "LONG/SHORT/NEUTRAL",
      "confidence": 0-100,
      "explanation": "kısa türkçe açıklama",
      "entry": float,
      "stop_loss": float,
      "take_profit": float
    }}
    """
    res = model.generate_content(prompt)
    text = res.text.strip().replace("```json","").replace("```","").strip()
    try:
        parsed = json.loads(text)
        return parsed
    except Exception:
        raise ValueError("Gemini'den JSON parse edilemedi.")

def get_combined_decision(indicators: Dict[str, Any], specter: Dict[str, Any], strategy: str, api_key: str = None) -> Dict[str, Any]:
    """
    Önce Gemini (varsa) denenir; uyum yoksa heuristic fallback ile birleşik karar üretilir.
    """
    heuristic = get_heuristic_signal(indicators, specter, strategy)
    if api_key and GEMINI_AVAILABLE:
        try:
            gemini = get_gemini_analysis(indicators, specter, strategy, api_key)
            # uyum kontrolü: aynı sinyal -> gemini'yi tercih et (kombine)
            if gemini.get('signal') == heuristic.get('signal'):
                # tamamlayıcı alanları ekle
                merged = heuristic.copy()
                merged.update({
                    "ai_source": "GEMINI",
                    "confidence": max(heuristic.get('confidence',0), int(gemini.get('confidence',0))),
                    "explanation": heuristic.get('explanation','') + "\nGemini ek: " + gemini.get('explanation', '')
                })
                merged['levels'].update({
                    'entry': gemini.get('entry', merged['levels'].get('entry')),
                    'stop_loss': gemini.get('stop_loss', merged['levels'].get('stop_loss')),
                    'take_profit': gemini.get('take_profit', merged['levels'].get('take_profit'))
                })
                return merged
            else:
                # Farklıysa heuristic'e not bırak
                heuristic['explanation'] += f"\nUYARI: Gemini farklı sinyal verdi ({gemini.get('signal')})."
                heuristic['ai_source'] = "HEURISTIC_WITH_GEMINI_CONTRAST"
                return heuristic
        except Exception as e:
            logging.warning(f"Gemini analizi başarısız: {e}. Heuristic geri dönüyor.")
            heuristic['ai_source'] = "HEURISTIC_FALLBACK"
            return heuristic
    else:
        heuristic['ai_source'] = "HEURISTIC_ONLY"
        return heuristic

# kayıt fonksiyonları
def load_records():
    if not RECORDS_FILE.exists():
        return []
    try:
        return json.loads(RECORDS_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []

def save_record(data: Dict[str, Any]):
    records = load_records()
    data['timestamp'] = datetime.utcnow().isoformat()
    records.append(data)
    try:
        RECORDS_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
        return True
    except Exception:
        return False
