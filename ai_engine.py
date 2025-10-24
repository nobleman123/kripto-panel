# ai_engine.py
# Gelişmiş Hibrit AI Analiz Motoru - Specter Trend Cloud + Gemini AI

import math
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Gemini AI kütüphanesini içe aktarmayı dene
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini kütüphanesi yüklü değil. 'pip install google-generativeai' ile yükleyin.")

# Kayıt dosyası
RECORDS_FILE = Path("ai_trading_records.json")

class AdvancedAIEngine:
    """
    Gelişmiş Hibrit AI Analiz Motoru
    """
    def __init__(self):
        # risk değerleri: fraction (0.3 = %30)
        self.risk_levels = {
            "LOW": 0.03,        # örnek: low = 3% risk
            "MEDIUM": 0.01,     # medium = 1% risk
            "HIGH": 0.02,       # high = 2% risk (kullanıcı tercihi ile değişebilir)
            "EXTREME": 0.05
        }
        
    def calculate_combined_score(self, ai_analysis: Dict, specter_analysis: Dict) -> float:
        """
        AI ve Specter analizlerini birleştirerek kombine skor hesaplar
        """
        ai_confidence = (ai_analysis.get('confidence', 0) / 100.0) if ai_analysis else 0.0
        ai_signal_strength = 1.0 if ai_analysis and ai_analysis.get('signal') in ['LONG', 'SHORT'] else 0.3
        
        specter_strength = specter_analysis.get('trend_strength', 0) if specter_analysis else 0.0
        specter_trend = 1.0 if specter_analysis and specter_analysis.get('trend') in ['BULLISH','BEARISH'] else 0.5
        
        retest_bonus = 0.2 if specter_analysis and specter_analysis.get('retest_signals') else 0.0
        
        combined = (
            ai_confidence * 0.45 +
            ai_signal_strength * 0.15 +
            specter_strength * 0.3 +
            retest_bonus * 0.1
        )
        return max(0.0, min(combined * 100, 100.0))
    
    def analyze_risk_level(self, indicators: Dict, specter_data: Dict) -> str:
        """
        Basit risk seviyesi analizi
        """
        volatility = indicators.get('atr_percent', 2.0)
        rsi = indicators.get('rsi', 50)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        risk_score = 0
        if volatility > 8:
            risk_score += 3
        elif volatility > 5:
            risk_score += 2
        elif volatility > 3:
            risk_score += 1
            
        if rsi > 80 or rsi < 20:
            risk_score += 2
        elif rsi > 70 or rsi < 30:
            risk_score += 1
            
        if volume_ratio > 3:
            risk_score += 2
        elif volume_ratio > 2:
            risk_score += 1
            
        if risk_score >= 5:
            return "EXTREME"
        elif risk_score >= 3:
            return "HIGH"
        elif risk_score >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_position_size(self, account_balance: float, risk_level: str, 
                              stop_distance: float, price: float) -> Dict[str, float]:
        """
        Risk yönetimine göre pozisyon büyüklüğü hesaplar.
        risk_levels değerleri fraction (örn 0.01 = %1)
        """
        risk_fraction = self.risk_levels.get(risk_level, 0.01)
        max_risk_amount = account_balance * risk_fraction
        if stop_distance <= 0 or price <= 0:
            return {'size': 0.0, 'usdt_size': 0.0, 'risk_amount': max_risk_amount, 'risk_fraction': risk_fraction}
        position_size = max_risk_amount / stop_distance
        usdt_size = position_size * price
        return {'size': position_size, 'usdt_size': usdt_size, 'risk_amount': max_risk_amount, 'risk_fraction': risk_fraction}

def calculate_trade_levels(price: float, signal: str, atr: float, risk_level: str = "MEDIUM") -> Dict[str, float]:
    """
    Ticaret seviyelerini hesaplar (aynı mantık trendlerde kullanılır)
    """
    risk_multipliers = {"LOW": 1.0, "MEDIUM": 1.5, "HIGH": 2.0, "EXTREME": 2.5}
    multiplier = risk_multipliers.get(risk_level, 1.5)
    stop_distance = max(atr * multiplier, atr * 0.5)
    if signal == "LONG":
        return {'entry': price, 'stop_loss': price - stop_distance, 'take_profit': price + (stop_distance * 2)}
    elif signal == "SHORT":
        return {'entry': price, 'stop_loss': price + stop_distance, 'take_profit': price - (stop_distance * 2)}
    else:
        return {'entry': price, 'stop_loss': 0.0, 'take_profit': 0.0}

def get_heuristic_analysis(indicators: Dict, specter_data: Dict) -> Dict[str, Any]:
    """
    Gelişmiş kural-bazlı analiz motoru (heuristic)
    """
    long_score = 0.0
    short_score = 0.0
    explanation_points = []
    weights = {
        'specter_trend': 30,
        'rsi_extreme': 20,
        'macd_signal': 25,
        'momentum_alignment': 15,
        'volume_confirmation': 10
    }
    
    trend = specter_data.get('trend', 'NEUTRAL') if specter_data else 'NEUTRAL'
    trend_strength = specter_data.get('trend_strength', 0) if specter_data else 0
    
    if trend == "BULLISH":
        long_score += weights['specter_trend'] * trend_strength
        explanation_points.append(f"Specter BULLISH trend (Güç: {trend_strength:.2f})")
    elif trend == "BEARISH":
        short_score += weights['specter_trend'] * trend_strength
        explanation_points.append(f"Specter BEARISH trend (Güç: {trend_strength:.2f})")
    
    rsi = indicators.get('rsi', 50)
    if rsi < 30:
        long_score += weights['rsi_extreme']
        explanation_points.append(f"RSI({rsi:.1f}) aşırı satım - ALIM sinyali")
    elif rsi > 70:
        short_score += weights['rsi_extreme']
        explanation_points.append(f"RSI({rsi:.1f}) aşırı alım - SATIM sinyali")
    
    macd_hist = indicators.get('macd_histogram', 0)
    if macd_hist > 0:
        long_score += weights['macd_signal']
        explanation_points.append(f"MACD histogram pozitif ({macd_hist:.4f})")
    elif macd_hist < 0:
        short_score += weights['macd_signal']
        explanation_points.append(f"MACD histogram negatif ({macd_hist:.4f})")
    
    price_momentum = indicators.get('momentum_5', 0)
    if (trend == "BULLISH" and price_momentum > 0) or (trend == "BEARISH" and price_momentum < 0):
        if trend == "BULLISH":
            long_score += weights['momentum_alignment']
        else:
            short_score += weights['momentum_alignment']
        explanation_points.append(f"Momentum trend ile uyumlu ({price_momentum:.2f})")
    
    volume_ratio = indicators.get('volume_ratio', 1.0)
    if volume_ratio > 1.5:
        if long_score > short_score:
            long_score += weights['volume_confirmation']
        elif short_score > long_score:
            short_score += weights['volume_confirmation']
        explanation_points.append(f"Yüksek hacim onayı (x{volume_ratio:.2f})")
    
    # Retest bonusları
    for r in specter_data.get('retest_signals', []) if specter_data else []:
        if 'BULLISH' in r and trend == "BULLISH":
            long_score += 15
            explanation_points.append("Bullish Retest - ALIM fırsatı")
        if 'BEARISH' in r and trend == "BEARISH":
            short_score += 15
            explanation_points.append("Bearish Retest - SATIM fırsatı")
    
    signal = "NEUTRAL"
    confidence = 0
    threshold = 25.0
    if long_score > short_score + threshold:
        signal = "LONG"
        confidence = int(min(100, (long_score - short_score)))
    elif short_score > long_score + threshold:
        signal = "SHORT"
        confidence = int(min(100, (short_score - long_score)))
    else:
        confidence = int(min(100, max(long_score, short_score) / 2))
    
    price = indicators.get('price', 0.0)
    atr = indicators.get('atr', max(price * 0.01, 0.0))
    risk_level = "MEDIUM"
    levels = calculate_trade_levels(price, signal, atr, risk_level)
    
    explanation = f"Heuristic Sinyal: {signal} (Güven: {confidence}%)\nAnaliz Notları:\n"
    explanation += "\n".join([f"- {pt}" for pt in explanation_points]) if explanation_points else "- Not yok"
    
    # Top-level uyumluluk: entry/stop/tp anahtarları
    result = {
        "signal": signal,
        "confidence": confidence,
        "explanation": explanation,
        "levels": levels,
        "entry": levels.get('entry', 0.0),
        "stop_loss": levels.get('stop_loss', 0.0),
        "take_profit": levels.get('take_profit', 0.0),
        "scores": {"long_score": long_score, "short_score": short_score}
    }
    return result

def get_gemini_analysis(indicators: Dict, specter_data: Dict, api_key: str) -> Dict[str, Any]:
    """
    Gemini AI ile gelişmiş analiz - eğer genai yoksa ImportError fırlatır
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini AI kütüphanesi yüklü değil.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    analysis_context = {
        "price_action": {
            "current_price": indicators.get('price'),
            "24h_high": indicators.get('high_24h'),
            "24h_low": indicators.get('low_24h'),
            "24h_change": indicators.get('price_change_24h')
        },
        "technical_indicators": {
            "rsi": indicators.get('rsi'),
            "macd": indicators.get('macd_histogram'),
            "atr_percent": indicators.get('atr_percent'),
            "bb_position": indicators.get('bb_position'),
            "volume_ratio": indicators.get('volume_ratio')
        },
        "specter_analysis": {
            "trend": specter_data.get('trend'),
            "trend_strength": specter_data.get('trend_strength'),
            "retest_signals": specter_data.get('retest_signals', []),
            "momentum": specter_data.get('momentum')
        }
    }
    prompt = f"""
    SEN BİR USTA KRİPTO PARA VADELİ İŞLEM ANALİSTİSİN.

    ANALİZ VERİLERİ:
    {json.dumps(analysis_context, indent=2, ensure_ascii=False)}

    TALEP:
    1) Sinyal: LONG/SHORT/NEUTRAL
    2) Güven (0-100)
    3) Kısa açıklama (Türkçe)
    4) Giriş, stop_loss, take_profit (float)
    CEVAP SADECE JSON olarak:
    {{ "signal": "...", "confidence": 85, "explanation": "...", "entry": 123, "stop_loss": 120, "take_profit": 130 }}
    """
    response = model.generate_content(prompt)
    cleaned = response.text.strip().replace("```json", "").replace("```", "").strip()
    ai_analysis = json.loads(cleaned)
    ai_analysis['explanation'] = f"Gemini AI Sinyal: {ai_analysis.get('signal')} (G:{ai_analysis.get('confidence',0)}%)\n" + ai_analysis.get('explanation','')
    return ai_analysis

def get_ai_prediction(indicators: Dict, specter_data: Dict, api_key: str = None) -> Dict[str, Any]:
    """
    Ana AI tahmin fonksiyonu - Gemini yoksa veya hata varsa heuristic fallback
    """
    if specter_data is None:
        return get_heuristic_analysis(indicators, {})
    if api_key and GEMINI_AVAILABLE:
        try:
            gemini_result = get_gemini_analysis(indicators, specter_data, api_key)
            heuristic_result = get_heuristic_analysis(indicators, specter_data)
            # uyum kontrolü
            if gemini_result.get('signal') == heuristic_result.get('signal'):
                # eksik alanları tamamla
                for k in ['entry','stop_loss','take_profit']:
                    if k not in gemini_result:
                        gemini_result[k] = heuristic_result.get(k, 0.0)
                return gemini_result
            else:
                # uyumsuzsa heuristic'e ek bilgi ekle
                heuristic_result['confidence'] = max(0, min(100, heuristic_result.get('confidence', 0) - 10))
                heuristic_result['explanation'] += f"\nUYARI: Gemini farklı sinyal üretti ({gemini_result.get('signal')})."
                return heuristic_result
        except Exception as e:
            logging.warning(f"Gemini başarısız, heuristic'e dönülüyor: {e}")
            return get_heuristic_analysis(indicators, specter_data)
    else:
        return get_heuristic_analysis(indicators, specter_data)

def calculate_combined_score(ai_analysis: Dict, specter_analysis: Dict) -> float:
    engine = AdvancedAIEngine()
    return engine.calculate_combined_score(ai_analysis or {}, specter_analysis or {})

# Kayıt yönetimi
def load_records() -> list:
    if not RECORDS_FILE.exists():
        return []
    try:
        return json.loads(RECORDS_FILE.read_text(encoding='utf-8'))
    except Exception:
        return []

def save_record(record: Dict[str, Any]) -> bool:
    records = load_records()
    record['timestamp'] = datetime.utcnow().isoformat()
    record['id'] = len(records) + 1
    records.append(record)
    try:
        RECORDS_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        return True
    except Exception as e:
        logging.error(f"Kayıt kaydetme hatası: {e}")
        return False

def get_performance_stats() -> Dict[str, Any]:
    records = load_records()
    if not records:
        return {}
    successful_trades = [r for r in records if r.get('success')]
    total = len(records)
    return {
        'total_trades': total,
        'successful_trades': len(successful_trades),
        'success_rate': (len(successful_trades) / total * 100) if total>0 else 0,
        'avg_confidence': sum(r.get('confidence',0) for r in records)/total if total>0 else 0
    }
