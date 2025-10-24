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
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini kütüphanesi yüklü değil. 'pip install google-generativeai' ile yükleyin.")

# Kayıt dosyası
RECORDS_FILE = Path("ai_trading_records.json")

class AdvancedAIEngine:
    """
    Gelişmiş Hibrit AI Analiz Motoru
    Specter Trend Cloud + Gemini AI + Heuristic Kurallar
    """
    
    def __init__(self):
        self.risk_levels = {
            "LOW": 0.3,
            "MEDIUM": 0.5, 
            "HIGH": 0.7,
            "EXTREME": 1.0
        }
        
    def calculate_combined_score(self, ai_analysis: Dict, specter_analysis: Dict) -> float:
        """
        AI ve Specter analizlerini birleştirerek kombine skor hesaplar
        """
        ai_confidence = ai_analysis.get('confidence', 0) / 100.0
        ai_signal_strength = 1.0 if ai_analysis.get('signal') in ['LONG', 'SHORT'] else 0.3
        
        specter_strength = specter_analysis.get('trend_strength', 0)
        specter_trend = 1.0 if specter_analysis.get('trend') in ['BULLISH', 'BEARISH'] else 0.5
        
        # Retest sinyalleri bonus
        retest_bonus = 0
        retest_signals = specter_analysis.get('retest_signals', [])
        if retest_signals:
            retest_bonus = 0.2
        
        # Kombine skor
        combined = (
            ai_confidence * 0.4 +
            ai_signal_strength * 0.2 +
            specter_strength * 0.3 +
            retest_bonus * 0.1
        )
        
        return min(combined * 100, 100)
    
    def analyze_risk_level(self, indicators: Dict, specter_data: Dict) -> str:
        """
        Piyasa risk seviyesini analiz eder
        """
        volatility = indicators.get('atr_percent', 2.0)
        rsi = indicators.get('rsi', 50)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        
        risk_score = 0
        
        # Volatilite riski
        if volatility > 8:
            risk_score += 3
        elif volatility > 5:
            risk_score += 2
        elif volatility > 3:
            risk_score += 1
            
        # RSI riski
        if rsi > 80 or rsi < 20:
            risk_score += 2
        elif rsi > 70 or rsi < 30:
            risk_score += 1
            
        # Hacim riski
        if volume_ratio > 3:
            risk_score += 2
        elif volume_ratio > 2:
            risk_score += 1
            
        # Risk seviyesi belirle
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
        Risk yönetimine göre pozisyon büyüklüğü hesaplar
        """
        risk_percentage = self.risk_levels.get(risk_level, 0.5)
        max_risk_amount = account_balance * risk_percentage / 100
        
        if stop_distance <= 0:
            return {'size': 0, 'risk_amount': 0}
            
        position_size = max_risk_amount / stop_distance
        usdt_size = position_size * price
        
        return {
            'size': position_size,
            'usdt_size': usdt_size,
            'risk_amount': max_risk_amount,
            'risk_percentage': risk_percentage
        }

def get_heuristic_analysis(indicators: Dict, specter_data: Dict) -> Dict[str, Any]:
    """
    Gelişmiş kural-bazlı analiz motoru
    """
    long_score = 0
    short_score = 0
    explanation_points = []
    
    # Ağırlıklar
    weights = {
        'specter_trend': 30,
        'rsi_extreme': 20,
        'macd_signal': 25,
        'momentum_alignment': 15,
        'volume_confirmation': 10,
        'risk_adjustment': 10
    }
    
    # 1. Specter Trend Analizi
    trend = specter_data.get('trend', 'NEUTRAL')
    trend_strength = specter_data.get('trend_strength', 0)
    
    if trend == "BULLISH":
        long_score += weights['specter_trend'] * trend_strength
        explanation_points.append(f"📈 Specter BULLISH trend (Güç: {trend_strength:.1f}%)")
    elif trend == "BEARISH":
        short_score += weights['specter_trend'] * trend_strength
        explanation_points.append(f"📉 Specter BEARISH trend (Güç: {trend_strength:.1f}%)")
    
    # 2. RSI Analizi
    rsi = indicators.get('rsi', 50)
    if rsi < 30:
        long_score += weights['rsi_extreme']
        explanation_points.append(f"🔻 RSI({rsi:.1f}) aşırı satım - ALIM sinyali")
    elif rsi > 70:
        short_score += weights['rsi_extreme']
        explanation_points.append(f"🔺 RSI({rsi:.1f}) aşırı alım - SATIM sinyali")
    
    # 3. MACD Analizi
    macd_hist = indicators.get('macd_histogram', 0)
    if macd_hist > 0:
        long_score += weights['macd_signal']
        explanation_points.append(f"🟢 MACD Histogram pozitif (+{macd_hist:.4f})")
    elif macd_hist < 0:
        short_score += weights['macd_signal']
        explanation_points.append(f"🔴 MACD Histogram negatif ({macd_hist:.4f})")
    
    # 4. Momentum Hizalama
    price_momentum = indicators.get('momentum_5', 0)
    if (trend == "BULLISH" and price_momentum > 0) or (trend == "BEARISH" and price_momentum < 0):
        alignment_bonus = weights['momentum_alignment']
        if trend == "BULLISH":
            long_score += alignment_bonus
        else:
            short_score += alignment_bonus
        explanation_points.append(f"⚡ Momentum trend ile uyumlu (%{price_momentum:.2f})")
    
    # 5. Hacim Onayı
    volume_ratio = indicators.get('volume_ratio', 1.0)
    if volume_ratio > 1.5:
        volume_bonus = weights['volume_confirmation']
        if long_score > short_score:
            long_score += volume_bonus
        elif short_score > long_score:
            short_score += volume_bonus
        explanation_points.append(f"📊 Yüksek hacim onayı (x{volume_ratio:.1f})")
    
    # 6. Retest Sinyalleri
    retest_signals = specter_data.get('retest_signals', [])
    for retest in retest_signals:
        if 'BULLISH' in retest and trend == "BULLISH":
            long_score += 15
            explanation_points.append("🎯 Bullish Retest - Güçlü ALIM fırsatı")
        elif 'BEARISH' in retest and trend == "BEARISH":
            short_score += 15
            explanation_points.append("🎯 Bearish Retest - Güçlü SATIM fırsatı")
    
    # Sinyal Kararı
    signal = "NEUTRAL"
    confidence = 0
    threshold = 25
    
    if long_score > short_score + threshold:
        signal = "LONG"
        confidence = int((long_score - short_score) / 100 * 100)
    elif short_score > long_score + threshold:
        signal = "SHORT" 
        confidence = int((short_score - long_score) / 100 * 100)
    else:
        confidence = int(max(long_score, short_score) / 2)
    
    # Ticaret Seviyeleri
    price = indicators.get('price', 0)
    atr = indicators.get('atr', price * 0.02)
    risk_level = "MEDIUM"
    
    levels = calculate_trade_levels(price, signal, atr, risk_level)
    
    explanation = f"**Heuristic Sinyal: {signal} (Güven: {confidence}%)**\n\n"
    explanation += "**Analiz Detayları:**\n" + "\n".join([f"• {point}" for point in explanation_points])
    
    return {
        "signal": signal,
        "confidence": confidence,
        "explanation": explanation,
        "levels": levels,
        "scores": {
            "long_score": long_score,
            "short_score": short_score
        }
    }

def get_gemini_analysis(indicators: Dict, specter_data: Dict, api_key: str) -> Dict[str, Any]:
    """
    Gemini AI ile gelişmiş analiz
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini AI kütüphanesi yüklü değil.")
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Analiz verilerini hazırla
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
    
    GÖREVİN: Aşağıdaki teknik analiz verilerini kullanarak profesyonel bir ticaret planı oluşturmak.
    
    ANALİZ VERİLERİ:
    {json.dumps(analysis_context, indent=2, ensure_ascii=False)}
    
    TALEPLER:
    1. Net bir ticaret sinyali belirle: "LONG", "SHORT" veya "NEUTRAL"
    2. Bu sinyale olan güvenini 0-100 arasında puanla
    3. Detaylı analiz açıklaması yaz (Türkçe)
    4. Profesyonel ticaret seviyeleri belirle:
       - Giriş (entry)
       - Stop Loss (stop_loss) 
       - Take Profit (take_profit)
    5. Risk/Ödül oranı en az 1:2 olmalı
    6. ATR ve volatilite verilerini dikkate al
    
    CEVAP FORMATI (SADECE JSON):
    {{
      "signal": "LONG",
      "confidence": 85,
      "explanation": "Detaylı analiz açıklaması...",
      "entry": 12345.67,
      "stop_loss": 12200.50,
      "take_profit": 12600.25
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        ai_analysis = json.loads(cleaned_response)
        
        # Açıklamayı formatla
        ai_analysis['explanation'] = f"**🤖 Gemini AI Sinyal: {ai_analysis['signal']}**\n*Güven: {ai_analysis['confidence']}%*\n\n{ai_analysis['explanation']}"
        
        return ai_analysis
        
    except Exception as e:
        logging.error(f"Gemini analiz hatası: {str(e)}")
        raise

def calculate_trade_levels(price: float, signal: str, atr: float, risk_level: str = "MEDIUM") -> Dict[str, float]:
    """
    Ticaret seviyelerini hesaplar
    """
    risk_multipliers = {
        "LOW": 1.0,
        "MEDIUM": 1.5,
        "HIGH": 2.0,
        "EXTREME": 2.5
    }
    
    multiplier = risk_multipliers.get(risk_level, 1.5)
    stop_distance = atr * multiplier
    
    if signal == "LONG":
        return {
            'entry': price,
            'stop_loss': price - stop_distance,
            'take_profit': price + (stop_distance * 2)  # 1:2 R/R
        }
    elif signal == "SHORT":
        return {
            'entry': price,
            'stop_loss': price + stop_distance,
            'take_profit': price - (stop_distance * 2)  # 1:2 R/R
        }
    else:
        return {
            'entry': price,
            'stop_loss': 0,
            'take_profit': 0
        }

def get_ai_prediction(indicators: Dict, specter_data: Dict, api_key: str = None) -> Dict[str, Any]:
    """
    Ana AI tahmin fonksiyonu
    """
    # Specter verisi yoksa heuristic kullan
    if not specter_data:
        return get_heuristic_analysis(indicators, {})
    
    # API anahtarı varsa Gemini'yi dene
    if api_key and GEMINI_AVAILABLE:
        try:
            gemini_result = get_gemini_analysis(indicators, specter_data, api_key)
            
            # Gemini sonucunu heuristic ile doğrula
            heuristic_result = get_heuristic_analysis(indicators, specter_data)
            
            # Sinyaller uyumluysa Gemini'yi kullan
            if gemini_result.get('signal') == heuristic_result.get('signal'):
                return gemini_result
            else:
                # Uyumsuzsa heuristic'i kullan ama Gemini'nin güven skorunu düşür
                heuristic_result['confidence'] = max(
                    heuristic_result['confidence'] - 20,
                    heuristic_result['confidence'] * 0.7
                )
                heuristic_result['explanation'] += f"\n\n⚠️ *AI sinyalleri uyumsuz - Gemini: {gemini_result.get('signal')}*"
                return heuristic_result
                
        except Exception as e:
            logging.warning(f"Gemini başarısız, heuristic kullanılıyor: {str(e)}")
            return get_heuristic_analysis(indicators, specter_data)
    else:
        # API yoksa heuristic kullan
        return get_heuristic_analysis(indicators, specter_data)

def calculate_combined_score(ai_analysis: Dict, specter_analysis: Dict) -> float:
    """
    Kombine skor hesaplama yardımcı fonksiyonu
    """
    engine = AdvancedAIEngine()
    return engine.calculate_combined_score(ai_analysis, specter_analysis)

# Kayıt yönetimi fonksiyonları
def load_records() -> list:
    """Kayıtlı ticaret verilerini yükle"""
    if not RECORDS_FILE.exists():
        return []
    try:
        with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []

def save_record(record: Dict[str, Any]) -> bool:
    """Yeni ticaret kaydı ekle"""
    records = load_records()
    record['timestamp'] = datetime.utcnow().isoformat()
    record['id'] = len(records) + 1
    records.append(record)
    
    try:
        with open(RECORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def get_performance_stats() -> Dict[str, Any]:
    """Performans istatistiklerini hesapla"""
    records = load_records()
    if not records:
        return {}
    
    successful_trades = [r for r in records if r.get('success', False)]
    total_trades = len(records)
    success_rate = len(successful_trades) / total_trades * 100 if total_trades > 0 else 0
    
    return {
        'total_trades': total_trades,
        'successful_trades': len(successful_trades),
        'success_rate': success_rate,
        'avg_confidence': sum(r.get('confidence', 0) for r in records) / total_trades if total_trades > 0 else 0
    }
