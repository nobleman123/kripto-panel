# ai_engine.py
# Gelişmiş AI Analiz Motoru - Premium Sinyal Sistemi

import math
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

# Gemini AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class PremiumAIEngine:
    """Premium AI analiz motoru"""
    
    def __init__(self):
        self.trading_styles = {
            "SCALP": {
                "timeframes": ["1m", "5m", "15m"],
                "risk_multiplier": 0.5,
                "min_confidence": 80,
                "indicators": ["rsi", "macd", "volume", "momentum"]
            },
            "SWING": {
                "timeframes": ["15m", "1h", "4h"], 
                "risk_multiplier": 1.0,
                "min_confidence": 75,
                "indicators": ["ema", "rsi", "macd", "bb", "volume"]
            },
            "POSITION": {
                "timeframes": ["4h", "1d"],
                "risk_multiplier": 2.0,
                "min_confidence": 70,
                "indicators": ["ema", "macd", "rsi", "trend", "volume"]
            }
        }
    
    def calculate_premium_score(self, ai_analysis: Dict, technical_data: Any, trading_style: str) -> float:
        """Premium sinyal skoru hesaplama"""
        try:
            base_score = ai_analysis.get('confidence', 0)
            
            # Teknik analiz bonusları
            tech_bonus = self._calculate_technical_bonus(technical_data, trading_style)
            
            # Risk bonusu
            risk_bonus = self._calculate_risk_bonus(ai_analysis)
            
            # Zaman bonusu
            time_bonus = self._calculate_time_bonus(trading_style)
            
            premium_score = (
                base_score * 0.6 +
                tech_bonus * 0.25 +
                risk_bonus * 0.10 +
                time_bonus * 0.05
            )
            
            return min(100, premium_score)
            
        except Exception as e:
            logger.error(f"Premium score error: {str(e)}")
            return ai_analysis.get('confidence', 50)
    
    def _calculate_technical_bonus(self, technical_data: Any, trading_style: str) -> float:
        """Teknik analiz bonusu"""
        try:
            if technical_data is None or technical_data.empty:
                return 0
                
            latest = technical_data.iloc[-1]
            bonus = 0
            
            # Trend gücü
            trend_strength = latest.get('trend_strength', 0)
            bonus += trend_strength * 20
            
            # Momentum
            momentum = latest.get('momentum_score', 0)
            bonus += abs(momentum) * 15
            
            # Volume confirmation
            volume_ratio = latest.get('volume_ratio', 1)
            if volume_ratio > 1.5:
                bonus += 10
                
            # EMA alignment
            ema_alignment = latest.get('ema_alignment', 0)
            bonus += abs(ema_alignment) * 10
            
            return min(30, bonus)
            
        except:
            return 0
    
    def _calculate_risk_bonus(self, ai_analysis: Dict) -> float:
        """Risk bonusu"""
        try:
            rr_ratio = ai_analysis.get('risk_reward', 0)
            if rr_ratio >= 3:
                return 15
            elif rr_ratio >= 2:
                return 10
            elif rr_ratio >= 1.5:
                return 5
            return 0
        except:
            return 0
    
    def _calculate_time_bonus(self, trading_style: str) -> float:
        """Zaman bonusu"""
        try:
            if trading_style == "SCALP":
                return 10  # Scalp için ek bonus
            return 5
        except:
            return 0

def get_ai_prediction(technical_data: Any, market_data: Dict, trading_style: str, api_key: str = None) -> Dict[str, Any]:
    """
    Premium AI tahmini
    """
    try:
        # API key varsa Gemini'yi kullan
        if api_key and GEMINI_AVAILABLE:
            try:
                return get_gemini_premium_analysis(technical_data, market_data, trading_style, api_key)
            except Exception as e:
                logger.warning(f"Gemini failed, using heuristic: {str(e)}")
        
        # Fallback: Gelişmiş heuristic
        return get_advanced_heuristic_analysis(technical_data, market_data, trading_style)
        
    except Exception as e:
        logger.error(f"AI prediction error: {str(e)}")
        return get_fallback_analysis()

def get_gemini_premium_analysis(technical_data: Any, market_data: Dict, trading_style: str, api_key: str) -> Dict[str, Any]:
    """Gemini premium analiz"""
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini not available")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    # Analiz verilerini hazırla
    analysis_context = create_premium_analysis_context(technical_data, market_data, trading_style)
    
    prompt = f"""
    SEN PROFESYONEL BİR KRİPTO PARA TRADER'SIN!
    
    GÖREV: Aşağıdaki verileri analiz ederek KESİN bir ticaret sinyali üret.
    
    TRADING STİL: {trading_style}
    ANALİZ VERİLERİ: {json.dumps(analysis_context, indent=2, ensure_ascii=False)}
    
    KRİTERLER:
    1. Net sinyal: "GÜÇLÜ AL", "AL", "NÖTR", "SAT", "GÜÇLÜ SAT"
    2. Güven skoru: 0-100 (sadece güçlü sinyallerde 80+)
    3. Detaylı Türkçe açıklama
    4. Kesin giriş/stop/hedef fiyatları
    5. Risk/Ödül oranı minimum 1:2
    
    CEVAP FORMATI:
    {{
      "signal": "GÜÇLÜ AL",
      "confidence": 85,
      "explanation": "Detaylı analiz...",
      "entry": 12345.67,
      "stop_loss": 12200.50,
      "take_profit": 12600.25,
      "risk_reward": 2.5
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        ai_result = json.loads(cleaned_response)
        
        # Format explanation
        ai_result['explanation'] = f"**🤖 PREMIUM AI SİNYAL: {ai_result['signal']}**\n*Güven: {ai_result['confidence']}% | Stil: {trading_style}*\n\n{ai_result['explanation']}"
        
        return ai_result
        
    except Exception as e:
        logger.error(f"Gemini analysis error: {str(e)}")
        raise

def get_advanced_heuristic_analysis(technical_data: Any, market_data: Dict, trading_style: str) -> Dict[str, Any]:
    """Gelişmiş heuristic analiz"""
    try:
        if technical_data is None or technical_data.empty:
            return get_fallback_analysis()
            
        latest = technical_data.iloc[-1]
        
        # Skor hesaplama
        long_score, short_score, reasons = calculate_advanced_scores(latest, trading_style)
        
        # Sinyal belirleme
        signal, confidence = determine_signal(long_score, short_score, trading_style)
        
        # Ticaret seviyeleri
        levels = calculate_trading_levels(latest, signal, trading_style)
        
        explanation = create_heuristic_explanation(signal, confidence, reasons, trading_style)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "explanation": explanation,
            **levels
        }
        
    except Exception as e:
        logger.error(f"Heuristic analysis error: {str(e)}")
        return get_fallback_analysis()

def calculate_advanced_scores(latest: Any, trading_style: str) -> tuple:
    """Gelişmiş skor hesaplama"""
    long_score = 0
    short_score = 0
    reasons = []
    
    # EMA Analysis
    ema_alignment = latest.get('ema_alignment', 0)
    if ema_alignment > 0:
        long_score += 25
        reasons.append("EMA dizilimi yükseliş destekliyor")
    elif ema_alignment < 0:
        short_score += 25
        reasons.append("EMA dizilimi düşüş destekliyor")
    
    # RSI Analysis
    rsi = latest.get('rsi_14', 50)
    if rsi < 30:
        long_score += 20
        reasons.append("RSI aşırı satım bölgesinde")
    elif rsi > 70:
        short_score += 20
        reasons.append("RSI aşırı alım bölgesinde")
    
    # MACD Analysis
    macd_hist = latest.get('macd_histogram', 0)
    if macd_hist > 0:
        long_score += 15
        reasons.append("MACD histogram pozitif")
    elif macd_hist < 0:
        short_score += 15
        reasons.append("MACD histogram negatif")
    
    # Volume Analysis
    volume_ratio = latest.get('volume_ratio', 1)
    if volume_ratio > 1.5:
        if long_score > short_score:
            long_score += 10
            reasons.append("Yüksek hacim yükselişi destekliyor")
        else:
            short_score += 10
            reasons.append("Yüksek hacim düşüşü destekliyor")
    
    # Trend Strength
    trend_strength = latest.get('trend_strength', 0)
    if trend_strength > 0.7:
        if long_score > short_score:
            long_score += 15
            reasons.append("Güçlü yükseliş trendi")
        else:
            short_score += 15
            reasons.append("Güçlü düşüş trendi")
    
    return long_score, short_score, reasons

def determine_signal(long_score: float, short_score: float, trading_style: str) -> tuple:
    """Sinyal belirleme"""
    threshold = 25
    
    if trading_style == "SCALP":
        threshold = 30
    elif trading_style == "POSITION":
        threshold = 20
    
    if long_score > short_score + threshold:
        if long_score > 60:
            signal = "GÜÇLÜ AL"
        else:
            signal = "AL"
        confidence = min(95, int((long_score - short_score) / 100 * 100))
    elif short_score > long_score + threshold:
        if short_score > 60:
            signal = "GÜÇLÜ SAT"
        else:
            signal = "SAT"
        confidence = min(95, int((short_score - long_score) / 100 * 100))
    else:
        signal = "NÖTR"
        confidence = max(30, int(max(long_score, short_score) / 2))
    
    return signal, confidence

def calculate_trading_levels(latest: Any, signal: str, trading_style: str) -> Dict[str, float]:
    """Ticaret seviyeleri"""
    price = latest.get('close', 0)
    atr = latest.get('atr', price * 0.02)
    
    if trading_style == "SCALP":
        multiplier = 0.8
        rr_ratio = 1.5
    elif trading_style == "SWING":
        multiplier = 1.5
        rr_ratio = 2.0
    else:  # POSITION
        multiplier = 2.0
        rr_ratio = 3.0
    
    stop_distance = atr * multiplier
    
    if signal in ["GÜÇLÜ AL", "AL"]:
        return {
            'entry': price,
            'stop_loss': price - stop_distance,
            'take_profit': price + (stop_distance * rr_ratio),
            'risk_reward': rr_ratio
        }
    elif signal in ["GÜÇLÜ SAT", "SAT"]:
        return {
            'entry': price,
            'stop_loss': price + stop_distance,
            'take_profit': price - (stop_distance * rr_ratio),
            'risk_reward': rr_ratio
        }
    else:
        return {
            'entry': price,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward': 0
        }

def create_heuristic_explanation(signal: str, confidence: int, reasons: List[str], trading_style: str) -> str:
    """Açıklama oluştur"""
    explanation = f"**🎯 HEURISTIC SİNYAL: {signal}**\n"
    explanation += f"*Güven: {confidence}% | Stil: {trading_style}*\n\n"
    explanation += "**Analiz Sonuçları:**\n"
    
    for reason in reasons:
        explanation += f"• {reason}\n"
    
    explanation += f"\n**İşlem Stili:** {trading_style}\n"
    explanation += "**Öneri:** " + get_trading_advice(signal, confidence, trading_style)
    
    return explanation

def get_trading_advice(signal: str, confidence: int, trading_style: str) -> str:
    """İşlem önerisi"""
    if signal in ["GÜÇLÜ AL", "GÜÇLÜ SAT"]:
        return "Yüksek güvenilirlik - Güçlü pozisyon alınabilir"
    elif signal in ["AL", "SAT"]:
        return "Orta güvenilirlik - Dikkatli pozisyon alınabilir"
    else:
        return "Düşük güvenilirlik - İşlem önerilmiyor"

def create_premium_analysis_context(technical_data: Any, market_data: Dict, trading_style: str) -> Dict[str, Any]:
    """Premium analiz context'i oluştur"""
    if technical_data is None or technical_data.empty:
        return {}
    
    latest = technical_data.iloc[-1]
    
    return {
        "trading_style": trading_style,
        "price_action": {
            "current_price": latest.get('close'),
            "high_24h": latest.get('high'),
            "low_24h": latest.get('low'),
            "volume": latest.get('volume')
        },
        "technical_indicators": {
            "ema_alignment": latest.get('ema_alignment'),
            "rsi_14": latest.get('rsi_14'),
            "macd_histogram": latest.get('macd_histogram'),
            "trend_strength": latest.get('trend_strength'),
            "volume_ratio": latest.get('volume_ratio'),
            "atr_percent": latest.get('atr_percent')
        },
        "market_conditions": market_data
    }

def get_fallback_analysis() -> Dict[str, Any]:
    """Fallback analiz"""
    return {
        "signal": "NÖTR",
        "confidence": 30,
        "explanation": "**⚠️ ANALİZ GEÇİCİ OLARAK KULLANILAMIYOR**\nTeknik bir sorun nedeniyle AI analizi yapılamıyor. Lütfen daha sonra tekrar deneyin.",
        "entry": 0,
        "stop_loss": 0,
        "take_profit": 0,
        "risk_reward": 0
    }

def calculate_premium_score(ai_analysis: Dict, technical_data: Any, trading_style: str) -> float:
    """Premium skor hesaplama"""
    engine = PremiumAIEngine()
    return engine.calculate_premium_score(ai_analysis, technical_data, trading_style)
