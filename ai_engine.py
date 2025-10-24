# ai_engine.py
# Basit ve Etkili AI Analiz Motoru

import logging
from typing import Dict, Any
import json

logger = logging.getLogger(__name__)

# Gemini AI kontrolü
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def get_ai_prediction(indicators: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    """
    Basit ve etkili AI analizi
    """
    try:
        # API key varsa Gemini'yi kullan
        if api_key and GEMINI_AVAILABLE:
            return get_gemini_analysis(indicators, api_key)
        else:
            # Basit heuristic analiz
            return get_heuristic_analysis(indicators)
            
    except Exception as e:
        logger.error(f"AI prediction error: {str(e)}")
        return get_fallback_analysis()

def get_gemini_analysis(indicators: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """Gemini AI analizi"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Kripto teknik analiz uzmanısın. Aşağıdaki verileri analiz et:
        
        {json.dumps(indicators, indent=2)}
        
        Kısa ve net bir trading sinyali ver:
        - Sinyal: AL, SAT veya BEKLE
        - Kısa açıklama
        - Giriş, Stop Loss, Take Profit seviyeleri
        
        JSON formatında cevap ver:
        """
        
        response = model.generate_content(prompt)
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        analysis = json.loads(text)
        
        return {
            'signal': analysis.get('signal', 'BEKLE'),
            'explanation': analysis.get('explanation', 'AI analizi'),
            'entry': analysis.get('entry', indicators.get('price', 0)),
            'stop_loss': analysis.get('stop_loss', 0),
            'take_profit': analysis.get('take_profit', 0)
        }
        
    except Exception as e:
        logger.warning(f"Gemini failed: {str(e)}")
        return get_heuristic_analysis(indicators)

def get_heuristic_analysis(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Basit heuristic analiz"""
    score = indicators.get('score', 0)
    rsi = indicators.get('rsi14', 50)
    macd_hist = indicators.get('macd_histogram', 0)
    
    if score > 40 and rsi < 70 and macd_hist > 0:
        signal = "AL"
        explanation = "Güçlü al sinyali - Trend yükseliş yönünde"
    elif score < -40 and rsi > 30 and macd_hist < 0:
        signal = "SAT"
        explanation = "Güçlü sat sinyali - Trend düşüş yönünde"
    else:
        signal = "BEKLE"
        explanation = "Piyasa dengede - Bekle ve gör"
    
    price = indicators.get('price', 0)
    
    return {
        'signal': signal,
        'explanation': explanation,
        'entry': price,
        'stop_loss': price * 0.98 if signal == "AL" else price * 1.02,
        'take_profit': price * 1.04 if signal == "AL" else price * 0.96
    }

def get_fallback_analysis() -> Dict[str, Any]:
    """Fallback analiz"""
    return {
        'signal': 'BEKLE',
        'explanation': 'Analiz geçici olarak kullanılamıyor',
        'entry': 0,
        'stop_loss': 0,
        'take_profit': 0
    }
