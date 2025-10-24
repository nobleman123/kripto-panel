# market_analysis.py
# Piyasa Analizi ve Sentiment Modülü

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Piyasa analizi ve sentiment tespiti"""
    
    def __init__(self):
        self.coinglass_base = "https://open-api.coinglass.com/api/pro/v1"
        # Not: Coinglass API key için environment variable kullanın
        
    def get_fear_greed_index(self) -> int:
        """Fear & Greed Index"""
        try:
            # Basit implementasyon - gerçek API entegrasyonu için API key gerekli
            markets = self.get_market_overview()
            volatility = markets.get('volatility', 0.5)
            momentum = markets.get('momentum', 0.5)
            
            # Basit hesaplama
            fgi = int((1 - volatility + momentum) / 2 * 100)
            return max(0, min(100, fgi))
        except:
            return 50  # Varsayılan
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Piyasa genel görünümü"""
        try:
            # BTC dominance, total market cap vs.
            return {
                'btc_dominance': 48.5,
                'total_cap_change_24h': 2.3,
                'volume_24h': 85.2,
                'volatility': 0.65,
                'momentum': 0.72
            }
        except:
            return {
                'btc_dominance': 50.0,
                'total_cap_change_24h': 0.0,
                'volume_24h': 0.0,
                'volatility': 0.5,
                'momentum': 0.5
            }
    
    def get_funding_rates(self) -> Dict[str, float]:
        """Önemli coinlerin funding rate'leri"""
        return {
            'BTC': 0.0001,
            'ETH': 0.0002,
            'ADA': -0.0003,
            'SOL': 0.0005,
            'DOT': 0.0001
        }
    
    def get_liquidation_data(self) -> Dict[str, float]:
        """Likitasyon verileri"""
        return {
            'total_24h': 45.2,
            'long_ratio': 0.65,
            'short_ratio': 0.35,
            'btc_liquidation': 15.3
        }
    
    def calculate_market_sentiment(self) -> Dict[str, Any]:
        """Piyasa sentiment analizi"""
        try:
            fgi = self.get_fear_greed_index()
            markets = self.get_market_overview()
            funding = self.get_funding_rates()
            liquidations = self.get_liquidation_data()
            
            # Sentiment skoru
            sentiment_score = 0
            
            # Fear & Greed etkisi
            if fgi > 75:
                sentiment_score += 2
            elif fgi > 55:
                sentiment_score += 1
            elif fgi < 25:
                sentiment_score -= 2
            elif fgi < 45:
                sentiment_score -= 1
            
            # Piyasa momentumu
            if markets['total_cap_change_24h'] > 3:
                sentiment_score += 2
            elif markets['total_cap_change_24h'] > 1:
                sentiment_score += 1
            elif markets['total_cap_change_24h'] < -3:
                sentiment_score -= 2
            elif markets['total_cap_change_24h'] < -1:
                sentiment_score -= 1
            
            # Volatilite
            if markets['volatility'] > 0.7:
                sentiment_score -= 1
            
            # Funding rates
            positive_funding = sum(1 for rate in funding.values() if rate > 0)
            if positive_funding >= 3:
                sentiment_score += 1
            elif positive_funding <= 1:
                sentiment_score -= 1
            
            # Sentiment belirleme
            if sentiment_score >= 3:
                sentiment = "BULLISH"
                strength = min(100, 60 + sentiment_score * 10)
                analysis = "Piyasa güçlü al sinyali veriyor. Yükseliş momentumu devam edebilir."
            elif sentiment_score >= 1:
                sentiment = "MILD_BULLISH"
                strength = 50 + sentiment_score * 8
                analysis = "Piyasa hafif al yönlü. Dikkatli işlem önerilir."
            elif sentiment_score <= -3:
                sentiment = "BEARISH"
                strength = min(100, 60 + abs(sentiment_score) * 10)
                analysis = "Piyasa satış baskısı altında. Düşüş trendi devam edebilir."
            elif sentiment_score <= -1:
                sentiment = "MILD_BEARISH"
                strength = 50 + abs(sentiment_score) * 8
                analysis = "Piyasa hafif sat yönlü. Risk yönetimi önemli."
            else:
                sentiment = "NEUTRAL"
                strength = 50
                analysis = "Piyasa dengede. Yön belirleyici haber bekleniyor."
            
            return {
                'sentiment': sentiment,
                'score': sentiment_score,
                'strength': strength,
                'fear_greed': fgi,
                'analysis': analysis,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sentiment calculation error: {str(e)}")
            return {
                'sentiment': 'NEUTRAL',
                'score': 0,
                'strength': 50,
                'fear_greed': 50,
                'analysis': 'Analiz geçici olarak kullanılamıyor.',
                'timestamp': datetime.utcnow().isoformat()
            }

def get_market_sentiment() -> Dict[str, Any]:
    """Piyasa sentimentini getir"""
    analyzer = MarketAnalyzer()
    return analyzer.calculate_market_sentiment()

def get_market_snapshot(symbol: str) -> Dict[str, Any]:
    """Sembole özel piyara verileri"""
    try:
        analyzer = MarketAnalyzer()
        
        return {
            'symbol': symbol,
            'fear_greed': analyzer.get_fear_greed_index(),
            'market_overview': analyzer.get_market_overview(),
            'funding_rates': analyzer.get_funding_rates(),
            'liquidation_data': analyzer.get_liquidation_data(),
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Market snapshot error for {symbol}: {str(e)}")
        return {
            'symbol': symbol,
            'fear_greed': 50,
            'market_overview': {},
            'funding_rates': {},
            'liquidation_data': {},
            'timestamp': datetime.utcnow().isoformat()
        }
