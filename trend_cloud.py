# trend_cloud.py
# Specter Trend Cloud Indicator - Advanced Moving Average Based Trend Analysis

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional
import logging

class SpecterTrendCloud:
    """
    Specter Trend Cloud - Gelişmiş Trend Analiz Sistemi
    İki adaptif MA (kısa vs uzun) ve ATR tabanlı volatilite ayarlamalı cloud sistemi
    """
    
    def __init__(self, ma_type: str = "EMA", base_length: int = 20, atr_multiplier: float = 1.5):
        self.ma_type = ma_type
        self.base_length = base_length
        self.atr_multiplier = atr_multiplier
        self.short_length = base_length
        self.long_length = base_length * 2
        
    def calculate_ma(self, series: pd.Series, length: int) -> pd.Series:
        """Çeşitli MA türlerini hesaplar"""
        if self.ma_type == "SMA":
            return ta.sma(series, length=length)
        elif self.ma_type == "EMA":
            return ta.ema(series, length=length)
        elif self.ma_type == "WMA":
            return ta.wma(series, length=length)
        elif self.ma_type == "DEMA":
            return ta.dema(series, length=length)
        else:
            return ta.ema(series, length=length)  # Varsayılan EMA
    
    def calculate_specter_cloud(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Specter Trend Cloud hesaplamalarını yapar
        """
        try:
            # Temel MA'ları hesapla
            df['ma_short'] = self.calculate_ma(df['close'], self.short_length)
            df['ma_long'] = self.calculate_ma(df['close'], self.long_length)
            
            # ATR hesapla (volatilite için)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=200)
            
            # Trend belirle
            df['trend'] = np.where(df['ma_short'] > df['ma_long'], 1, -1)
            
            # ATR offset uygula
            df['atr_offset'] = df['atr'] * self.atr_multiplier
            
            # MA'ları offset ile kaydır
            df['ma_upper'] = np.where(
                df['trend'] == 1, 
                df['ma_short'] + df['atr_offset'],  # Bull trend - yukarı kaydır
                df['ma_long'] + df['atr_offset']    # Bear trend
            )
            
            df['ma_lower'] = np.where(
                df['trend'] == 1,
                df['ma_long'] - df['atr_offset'],   # Bull trend
                df['ma_short'] - df['atr_offset']   # Bear trend - aşağı kaydır
            )
            
            # Momentum gücü
            df['momentum_strength'] = (df['ma_short'] - df['ma_long']) / df['ma_long'] * 100
            
            # Retest sinyallerini hesapla
            df = self._calculate_retest_signals(df)
            
            # Trend gücü indeksi
            df['trend_power'] = self._calculate_trend_power(df)
            
            return df.dropna()
            
        except Exception as e:
            logging.error(f"Specter Cloud hesaplama hatası: {str(e)}")
            return None
    
    def _calculate_retest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retest sinyallerini hesaplar"""
        
        df['bullish_retest'] = False
        df['bearish_retest'] = False
        
        # Cooldown period (5 bar)
        cooldown = 5
        
        for i in range(cooldown, len(df)):
            current = df.iloc[i]
            prev_trend = df.iloc[i-1]['trend']
            current_trend = current['trend']
            
            # Trend devam ediyorsa
            if current_trend == prev_trend:
                if current_trend == 1:  # Bull trend
                    # Bullish retest: Low, short MA'ya dokunuyor
                    if (current['low'] <= current['ma_short'] <= current['high']):
                        # Cooldown kontrolü
                        recent_retests = any(df['bullish_retest'].iloc[i-cooldown:i])
                        if not recent_retests:
                            df.loc[df.index[i], 'bullish_retest'] = True
                
                elif current_trend == -1:  # Bear trend
                    # Bearish retest: High, short MA'ya dokunuyor
                    if (current['low'] <= current['ma_short'] <= current['high']):
                        recent_retests = any(df['bearish_retest'].iloc[i-cooldown:i])
                        if not recent_retests:
                            df.loc[df.index[i], 'bearish_retest'] = True
        
        return df
    
    def _calculate_trend_power(self, df: pd.DataFrame) -> pd.Series:
        """Trend gücünü hesaplar"""
        # Momentum, hacim ve volatilite kombinasyonu
        price_momentum = df['close'].pct_change(5)
        volume_trend = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
        volatility_normalized = df['atr'] / df['close']
        
        # Normalize edilmiş trend gücü
        trend_power = (
            price_momentum.rolling(5).mean() * 0.4 +
            volume_trend * 0.3 +
            (1 - volatility_normalized) * 0.3  # Düşük volatilite = güçlü trend
        )
        
        return trend_power

def calculate_specter_cloud(df: pd.DataFrame, ma_type: str = "EMA", 
                          base_length: int = 20, atr_multiplier: float = 1.5) -> Optional[pd.DataFrame]:
    """
    Specter Trend Cloud hesaplaması için ana fonksiyon
    """
    specter = SpecterTrendCloud(ma_type, base_length, atr_multiplier)
    return specter.calculate_specter_cloud(df)

def create_ai_snapshot(df: pd.DataFrame, funding_data: Dict = None) -> Dict[str, Any]:
    """
    AI analizi için indikatör snapshot'ı oluşturur
    """
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    # Temel indikatörler
    snapshot = {
        'price': float(latest['close']),
        'volume': float(latest['volume']),
        'high_24h': float(df['high'].tail(24).max()),
        'low_24h': float(df['low'].tail(24).min()),
        'price_change_24h': float((latest['close'] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100)
    }
    
    # Teknik indikatörler
    try:
        # RSI
        rsi = ta.rsi(df['close'], length=14)
        snapshot['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else 50.0
        
        # MACD
        macd = ta.macd(df['close'])
        if not macd.empty:
            snapshot['macd'] = float(macd.iloc[-1, 0]) if macd.shape[1] > 0 else 0
            snapshot['macd_signal'] = float(macd.iloc[-1, 1]) if macd.shape[1] > 1 else 0
            snapshot['macd_histogram'] = float(macd.iloc[-1, 2]) if macd.shape[1] > 2 else 0
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        if not bb.empty and bb.shape[1] >= 3:
            snapshot['bb_upper'] = float(bb.iloc[-1, 0])
            snapshot['bb_middle'] = float(bb.iloc[-1, 1])
            snapshot['bb_lower'] = float(bb.iloc[-1, 2])
            snapshot['bb_position'] = float((latest['close'] - bb.iloc[-1, 2]) / (bb.iloc[-1, 0] - bb.iloc[-1, 2]) * 100)
        
        # ATR
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        snapshot['atr'] = float(atr.iloc[-1]) if not atr.empty else 0
        snapshot['atr_percent'] = float((atr.iloc[-1] / latest['close']) * 100) if not atr.empty else 0
        
        # Volume indicators
        volume_ma = df['volume'].rolling(20).mean()
        snapshot['volume_ratio'] = float(latest['volume'] / volume_ma.iloc[-1]) if not volume_ma.empty else 1.0
        
        # Price momentum
        snapshot['momentum_5'] = float((latest['close'] / df['close'].iloc[-5] - 1) * 100)
        snapshot['momentum_10'] = float((latest['close'] / df['close'].iloc[-10] - 1) * 100)
        
        # Funding rate
        if funding_data:
            snapshot['funding_rate'] = funding_data.get('fundingRate', 0.0)
        else:
            snapshot['funding_rate'] = 0.0
            
    except Exception as e:
        logging.error(f"AI snapshot oluşturma hatası: {str(e)}")
    
    return snapshot

def generate_trading_levels(price: float, trend_direction: str, atr: float, 
                          volatility_regime: str = "NORMAL") -> Dict[str, float]:
    """
    Trend yönüne göre ticaret seviyeleri oluşturur
    """
    # Volatilite çarpanları
    volatility_multipliers = {
        "LOW": 0.8,
        "NORMAL": 1.0,
        "HIGH": 1.3,
        "EXTREME": 1.8
    }
    
    multiplier = volatility_multipliers.get(volatility_regime, 1.0)
    
    if trend_direction == "BULLISH":
        return {
            'entry': price,
            'stop_loss': price - (atr * 1.5 * multiplier),
            'take_profit_1': price + (atr * 2 * multiplier),
            'take_profit_2': price + (atr * 3 * multiplier),
            'take_profit_3': price + (atr * 4 * multiplier)
        }
    elif trend_direction == "BEARISH":
        return {
            'entry': price,
            'stop_loss': price + (atr * 1.5 * multiplier),
            'take_profit_1': price - (atr * 2 * multiplier),
            'take_profit_2': price - (atr * 3 * multiplier),
            'take_profit_3': price - (atr * 4 * multiplier)
        }
    else:
        return {
            'entry': price,
            'stop_loss': price - (atr * 2 * multiplier),
            'take_profit_1': price + (atr * 2 * multiplier),
            'take_profit_2': 0,
            'take_profit_3': 0
        }

# Hızlı analiz fonksiyonu
def quick_specter_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Hızlı Specter analizi için basit fonksiyon
    """
    if df.empty:
        return {}
    
    specter_data = calculate_specter_cloud(df)
    if specter_data is None:
        return {}
    
    latest = specter_data.iloc[-1]
    
    return {
        'trend': "BULLISH" if latest['trend'] == 1 else "BEARISH",
        'trend_strength': abs(latest['momentum_strength']),
        'has_bullish_retest': bool(latest['bullish_retest']),
        'has_bearish_retest': bool(latest['bearish_retest']),
        'cloud_top': float(latest['ma_upper']),
        'cloud_bottom': float(latest['ma_lower']),
        'momentum': float(latest['momentum_strength'])
    }
