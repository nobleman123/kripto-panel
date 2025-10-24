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
        self.ma_type = ma_type.upper() if ma_type else "EMA"
        self.base_length = max(2, int(base_length))
        self.atr_multiplier = float(atr_multiplier)
        self.short_length = self.base_length
        self.long_length = max(3, self.base_length * 2)
        
    def calculate_ma(self, series: pd.Series, length: int) -> pd.Series:
        """Çeşitli MA türlerini hesaplar"""
        try:
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
        except Exception as e:
            logging.warning(f"MA hesaplama hatası: {e}")
            return series.rolling(length, min_periods=1).mean()
    
    def calculate_specter_cloud(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Specter Trend Cloud hesaplamalarını yapar
        """
        if df is None or df.empty:
            return None
        df = df.copy().reset_index(drop=True)
        try:
            # Temel MA'ları hesapla
            df['ma_short'] = self.calculate_ma(df['close'], self.short_length)
            df['ma_long'] = self.calculate_ma(df['close'], self.long_length)
            
            # ATR hesapla (volatilite için) - kısa ve uzun versiyonlar
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Trend belirle (1 = bull, -1 = bear)
            df['trend'] = np.where(df['ma_short'] > df['ma_long'], 1, -1)
            
            # ATR offset uygula
            df['atr_offset'] = df['atr'] * self.atr_multiplier.fillna(1) if hasattr(self.atr_multiplier, 'fillna') else df['atr'] * self.atr_multiplier
            
            # MA'ları offset ile kaydır
            df['ma_upper'] = np.where(
                df['trend'] == 1, 
                df['ma_short'] + df['atr_offset'],
                df['ma_long'] + df['atr_offset']
            )
            
            df['ma_lower'] = np.where(
                df['trend'] == 1,
                df['ma_long'] - df['atr_offset'],
                df['ma_short'] - df['atr_offset']
            )
            
            # Momentum gücü
            df['momentum_strength'] = ((df['ma_short'] - df['ma_long']) / (df['ma_long'].replace(0, np.nan))) * 100
            df['momentum_strength'] = df['momentum_strength'].fillna(0)
            
            # Retest sinyallerini hesapla
            df = self._calculate_retest_signals(df)
            
            # Trend gücü indeksi
            df['trend_power'] = self._calculate_trend_power(df)
            
            return df.dropna().reset_index(drop=True)
            
        except Exception as e:
            logging.error(f"Specter Cloud hesaplama hatası: {str(e)}")
            return None
    
    def _calculate_retest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retest sinyallerini hesaplar"""
        df = df.copy()
        df['bullish_retest'] = False
        df['bearish_retest'] = False
        
        cooldown = 5
        for i in range(cooldown, len(df)):
            try:
                current = df.iloc[i]
                prev_trend = df.iloc[i-1]['trend']
                current_trend = current['trend']
                
                if current_trend == prev_trend:
                    # Kontrollü retest mantığı: short MA çevresine dokunma
                    if current_trend == 1:  # Bull
                        if (current['low'] <= current['ma_short'] <= current['high']):
                            if not df['bullish_retest'].iloc[i-cooldown:i].any():
                                df.at[i, 'bullish_retest'] = True
                    else:  # Bear
                        if (current['low'] <= current['ma_short'] <= current['high']):
                            if not df['bearish_retest'].iloc[i-cooldown:i].any():
                                df.at[i, 'bearish_retest'] = True
            except Exception:
                continue
        return df
    
    def _calculate_trend_power(self, df: pd.DataFrame) -> pd.Series:
        """Trend gücünü hesaplar"""
        try:
            price_momentum = df['close'].pct_change(5).fillna(0)
            vol_10 = df['volume'].rolling(10).mean()
            vol_30 = df['volume'].rolling(30).mean().replace(0, np.nan)
            volume_trend = (vol_10 / vol_30).fillna(1)
            volatility_normalized = (df['atr'] / df['close']).fillna(0)
            
            trend_power = (
                price_momentum.rolling(5).mean().fillna(0) * 0.4 +
                volume_trend.fillna(1) * 0.3 +
                (1 - volatility_normalized) * 0.3
            )
            return trend_power.fillna(0)
        except Exception:
            return pd.Series(0, index=df.index)

# Basit yardımcı fonksiyonlar
def calculate_specter_cloud(df: pd.DataFrame, ma_type: str = "EMA", 
                          base_length: int = 20, atr_multiplier: float = 1.5) -> Optional[pd.DataFrame]:
    specter = SpecterTrendCloud(ma_type, base_length, atr_multiplier)
    return specter.calculate_specter_cloud(df)

def create_ai_snapshot(df: pd.DataFrame, funding_data: dict = None) -> dict:
    """
    AI analizi için indikatör snapshot'ı oluşturur
    """
    snapshot = {}
    if df is None or df.empty:
        return snapshot
    try:
        latest = df.iloc[-1]
        snapshot['price'] = float(latest['close'])
        snapshot['volume'] = float(latest.get('volume', 0))
        # 24 bar yerine mümkünse 24*TF çarpımı gerekir; burada basit 24 bar uygulandı
        snapshot['high_24h'] = float(df['high'].tail(24).max()) if len(df) >= 24 else float(df['high'].max())
        snapshot['low_24h'] = float(df['low'].tail(24).min()) if len(df) >= 24 else float(df['low'].min())
        snapshot['price_change_24h'] = float(((latest['close'] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100) if len(df) >= 24 else 0)
        
        # RSI
        rsi = ta.rsi(df['close'], length=14)
        snapshot['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else 50.0
        
        # MACD
        macd = ta.macd(df['close'])
        if not macd.empty:
            macd_vals = macd.iloc[-1].to_list()
            snapshot['macd'] = float(macd_vals[0]) if len(macd_vals) > 0 else 0.0
            snapshot['macd_signal'] = float(macd_vals[1]) if len(macd_vals) > 1 else 0.0
            snapshot['macd_histogram'] = float(macd_vals[2]) if len(macd_vals) > 2 else 0.0
        
        # Bollinger
        bb = ta.bbands(df['close'], length=20)
        if not bb.empty and bb.shape[1] >= 3:
            snapshot['bb_upper'] = float(bb.iloc[-1, 0])
            snapshot['bb_middle'] = float(bb.iloc[-1, 1])
            snapshot['bb_lower'] = float(bb.iloc[-1, 2])
            snapshot['bb_position'] = float((latest['close'] - bb.iloc[-1, 2]) / max((bb.iloc[-1, 0] - bb.iloc[-1, 2]), 1e-9) * 100)
        
        # ATR
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        snapshot['atr'] = float(atr.iloc[-1]) if not atr.empty else 0.0
        snapshot['atr_percent'] = float((atr.iloc[-1] / latest['close']) * 100) if not atr.empty and latest['close'] != 0 else 0.0
        
        # Volume ratio
        volume_ma = df['volume'].rolling(20).mean()
        snapshot['volume_ratio'] = float(latest['volume'] / volume_ma.iloc[-1]) if not volume_ma.empty and volume_ma.iloc[-1] != 0 else 1.0
        
        # Momentum
        snapshot['momentum_5'] = float((latest['close'] / df['close'].iloc[-5] - 1) * 100) if len(df) > 5 else 0.0
        snapshot['momentum_10'] = float((latest['close'] / df['close'].iloc[-10] - 1) * 100) if len(df) > 10 else 0.0
        
        snapshot['funding_rate'] = funding_data.get('fundingRate', 0.0) if funding_data else 0.0
    except Exception as e:
        logging.error(f"AI snapshot oluşturma hatası: {e}")
    return snapshot

def generate_trading_levels(price: float, trend_direction: str, atr: float, 
                          volatility_regime: str = "NORMAL") -> dict:
    """
    Trend yönüne göre ticaret seviyeleri oluşturur
    """
    if price is None or price == 0:
        return {'entry': 0, 'stop_loss': 0, 'take_profit_1': 0, 'take_profit_2': 0, 'take_profit_3': 0}
    volatility_multipliers = {
        "LOW": 0.8,
        "NORMAL": 1.0,
        "HIGH": 1.3,
        "EXTREME": 1.8
    }
    multiplier = volatility_multipliers.get(volatility_regime, 1.0)
    if trend_direction == "BULLISH":
        entry = price
        stop = price - (atr * 1.5 * multiplier)
        tp1 = price + (atr * 2 * multiplier)
        tp2 = price + (atr * 3 * multiplier)
        tp3 = price + (atr * 4 * multiplier)
        return {'entry': entry, 'stop_loss': stop, 'take_profit_1': tp1, 'take_profit_2': tp2, 'take_profit_3': tp3}
    elif trend_direction == "BEARISH":
        entry = price
        stop = price + (atr * 1.5 * multiplier)
        tp1 = price - (atr * 2 * multiplier)
        tp2 = price - (atr * 3 * multiplier)
        tp3 = price - (atr * 4 * multiplier)
        return {'entry': entry, 'stop_loss': stop, 'take_profit_1': tp1, 'take_profit_2': tp2, 'take_profit_3': tp3}
    else:
        return {'entry': price, 'stop_loss': price - (atr * 2 * multiplier), 'take_profit_1': price + (atr * 2 * multiplier), 'take_profit_2': 0, 'take_profit_3': 0}

def quick_specter_analysis(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    specter_data = calculate_specter_cloud(df)
    if specter_data is None or specter_data.empty:
        return {}
    latest = specter_data.iloc[-1]
    return {
        'trend': "BULLISH" if latest['trend'] == 1 else "BEARISH",
        'trend_strength': abs(latest['momentum_strength']),
        'has_bullish_retest': bool(latest.get('bullish_retest', False)),
        'has_bearish_retest': bool(latest.get('bearish_retest', False)),
        'cloud_top': float(latest.get('ma_upper', 0)),
        'cloud_bottom': float(latest.get('ma_lower', 0)),
        'momentum': float(latest.get('momentum_strength', 0))
    }
