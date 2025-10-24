# trend_cloud.py
# Specter Trend Cloud - Gelişmiş (Strateji Ayrımlı) Versiyon

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import Optional

class SpecterTrendCloud:
    def __init__(self, ma_type: str = "EMA", base_length: int = 20, atr_multiplier: float = 1.5):
        self.ma_type = (ma_type or "EMA").upper()
        self.base_length = max(2, int(base_length))
        self.atr_multiplier = float(atr_multiplier)
        self.short_length = self.base_length
        self.long_length = max(3, self.base_length * 2)
    
    def calculate_ma(self, series: pd.Series, length: int) -> pd.Series:
        try:
            if self.ma_type == "SMA":
                return ta.sma(series, length=length)
            elif self.ma_type == "WMA":
                return ta.wma(series, length=length)
            elif self.ma_type == "DEMA":
                return ta.dema(series, length=length)
            else:
                return ta.ema(series, length=length)
        except Exception as e:
            logging.warning(f"MA hesaplama hatası: {e}")
            return series.rolling(length, min_periods=1).mean()
    
    def calculate_specter_cloud(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        df = df.copy().reset_index(drop=True)
        try:
            df['ma_short'] = self.calculate_ma(df['close'], self.short_length)
            df['ma_long'] = self.calculate_ma(df['close'], self.long_length)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14).fillna(0)
            df['trend'] = np.where(df['ma_short'] > df['ma_long'], 1, -1)
            df['atr_offset'] = df['atr'] * self.atr_multiplier
            df['ma_upper'] = np.where(df['trend'] == 1, df['ma_short'] + df['atr_offset'], df['ma_long'] + df['atr_offset'])
            df['ma_lower'] = np.where(df['trend'] == 1, df['ma_long'] - df['atr_offset'], df['ma_short'] - df['atr_offset'])
            df['momentum_strength'] = ((df['ma_short'] - df['ma_long']) / df['ma_long'].replace(0, np.nan)).fillna(0) * 100
            df = self._calculate_retest_signals(df)
            df['trend_power'] = self._calculate_trend_power(df)
            df['timestamp'] = df.get('timestamp', pd.to_datetime(df.index, unit='m'))
            return df.reset_index(drop=True)
        except Exception as e:
            logging.error(f"Specter hesaplama hatası: {e}")
            return None
    
    def _calculate_retest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['bullish_retest'] = False
        df['bearish_retest'] = False
        cooldown = 5
        for i in range(cooldown, len(df)):
            try:
                cur = df.iloc[i]
                prev = df.iloc[i-1]
                if cur['trend'] == 1 and prev['trend'] == 1:
                    if cur['low'] <= cur['ma_short'] <= cur['high']:
                        if not df['bullish_retest'].iloc[i-cooldown:i].any():
                            df.at[i, 'bullish_retest'] = True
                if cur['trend'] == -1 and prev['trend'] == -1:
                    if cur['high'] >= cur['ma_short'] >= cur['low']:
                        if not df['bearish_retest'].iloc[i-cooldown:i].any():
                            df.at[i, 'bearish_retest'] = True
            except Exception:
                continue
        return df
    
    def _calculate_trend_power(self, df: pd.DataFrame) -> pd.Series:
        try:
            price_mom = df['close'].pct_change(5).fillna(0)
            vol_10 = df['volume'].rolling(10).mean().fillna(0)
            vol_30 = df['volume'].rolling(30).mean().replace(0, np.nan).fillna(1)
            volume_trend = (vol_10 / vol_30).fillna(1)
            vol_norm = (df['atr'] / df['close']).fillna(0)
            trend_power = (price_mom.rolling(5).mean().fillna(0) * 0.45 +
                           volume_trend.fillna(1) * 0.35 +
                           (1 - vol_norm) * 0.2)
            return trend_power.fillna(0)
        except Exception:
            return pd.Series(0, index=df.index)

# Helper wrapper
def calculate_specter_cloud(df: pd.DataFrame, ma_type="EMA", base_length=20, atr_multiplier=1.5):
    s = SpecterTrendCloud(ma_type, base_length, atr_multiplier)
    return s.calculate_specter_cloud(df)

def quick_specter_analysis(df: pd.DataFrame):
    if df is None or df.empty:
        return {}
    specter = calculate_specter_cloud(df)
    if specter is None or specter.empty:
        return {}
    latest = specter.iloc[-1]
    return {
        'trend': "BULLISH" if latest['trend'] == 1 else "BEARISH",
        'trend_strength': float(abs(latest.get('momentum_strength', 0))),
        'has_bullish_retest': bool(latest.get('bullish_retest', False)),
        'has_bearish_retest': bool(latest.get('bearish_retest', False)),
        'cloud_top': float(latest.get('ma_upper', 0)),
        'cloud_bottom': float(latest.get('ma_lower', 0)),
        'momentum': float(latest.get('momentum_strength', 0)),
        'atr': float(latest.get('atr', 0))
    }
