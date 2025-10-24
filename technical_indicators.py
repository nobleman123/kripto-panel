# technical_indicators.py
# Çoklu Teknik İndikatör Hesaplama Sistemi

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional
import logging

class AdvancedTechnicalIndicators:
    """
    Gelişmiş Çoklu İndikatör Sistemi
    20+ teknik indikatör ile kapsamlı analiz
    """
    
    def __init__(self):
        self.indicators_config = {
            'moving_averages': [5, 10, 20, 50, 100, 200],
            'rsi_periods': [6, 14, 24],
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'stoch_period': 14,
            'atr_period': 14
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Tüm teknik indikatörleri hesaplar
        """
        try:
            if df.empty or len(df) < 100:
                return None
            
            result_df = df.copy()
            
            # 1. Moving Averages
            result_df = self._calculate_moving_averages(result_df)
            
            # 2. Oscillators
            result_df = self._calculate_oscillators(result_df)
            
            # 3. Momentum Indicators
            result_df = self._calculate_momentum_indicators(result_df)
            
            # 4. Volatility Indicators
            result_df = self._calculate_volatility_indicators(result_df)
            
            # 5. Volume Indicators
            result_df = self._calculate_volume_indicators(result_df)
            
            # 6. Trend Analysis
            result_df = self._calculate_trend_analysis(result_df)
            
            # 7. Signal Strength
            result_df = self._calculate_signal_strength(result_df)
            
            return result_df.dropna()
            
        except Exception as e:
            logging.error(f"İndikatör hesaplama hatası: {str(e)}")
            return None
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Moving average hesaplamaları"""
        periods = self.indicators_config['moving_averages']
        
        for period in periods:
            # EMA
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)
            # SMA
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
            # WMA
            df[f'wma_{period}'] = ta.wma(df['close'], length=period)
        
        # EMA Alignment Score
        df['ema_alignment'] = self._calculate_ema_alignment(df)
        
        return df
    
    def _calculate_ema_alignment(self, df: pd.DataFrame) -> pd.Series:
        """EMA dizilim skoru hesaplar"""
        alignment_score = np.zeros(len(df))
        
        for i in range(len(df)):
            score = 0
            # EMA 20 > EMA 50 > EMA 200 (Bullish)
            if (df['ema_20'].iloc[i] > df['ema_50'].iloc[i] > df['ema_200'].iloc[i]):
                score += 1
            # EMA 20 < EMA 50 < EMA 200 (Bearish)
            elif (df['ema_20'].iloc[i] < df['ema_50'].iloc[i] < df['ema_200'].iloc[i]):
                score -= 1
            
            # Kısa vadeli EMA'ların sıralaması
            if (df['ema_5'].iloc[i] > df['ema_10'].iloc[i] > df['ema_20'].iloc[i]):
                score += 0.5
            elif (df['ema_5'].iloc[i] < df['ema_10'].iloc[i] < df['ema_20'].iloc[i]):
                score -= 0.5
                
            alignment_score[i] = score
        
        return pd.Series(alignment_score, index=df.index)
    
    def _calculate_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Osilatör indikatörleri"""
        # RSI
        for period in self.indicators_config['rsi_periods']:
            df[f'rsi_{period}'] = ta.rsi(df['close'], length=period)
        
        # Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'], 
                        k=self.indicators_config['stoch_period'],
                        d=3)
        if stoch is not None:
            df['stoch_k'] = stoch['STOCHk_' + str(self.indicators_config['stoch_period'])]
            df['stoch_d'] = stoch['STOCHd_' + str(self.indicators_config['stoch_period'])]
        
        # Williams %R
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # CCI
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=20)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indikatörleri"""
        # MACD
        macd = ta.macd(df['close'], 
                      fast=self.indicators_config['macd_fast'],
                      slow=self.indicators_config['macd_slow'],
                      signal=self.indicators_config['macd_signal'])
        if macd is not None:
            df['macd'] = macd['MACD_' + str(self.indicators_config['macd_fast']) + '_' + 
                             str(self.indicators_config['macd_slow']) + '_' + 
                             str(self.indicators_config['macd_signal'])]
            df['macd_signal'] = macd['MACDs_' + str(self.indicators_config['macd_fast']) + '_' + 
                                   str(self.indicators_config['macd_slow']) + '_' + 
                                   str(self.indicators_config['macd_signal'])]
            df['macd_histogram'] = macd['MACDh_' + str(self.indicators_config['macd_fast']) + '_' + 
                                      str(self.indicators_config['macd_slow']) + '_' + 
                                      str(self.indicators_config['macd_signal'])]
        
        # ADX
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['adx'] = adx['ADX_14']
            df['di_plus'] = adx['DMP_14']
            df['di_minus'] = adx['DMN_14']
        
        # Momentum
        df['momentum_10'] = ta.mom(df['close'], length=10)
        df['roc_10'] = ta.roc(df['close'], length=10)
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatilite indikatörleri"""
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=self.indicators_config['bb_period'])
        if bb is not None:
            df['bb_upper'] = bb['BBU_' + str(self.indicators_config['bb_period']) + '_2.0']
            df['bb_middle'] = bb['BBM_' + str(self.indicators_config['bb_period']) + '_2.0']
            df['bb_lower'] = bb['BBL_' + str(self.indicators_config['bb_period']) + '_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], 
                          length=self.indicators_config['atr_period'])
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Keltner Channel
        kc = ta.kc(df['high'], df['low'], df['close'], length=20)
        if kc is not None:
            df['kc_upper'] = kc['KCUe_20_2']
            df['kc_lower'] = kc['KCLe_20_2']
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hacim indikatörleri"""
        # OBV
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Volume SMA
        df['volume_sma_20'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # MFI
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        
        # VWAP
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        return df
    
    def _calculate_trend_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend analizi"""
        # Ichimoku Cloud
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        if ichimoku is not None:
            df['ichimoku_a'] = ichimoku['ITS_9']
            df['ichimoku_b'] = ichimoku['IKS_26']
        
        # Parabolic SAR
        df['psar'] = ta.psar(df['high'], df['low'], df['close'])
        
        # Trend Strength
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Primary Trend
        df['primary_trend'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)
        
        return df
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Trend gücü hesaplama"""
        strength = np.zeros(len(df))
        
        for i in range(len(df)):
            score = 0
            
            # ADX trend gücü
            if df['adx'].iloc[i] > 25:
                score += 1
            if df['adx'].iloc[i] > 40:
                score += 1
            if df['adx'].iloc[i] > 60:
                score += 1
            
            # EMA trend gücü
            ema_distance = abs(df['ema_20'].iloc[i] - df['ema_50'].iloc[i]) / df['close'].iloc[i]
            if ema_distance > 0.02:
                score += 1
            if ema_distance > 0.05:
                score += 1
            
            # Volume confirmation
            if df['volume_ratio'].iloc[i] > 1.5:
                score += 1
            
            strength[i] = min(score / 6, 1.0)  # Normalize to 0-1
        
        return pd.Series(strength, index=df.index)
    
    def _calculate_signal_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sinyal gücü hesaplama"""
        signal_strength = np.zeros(len(df))
        momentum_score = np.zeros(len(df))
        
        for i in range(len(df)):
            # RSI sinyali
            rsi_signal = 0
            if df['rsi_14'].iloc[i] < 30:
                rsi_signal = 1
            elif df['rsi_14'].iloc[i] > 70:
                rsi_signal = -1
            
            # MACD sinyali
            macd_signal = 0
            if df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
                macd_signal = 1
            elif df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
                macd_signal = -1
            
            # Stochastic sinyali
            stoch_signal = 0
            if df['stoch_k'].iloc[i] < 20 and df['stoch_d'].iloc[i] < 20:
                stoch_signal = 1
            elif df['stoch_k'].iloc[i] > 80 and df['stoch_d'].iloc[i] > 80:
                stoch_signal = -1
            
            # Toplam sinyal gücü
            total_signal = (rsi_signal + macd_signal + stoch_signal) / 3
            signal_strength[i] = total_signal
            
            # Momentum skoru
            momentum_score[i] = (rsi_signal + macd_signal + stoch_signal + df['ema_alignment'].iloc[i]) / 4
        
        df['signal_strength'] = signal_strength
        df['momentum_score'] = momentum_score
        df['volatility_state'] = np.where(df['atr_percent'] > 5, 'HIGH', 
                                         np.where(df['atr_percent'] > 2, 'MEDIUM', 'LOW'))
        
        return df

def calculate_all_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Tüm teknik indikatörleri hesaplar
    """
    indicator = AdvancedTechnicalIndicators()
    return indicator.calculate_all_indicators(df)

def create_ai_snapshot(df: pd.DataFrame, funding_data: Dict, technical_analysis: Dict) -> Dict[str, Any]:
    """
    AI analizi için indikatör snapshot'ı oluşturur
    """
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    # Temel fiyat verileri
    snapshot = {
        'price': float(latest['close']),
        'volume': float(latest['volume']),
        'high_24h': float(df['high'].tail(24).max()),
        'low_24h': float(df['low'].tail(24).min()),
        'price_change_24h': float((latest['close'] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100)
    }
    
    # Moving Averages
    ma_columns = [col for col in df.columns if col.startswith(('ema_', 'sma_'))]
    for col in ma_columns:
        snapshot[col] = float(latest[col]) if col in latest else 0.0
    
    # Oscillators
    oscillator_columns = [col for col in df.columns if col.startswith(('rsi_', 'stoch_', 'williams_', 'cci'))]
    for col in oscillator_columns:
        snapshot[col] = float(latest[col]) if col in latest else 50.0
    
    # Momentum Indicators
    momentum_columns = [col for col in df.columns if col.startswith(('macd', 'adx', 'momentum', 'roc'))]
    for col in momentum_columns:
        snapshot[col] = float(latest[col]) if col in latest else 0.0
    
    # Volatility Indicators
    volatility_columns = [col for col in df.columns if col.startswith(('bb_', 'atr', 'kc_'))]
    for col in volatility_columns:
        snapshot[col] = float(latest[col]) if col in latest else 0.0
    
    # Volume Indicators
    volume_columns = [col for col in df.columns if col.startswith(('volume_', 'obv', 'mfi', 'vwap'))]
    for col in volume_columns:
        snapshot[col] = float(latest[col]) if col in latest else 0.0
    
    # Trend Analysis
    trend_columns = ['trend_strength', 'primary_trend', 'signal_strength', 'momentum_score', 'volatility_state']
    for col in trend_columns:
        if col in latest:
            if col == 'volatility_state':
                snapshot[col] = latest[col]
            else:
                snapshot[col] = float(latest[col])
    
    # Funding rate
    snapshot['funding_rate'] = funding_data.get('fundingRate', 0.0)
    
    # Technical analysis summary
    snapshot['technical_summary'] = {
        'trend': technical_analysis.get('trend', 'NEUTRAL'),
        'trend_strength': technical_analysis.get('trend_strength', 0),
        'signal_strength': technical_analysis.get('signal_strength', 0),
        'momentum': technical_analysis.get('momentum', 0),
        'volatility': technical_analysis.get('volatility', 'MEDIUM')
    }
    
    return snapshot

# Hızlı analiz fonksiyonu
def quick_technical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Hızlı teknik analiz için basit fonksiyon
    """
    if df.empty:
        return {}
    
    indicator_data = calculate_all_indicators(df)
    if indicator_data is None:
        return {}
    
    latest = indicator_data.iloc[-1]
    
    return {
        'trend': "BULLISH" if latest['primary_trend'] == 1 else "BEARISH",
        'trend_strength': latest['trend_strength'],
        'signal_strength': latest['signal_strength'],
        'momentum': latest['momentum_score'],
        'volatility': latest['volatility_state']
    }
