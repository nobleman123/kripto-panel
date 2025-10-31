# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sayfa ayarı - daha basit konfigürasyon
st.set_page_config(
    page_title="Kripto Sinyal Sistemi",
    page_icon="🚀",
    layout="wide"
)

# Basitleştirilmiş CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .signal-buy {
        background-color: #d4edda;
        border-radius: 5px;
        padding: 8px;
        color: #155724;
    }
    .signal-sell {
        background-color: #f8d7da;
        border-radius: 5px;
        padding: 8px;
        color: #721c24;
    }
    .signal-hold {
        background-color: #fff3cd;
        border-radius: 5px;
        padding: 8px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

class CryptoSignalApp:
    def __init__(self):
        self.exchanges = {
            'binance': 'Binance',
            'mexc': 'MEXC'
        }
        
    def get_high_volume_pairs(self, exchange='binance', volume_threshold=1000000, limit=20):
        """Yüksek hacimli çiftleri getir - limiti düşürdüm"""
        try:
            if exchange == 'binance':
                url = "https://api.binance.com/api/v3/ticker/24hr"
            elif exchange == 'mexc':
                url = "https://api.mexc.com/api/v3/ticker/24hr"
            else:
                return []
            
            response = requests.get(url, timeout=10)
            tickers = response.json()
            high_volume_pairs = []
            
            for ticker in tickers:
                if isinstance(ticker, dict) and 'symbol' in ticker and ticker['symbol'].endswith('USDT'):
                    try:
                        volume = float(ticker.get('quoteVolume', 0))
                        if volume > volume_threshold:
                            high_volume_pairs.append({
                                'symbol': ticker['symbol'],
                                'volume': volume,
                                'price': float(ticker.get('lastPrice', 0)),
                                'change': float(ticker.get('priceChangePercent', 0))
                            })
                    except (ValueError, TypeError):
                        continue
            
            # Hacime göre sırala ve limit uygula
            high_volume_pairs.sort(key=lambda x: x['volume'], reverse=True)
            return high_volume_pairs[:limit]
            
        except Exception as e:
            st.error(f"Veri çekme hatası: {e}")
            return []
    
    def fetch_ohlcv_data(self, symbol, exchange='binance', interval='1h', limit=50):  # Limit düşürüldü
        """OHLCV verilerini getir"""
        try:
            if exchange == 'binance':
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            elif exchange == 'mexc':
                url = f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            else:
                return pd.DataFrame()
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not isinstance(data, list):
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Veri tiplerini dönüştür
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            df = df.dropna()
            return df
            
        except Exception as e:
            return pd.DataFrame()
    
    def calculate_all_indicators(self, df):
        """8 temel göstergeyi hesapla"""
        if df.empty or len(df) < 20:  # Minimum veri sayısını düşürdüm
            return {}
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        indicators = {}
        
        try:
            # 1. RSI
            indicators['rsi'] = ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1]
            
            # 2. MACD
            macd = ta.trend.MACD(close=close)
            indicators['macd'] = macd.macd().iloc[-1]
            indicators['macd_signal'] = macd.macd_signal().iloc[-1]
            
            # 3. Bollinger Bantları
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            current_price = close.iloc[-1]
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            if bb_upper != bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5
            indicators['bb_position'] = bb_position
            
            # 4. EMA'lar
            indicators['ema_9'] = ta.trend.EMAIndicator(close=close, window=9).ema_indicator().iloc[-1]
            indicators['ema_21'] = ta.trend.EMAIndicator(close=close, window=21).ema_indicator().iloc[-1]
            
            # 5. VWAP (Basitleştirilmiş)
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            indicators['vwap'] = vwap.iloc[-1]
            
            # 6. Stochastic RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(close=close)
            stoch_rsi_val = stoch_rsi.stochrsi().iloc[-1]
            indicators['stoch_rsi'] = stoch_rsi_val if not pd.isna(stoch_rsi_val) else 0.5
            
            # Ek göstergeler
            indicators['current_price'] = current_price
            
        except Exception as e:
            return {}
        
        return indicators
    
    def generate_signal(self, indicators):
        """Göstergelere göre sinyal üret"""
        if not indicators:
            return 'NÖTR', 0
        
        buy_signals = 0
        sell_signals = 0
        
        # RSI Sinyali
        if indicators['rsi'] < 30:
            buy_signals += 1
        elif indicators['rsi'] > 70:
            sell_signals += 1
        
        # MACD Sinyali
        if indicators['macd'] > indicators['macd_signal']:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Bollinger Bantları Sinyali
        if indicators['bb_position'] < 0.2:
            buy_signals += 1
        elif indicators['bb_position'] > 0.8:
            sell_signals += 1
        
        # EMA Sinyali
        if indicators['ema_9'] > indicators['ema_21']:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # VWAP Sinyali
        if indicators['current_price'] > indicators['vwap']:
            buy_signals += 1
        else:
            sell_signals += 1
        
        # Toplam sinyal sayısı
        total_signals = 5
        
        # Güven skoru
        confidence = max(buy_signals, sell_signals) / total_signals * 100
        
        if buy_signals >= 3:
            return 'AL', confidence
        elif sell_signals >= 3:
            return 'SAT', confidence
        else:
            return 'NÖTR', confidence

def main():
    app = CryptoSignalApp()
    
    # Başlık
    st.markdown('<h1 class="main-header">🚀 Kripto Sinyal Sistemi</h1>', unsafe_allow_html=True)
    
    # Sidebar - Basitleştirilmiş kontroller
    with st.sidebar:
        st.header("⚙️ Ayarlar")
        
        exchange = st.selectbox(
            "Borsa:",
            options=list(app.exchanges.keys()),
            format_func=lambda x: app.exchanges[x]
        )
        
        volume_threshold = st.selectbox(
            "Minimum Hacim:",
            options=[100000, 500000, 1000000, 5000000],
            index=2
        )
        
        analysis_limit = st.slider(
            "Analiz Sayısı:",
            min_value=5,
            max_value=30,
            value=15
        )

    # Ana içerik
    if st.button("🔍 Sinyalleri Tara", type="primary", use_container_width=True):
        with st.spinner("Çiftler taranıyor..."):
            # Yüksek hacimli çiftleri getir
            pairs = app.get_high_volume_pairs(
                exchange=exchange,
                volume_threshold=volume_threshold,
                limit=analysis_limit
            )
            
            if pairs:
                st.success(f"{len(pairs)} çift bulundu!")
                
                # Sinyal analizi
                signals = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, pair in enumerate(pairs):
                    symbol = pair['symbol']
                    status_text.text(f"Analiz: {symbol}")
                    
                    # OHLCV verilerini getir
                    df = app.fetch_ohlcv_data(symbol, exchange, '1h')
                    
                    if not df.empty:
                        # Göstergeleri hesapla
                        indicators = app.calculate_all_indicators(df)
                        
                        if indicators:
                            # Sinyal üret
                            signal, confidence = app.generate_signal(indicators)
                            
                            signals.append({
                                'Sembol': symbol,
                                'Sinyal': signal,
                                'Güven': f"%{confidence:.0f}",
                                'Fiyat': f"${indicators['current_price']:.4f}",
                                'RSI': f"{indicators['rsi']:.1f}",
                                'Volume': f"${pair['volume']:,.0f}"
                            })
                    
                    # Progress bar güncelle
                    progress_bar.progress((i + 1) / len(pairs))
                
                status_text.empty()
                progress_bar.empty()
                
                # Sonuçları göster
                if signals:
                    st.subheader("📊 Sinyal Sonuçları")
                    
                    # Sinyalleri DataFrame'e dönüştür
                    signals_df = pd.DataFrame(signals)
                    
                    # Renk fonksiyonu
                    def color_row(row):
                        if row['Sinyal'] == 'AL':
                            return ['background-color: #d4edda'] * len(row)
                        elif row['Sinyal'] == 'SAT':
                            return ['background-color: #f8d7da'] * len(row)
                        else:
                            return ['background-color: #fff3cd'] * len(row)
                    
                    # DataFrame'i göster
                    st.dataframe(
                        signals_df.style.apply(color_row, axis=1),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Sadece AL sinyallerini göster
                    buy_signals = [s for s in signals if s['Sinyal'] == 'AL']
                    if buy_signals:
                        st.subheader("🎯 AL Sinyalleri")
                        for signal in buy_signals:
                            st.info(f"**{signal['Sembol']}** - Güven: {signal['Güven']} - Fiyat: {signal['Fiyat']}")
                    
                else:
                    st.warning("Hiç sinyal bulunamadı.")
                
                # Analiz verilerini session state'e kaydet
                st.session_state.analysis_data = signals
                
            else:
                st.error("Çift bulunamadı. Ayarları değiştirin.")

    # Önceki analizi göster
    if st.session_state.analysis_data:
        st.subheader("📈 Önceki Analiz")
        signals_df = pd.DataFrame(st.session_state.analysis_data)
        st.dataframe(signals_df, use_container_width=True)

if __name__ == "__main__":
    main()
