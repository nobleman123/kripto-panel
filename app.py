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

# Sayfa ayarı
st.set_page_config(
    page_title="Profesyonel Kripto Sinyal Sistemi",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .signal-buy {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .signal-sell {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
    }
    .signal-hold {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        border-left: 4px solid #1f77b4;
    }
    .indicator-positive {
        color: #28a745;
        font-weight: bold;
    }
    .indicator-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .indicator-neutral {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class CryptoSignalApp:
    def __init__(self):
        self.exchanges = {
            'binance': 'Binance',
            'mexc': 'MEXC'
        }
        
    def get_high_volume_pairs(self, exchange='binance', volume_threshold=1000000, limit=50):
        """Yüksek hacimli çiftleri getir"""
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
                if 'symbol' in ticker and ticker['symbol'].endswith('USDT'):
                    volume = float(ticker.get('quoteVolume', 0))
                    if volume > volume_threshold:
                        high_volume_pairs.append({
                            'symbol': ticker['symbol'],
                            'volume': volume,
                            'price': float(ticker.get('lastPrice', 0)),
                            'change': float(ticker.get('priceChangePercent', 0))
                        })
            
            # Hacime göre sırala ve limit uygula
            high_volume_pairs.sort(key=lambda x: x['volume'], reverse=True)
            return high_volume_pairs[:limit]
            
        except Exception as e:
            st.error(f"Veri çekme hatası: {e}")
            return []
    
    def fetch_ohlcv_data(self, symbol, exchange='binance', interval='1h', limit=100):
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
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Veri tiplerini dönüştür
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            st.error(f"{symbol} veri çekme hatası: {e}")
            return pd.DataFrame()
    
    def calculate_all_indicators(self, df):
        """8 temel göstergeyi hesapla"""
        if df.empty or len(df) < 50:
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
            indicators['macd_histogram'] = macd.macd_diff().iloc[-1]
            
            # 3. Bollinger Bantları
            bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
            indicators['bb_upper'] = bb.bollinger_hband().iloc[-1]
            indicators['bb_lower'] = bb.bollinger_lband().iloc[-1]
            indicators['bb_middle'] = bb.bollinger_mavg().iloc[-1]
            current_price = close.iloc[-1]
            bb_position = (current_price - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            indicators['bb_position'] = bb_position
            
            # 4. EMA'lar
            indicators['ema_9'] = ta.trend.EMAIndicator(close=close, window=9).ema_indicator().iloc[-1]
            indicators['ema_21'] = ta.trend.EMAIndicator(close=close, window=21).ema_indicator().iloc[-1]
            indicators['ema_50'] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]
            
            # 5. SMA'lar
            indicators['sma_20'] = ta.trend.SMAIndicator(close=close, window=20).sma_indicator().iloc[-1]
            indicators['sma_50'] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator().iloc[-1]
            
            # 6. VWAP (Basitleştirilmiş)
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            indicators['vwap'] = vwap.iloc[-1]
            
            # 7. Stochastic RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(close=close)
            indicators['stoch_rsi'] = stoch_rsi.stochrsi().iloc[-1]
            
            # 8. OBV (On Balance Volume)
            indicators['obv'] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume().iloc[-1]
            
            # Ek göstergeler
            indicators['current_price'] = current_price
            indicators['volume_change'] = volume.pct_change().iloc[-1] if not volume.pct_change().empty else 0
            
        except Exception as e:
            st.error(f"Gösterge hesaplama hatası: {e}")
            return {}
        
        return indicators
    
    def generate_signal(self, indicators):
        """Göstergelere göre sinyal üret"""
        if not indicators:
            return 'NÖTR', 0
        
        buy_signals = 0
        sell_signals = 0
        total_indicators = 0
        
        # RSI Sinyali
        if indicators['rsi'] < 30:
            buy_signals += 1
        elif indicators['rsi'] > 70:
            sell_signals += 1
        total_indicators += 1
        
        # MACD Sinyali
        if indicators['macd'] > indicators['macd_signal']:
            buy_signals += 1
        else:
            sell_signals += 1
        total_indicators += 1
        
        # Bollinger Bantları Sinyali
        if indicators['bb_position'] < 0.2:  # Alt bant yakını
            buy_signals += 1
        elif indicators['bb_position'] > 0.8:  # Üst bant yakını
            sell_signals += 1
        total_indicators += 1
        
        # EMA Sinyali
        if indicators['ema_9'] > indicators['ema_21']:
            buy_signals += 1
        else:
            sell_signals += 1
        total_indicators += 1
        
        # VWAP Sinyali
        if indicators['current_price'] > indicators['vwap']:
            buy_signals += 1
        else:
            sell_signals += 1
        total_indicators += 1
        
        # Stochastic RSI Sinyali
        if indicators['stoch_rsi'] < 0.2:
            buy_signals += 1
        elif indicators['stoch_rsi'] > 0.8:
            sell_signals += 1
        total_indicators += 1
        
        # Güven skoru
        confidence = max(buy_signals, sell_signals) / total_indicators * 100
        
        if buy_signals >= 4:
            return 'GÜÇLÜ AL', confidence
        elif buy_signals >= 3:
            return 'ZAYIF AL', confidence
        elif sell_signals >= 4:
            return 'GÜÇLÜ SAT', confidence
        elif sell_signals >= 3:
            return 'ZAYIF SAT', confidence
        else:
            return 'NÖTR', confidence
    
    def create_technical_chart(self, df, indicators, symbol):
        """Teknik analiz grafiği oluştur"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Fiyat ve Göstergeler', 'MACD', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Fiyat grafiği
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Fiyat'
            ),
            row=1, col=1
        )
        
        # Bollinger Bantları
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=[indicators['bb_upper']] * len(df),
                line=dict(color='rgba(255,0,0,0.3)'),
                name='BB Üst'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=[indicators['bb_lower']] * len(df),
                line=dict(color='rgba(0,255,0,0.3)'),
                name='BB Alt'
            ),
            row=1, col=1
        )
        
        # MACD
        macd_line = ta.trend.MACD(close=df['close']).macd()
        macd_signal = ta.trend.MACD(close=df['close']).macd_signal()
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=macd_line, name='MACD', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=macd_signal, name='Sinyal', line=dict(color='red')),
            row=2, col=1
        )
        
        # RSI
        rsi = ta.momentum.RSIIndicator(close=df['close']).rsi()
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=rsi, name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        
        # RSI seviye çizgileri
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=800,
            title_text=f"{symbol} Teknik Analiz",
            xaxis_rangeslider_visible=False
        )
        
        return fig

def main():
    app = CryptoSignalApp()
    
    # Başlık
    st.markdown('<h1 class="main-header">🚀 Profesyonel Kripto Sinyal Sistemi</h1>', unsafe_allow_html=True)
    
    # Sidebar - Kontroller
    with st.sidebar:
        st.header("⚙️ Kontroller")
        
        exchange = st.selectbox(
            "Borsa Seçin:",
            options=list(app.exchanges.keys()),
            format_func=lambda x: app.exchanges[x]
        )
        
        volume_threshold = st.number_input(
            "Minimum Hacim ($):",
            min_value=100000,
            value=1000000,
            step=100000
        )
        
        analysis_limit = st.slider(
            "Analiz Edilecek Çift Sayısı:",
            min_value=10,
            max_value=200,
            value=50
        )
        
        interval = st.selectbox(
            "Zaman Aralığı:",
            options=['1h', '4h', '1d', '1w'],
            index=0
        )
        
        auto_refresh = st.checkbox("Otomatik Yenile", value=False)
        refresh_interval = st.number_input("Yenileme Aralığı (saniye):", min_value=10, value=60)
    
    # Ana içerik
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Sinyal Tara", use_container_width=True):
            with st.spinner("Yüksek hacimli çiftler taranıyor..."):
                # Yüksek hacimli çiftleri getir
                pairs = app.get_high_volume_pairs(
                    exchange=exchange,
                    volume_threshold=volume_threshold,
                    limit=analysis_limit
                )
                
                if pairs:
                    st.success(f"{len(pairs)} yüksek hacimli çift bulundu!")
                    
                    # Sinyal analizi
                    signals = []
                    progress_bar = st.progress(0)
                    
                    for i, pair in enumerate(pairs):
                        symbol = pair['symbol']
                        
                        # OHLCV verilerini getir
                        df = app.fetch_ohlcv_data(symbol, exchange, interval)
                        
                        if not df.empty:
                            # Göstergeleri hesapla
                            indicators = app.calculate_all_indicators(df)
                            
                            if indicators:
                                # Sinyal üret
                                signal, confidence = app.generate_signal(indicators)
                                
                                if signal != 'NÖTR':
                                    signals.append({
                                        'Sembol': symbol,
                                        'Sinyal': signal,
                                        'Güven': f"%{confidence:.1f}",
                                        'Fiyat': f"${indicators['current_price']:.4f}",
                                        'RSI': f"{indicators['rsi']:.2f}",
                                        'MACD': f"{indicators['macd']:.4f}",
                                        'Volume': f"${pair['volume']:,.0f}"
                                    })
                        
                        # Progress bar güncelle
                        progress_bar.progress((i + 1) / len(pairs))
                    
                    # Sonuçları göster
                    if signals:
                        st.subheader("📊 Sinyal Sonuçları")
                        
                        # Sinyalleri DataFrame'e dönüştür
                        signals_df = pd.DataFrame(signals)
                        
                        # Sinyal renklendirme
                        def color_signal(val):
                            if 'AL' in val:
                                return 'color: green; font-weight: bold'
                            elif 'SAT' in val:
                                return 'color: red; font-weight: bold'
                            else:
                                return 'color: orange; font-weight: bold'
                        
                        styled_df = signals_df.style.applymap(color_signal, subset=['Sinyal'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Detaylı analiz için seçim
                        st.subheader("🔍 Detaylı Analiz")
                        selected_symbol = st.selectbox(
                            "Sembol Seçin:",
                            options=[s['Sembol'] for s in signals]
                        )
                        
                        if selected_symbol:
                            # Seçilen sembol için detaylı analiz
                            df_detail = app.fetch_ohlcv_data(selected_symbol, exchange, '1h', 100)
                            indicators_detail = app.calculate_all_indicators(df_detail)
                            
                            if indicators_detail:
                                # Gösterge kartları
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    rsi_color = "indicator-positive" if indicators_detail['rsi'] < 30 else "indicator-negative" if indicators_detail['rsi'] > 70 else "indicator-neutral"
                                    st.metric("RSI", f"{indicators_detail['rsi']:.2f}", delta=None, delta_color="off")
                                    st.markdown(f'<p class="{rsi_color}">{"Aşırı Satım" if indicators_detail["rsi"] < 30 else "Aşırı Alım" if indicators_detail["rsi"] > 70 else "Nötr"}</p>', unsafe_allow_html=True)
                                
                                with col2:
                                    macd_color = "indicator-positive" if indicators_detail['macd'] > indicators_detail['macd_signal'] else "indicator-negative"
                                    st.metric("MACD", f"{indicators_detail['macd']:.4f}", delta=None)
                                    st.markdown(f'<p class="{macd_color}">{"Yükseliş" if indicators_detail["macd"] > indicators_detail["macd_signal"] else "Düşüş"}</p>', unsafe_allow_html=True)
                                
                                with col3:
                                    bb_color = "indicator-positive" if indicators_detail['bb_position'] < 0.2 else "indicator-negative" if indicators_detail['bb_position'] > 0.8 else "indicator-neutral"
                                    st.metric("BB Pozisyon", f"%{indicators_detail['bb_position']*100:.1f}", delta=None)
                                    st.markdown(f'<p class="{bb_color}">{"Alt Bant" if indicators_detail["bb_position"] < 0.2 else "Üst Bant" if indicators_detail["bb_position"] > 0.8 else "Orta"}</p>', unsafe_allow_html=True)
                                
                                with col4:
                                    st.metric("VWAP", f"${indicators_detail['vwap']:.4f}", delta=None)
                                    vwap_status = "Üzerinde" if indicators_detail['current_price'] > indicators_detail['vwap'] else "Altında"
                                    status_color = "indicator-positive" if vwap_status == "Altında" else "indicator-negative"
                                    st.markdown(f'<p class="{status_color}">{vwap_status}</p>', unsafe_allow_html=True)
                                
                                # Grafik göster
                                st.plotly_chart(
                                    app.create_technical_chart(df_detail, indicators_detail, selected_symbol),
                                    use_container_width=True
                                )
                    else:
                        st.warning("Hiç sinyal bulunamadı. Filtreleri değiştirmeyi deneyin.")
                
                else:
                    st.error("Yüksek hacimli çift bulunamadı. Filtreleri değiştirmeyi deneyin.")
    
    # Otomatik yenileme
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
