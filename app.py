# app.py
# MEXC Profesyonel Sinyal Paneli - Çalışan Versiyon

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
import streamlit.components.v1 as components
import plotly.graph_objects as go
import json
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit sayfa ayarı
st.set_page_config(
    page_title="MEXC Pro Sinyal Terminali",
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# ---------------- CONFIG & STYLING ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']

# Modern CSS Styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .signal-card {
        background: rgba(30, 41, 59, 0.9);
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .signal-strong-long {
        border-left-color: #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(30, 41, 59, 0.9));
    }
    .signal-long {
        border-left-color: #22c55e;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(30, 41, 59, 0.9));
    }
    .signal-neutral {
        border-left-color: #6b7280;
    }
    .signal-short {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(30, 41, 59, 0.9));
    }
    .signal-strong-short {
        border-left-color: #dc2626;
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.1), rgba(30, 41, 59, 0.9));
    }
</style>
""", unsafe_allow_html=True)

# ---------------- API FONKSİYONLARI ----------------
def fetch_contract_ticker():
    """Tüm contract sembollerini getir"""
    url = f"{CONTRACT_BASE}/contract/ticker"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        return data.get('data', [])
    except Exception as e:
        logger.error(f"API Error: {e}")
        return []

def get_top_contracts_by_volume(limit=100):
    """Hacme göre en popüler coinleri getir"""
    try:
        data = fetch_contract_ticker()
        if not data:
            return ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "MATICUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"]
        
        # Hacim hesaplama
        def get_volume(item):
            return float(item.get('volume24', 0) or item.get('amount24', 0) or 0)
        
        sorted_data = sorted(data, key=get_volume, reverse=True)
        symbols = [item.get('symbol', '').replace('_', '') for item in sorted_data[:limit]]
        return [s for s in symbols if s]
    except Exception as e:
        logger.error(f"Volume sort error: {e}")
        return ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]

def fetch_contract_klines(symbol, interval, limit=100):
    """Kline verilerini getir"""
    try:
        # Sembol formatını düzelt
        if '_' not in symbol:
            if symbol.endswith('USDT'):
                symbol = symbol.replace('USDT', '_USDT')
        
        url = f"{CONTRACT_BASE}/contract/kline/{symbol}"
        params = {'interval': interval, 'limit': limit}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'data' not in data:
            return pd.DataFrame()
            
        kline_data = data['data']
        
        # Veri yapısını kontrol et
        if not kline_data or 'time' not in kline_data:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(kline_data['time'], unit='s'),
            'open': kline_data.get('open', []),
            'high': kline_data.get('high', []),
            'low': kline_data.get('low', []),
            'close': kline_data.get('close', []),
            'volume': kline_data.get('vol', [])
        })
        
        # Numeric dönüşüm
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.dropna()
        
    except Exception as e:
        logger.error(f"Kline error for {symbol}: {e}")
        return pd.DataFrame()

# ---------------- BASİT İNDİKATÖR HESAPLAMA ----------------
def calculate_simple_indicators(df):
    """Basit ve etkili indikatörler"""
    if df.empty or len(df) < 20:
        return None
        
    try:
        # Price based indicators
        close = df['close']
        
        # EMA'lar
        ema_20 = ta.ema(close, length=20)
        ema_50 = ta.ema(close, length=50)
        
        # RSI
        rsi = ta.rsi(close, length=14)
        
        # MACD
        macd = ta.macd(close)
        macd_line = macd['MACD_12_26_9'] if macd is not None else None
        macd_signal = macd['MACDs_12_26_9'] if macd is not None else None
        
        # Volume
        volume_sma = ta.sma(df['volume'], length=20)
        volume_ratio = df['volume'] / volume_sma
        
        # Son değerleri al
        result = {
            'price': float(close.iloc[-1]),
            'ema_20': float(ema_20.iloc[-1]) if ema_20 is not None else None,
            'ema_50': float(ema_50.iloc[-1]) if ema_50 is not None else None,
            'rsi': float(rsi.iloc[-1]) if rsi is not None else 50,
            'macd_line': float(macd_line.iloc[-1]) if macd_line is not None else 0,
            'macd_signal': float(macd_signal.iloc[-1]) if macd_signal is not None else 0,
            'volume_ratio': float(volume_ratio.iloc[-1]) if volume_ratio is not None else 1,
            'trend': 'UP' if close.iloc[-1] > close.iloc[-5] else 'DOWN'
        }
        
        # MACD histogram
        if macd_line is not None and macd_signal is not None:
            result['macd_histogram'] = result['macd_line'] - result['macd_signal']
        else:
            result['macd_histogram'] = 0
            
        return result
        
    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return None

# ---------------- BASİT SİNYAL SİSTEMİ ----------------
def generate_signal(indicators):
    """Basit ve etkili sinyal üretimi"""
    if not indicators:
        return "NÖTR", 0, []
    
    score = 0
    reasons = []
    
    try:
        # 1. EMA Trend Analizi (30 puan)
        if indicators['ema_20'] and indicators['ema_50']:
            if indicators['ema_20'] > indicators['ema_50']:
                score += 15
                reasons.append("EMA20 > EMA50 - Yükseliş trendi")
            else:
                score -= 15
                reasons.append("EMA20 < EMA50 - Düşüş trendi")
                
            # EMA mesafe bonusu
            ema_distance = abs(indicators['ema_20'] - indicators['ema_50']) / indicators['price']
            if ema_distance > 0.02:
                if indicators['ema_20'] > indicators['ema_50']:
                    score += 5
                    reasons.append("EMA'lar arası mesafe yüksek - Trend güçlü")
                else:
                    score -= 5
                    reasons.append("EMA'lar arası mesafe yüksek - Trend güçlü")
        
        # 2. RSI Analizi (25 puan)
        rsi = indicators['rsi']
        if rsi < 30:
            score += 20
            reasons.append(f"RSI {rsi:.1f} - Aşırı satım")
        elif rsi < 45:
            score += 10
            reasons.append(f"RSI {rsi:.1f} - Satım bölgesi")
        elif rsi > 70:
            score -= 20
            reasons.append(f"RSI {rsi:.1f} - Aşırı alım")
        elif rsi > 55:
            score -= 10
            reasons.append(f"RSI {rsi:.1f} - Alım bölgesi")
        
        # 3. MACD Analizi (25 puan)
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0:
            score += 15
            reasons.append("MACD pozitif - Bullish momentum")
        else:
            score -= 15
            reasons.append("MACD negatif - Bearish momentum")
            
        # MACD güç bonusu
        if abs(macd_hist) > 0.001:
            if macd_hist > 0:
                score += 5
            else:
                score -= 5
        
        # 4. Volume Analizi (20 puan)
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            # Volume trend ile aynı yönde mi?
            if (score > 0 and indicators['trend'] == 'UP') or (score < 0 and indicators['trend'] == 'DOWN'):
                score += 10
                reasons.append(f"Yüksek hacim x{volume_ratio:.1f} - Trend destekleniyor")
            else:
                score += 5
                reasons.append(f"Yüksek hacim x{volume_ratio:.1f} - Dikkat")
        elif volume_ratio < 0.7:
            score -= 5
            reasons.append(f"Düşük hacim x{volume_ratio:.1f} - Trend zayıf")
        
        # Skoru sınırla
        score = max(-100, min(100, score))
        
        # Sinyal belirle
        if score >= 40:
            signal = "GÜÇLÜ AL"
        elif score >= 15:
            signal = "AL"
        elif score <= -40:
            signal = "GÜÇLÜ SAT" 
        elif score <= -15:
            signal = "SAT"
        else:
            signal = "NÖTR"
            
        return signal, score, reasons
        
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        return "NÖTR", 0, ["Sinyal hesaplama hatası"]

# ---------------- TARAMA MOTORU ----------------
def scan_symbols(symbols, timeframes, top_n=50):
    """Sembolleri tara"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols[:top_n]):
        status_text.text(f"🔍 Taranıyor: {symbol} ({i+1}/{min(top_n, len(symbols))})")
        progress_bar.progress((i + 1) / min(top_n, len(symbols)))
        
        best_signal = "NÖTR"
        best_score = 0
        best_tf = None
        best_indicators = None
        best_reasons = []
        
        for tf in timeframes:
            try:
                interval = INTERVAL_MAP.get(tf)
                if not interval:
                    continue
                    
                # Verileri getir
                df = fetch_contract_klines(symbol, interval, 100)
                if df.empty or len(df) < 30:
                    continue
                
                # İndikatörleri hesapla
                indicators = calculate_simple_indicators(df)
                if not indicators:
                    continue
                
                # Sinyal üret
                signal, score, reasons = generate_signal(indicators)
                
                # En iyi sinyali güncelle
                if abs(score) > abs(best_score):
                    best_score = score
                    best_signal = signal
                    best_tf = tf
                    best_indicators = indicators
                    best_reasons = reasons
                    
            except Exception as e:
                logger.error(f"Scan error for {symbol} {tf}: {e}")
                continue
        
        if best_tf:  # Sadece sinyal üretenleri ekle
            results.append({
                'symbol': symbol,
                'timeframe': best_tf,
                'signal': best_signal,
                'score': best_score,
                'price': best_indicators['price'] if best_indicators else 0,
                'reasons': best_reasons,
                'rsi': best_indicators.get('rsi', 50) if best_indicators else 50,
                'volume_ratio': best_indicators.get('volume_ratio', 1) if best_indicators else 1
            })
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# ---------------- ANA UYGULAMA ----------------
def main():
    try:
        # Header
        st.markdown("""
        <div class='main-header'>
            <h1 style='margin:0; color:white;'>🚀 MEXC PRO SINYAL TERMİNALİ</h1>
            <p style='margin:0; color:#e0f2fe;'>Basit & Etkili Sinyal Sistemi</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("### ⚙️ Tarama Ayarları")
            
            # Sembol seçimi
            st.markdown("#### 📊 Sembol Seçimi")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎯 Major Coins", use_container_width=True):
                    st.session_state.selected_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "MATICUSDT"]
            
            with col2:
                if st.button("🔥 Top 50", use_container_width=True):
                    st.session_state.selected_symbols = get_top_contracts_by_volume(50)
            
            # Custom sembol listesi
            custom_text = st.text_area(
                "📝 Özel Sembol Listesi",
                value="BTCUSDT,ETHUSDT,ADAUSDT,SOLUSDT,MATICUSDT,AVAXUSDT,DOTUSDT,LINKUSDT,BNBUSDT",
                help="Virgülle ayırarak sembolleri girin"
            )
            
            symbols = [s.strip().upper() for s in custom_text.split(',') if s.strip()]
            if not symbols:
                symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
            
            # Zaman dilimleri
            st.markdown("#### ⏰ Zaman Dilimleri")
            timeframes = st.multiselect(
                "Seçili Zaman Dilimleri",
                options=ALL_TFS,
                default=DEFAULT_TFS
            )
            
            if not timeframes:
                timeframes = DEFAULT_TFS
            
            # Tarama limiti
            top_n = st.slider(
                "🔢 Tarama Limiti",
                min_value=5,
                max_value=min(100, len(symbols)),
                value=min(30, len(symbols))
            )
            
            # Tarama butonu
            scan_clicked = st.button(
                "🚀 TARAMAYI BAŞLAT", 
                type="primary",
                use_container_width=True,
                use_container_width=True
            )
        
        # Session state
        if 'scan_results' not in st.session_state:
            st.session_state.scan_results = pd.DataFrame()
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = None
        
        # Tarama işlemi
        if scan_clicked:
            with st.spinner("🔄 Coinler taranıyor... Lütfen bekleyin"):
                results = scan_symbols(symbols, timeframes, top_n)
                st.session_state.scan_results = results
                st.session_state.last_scan = datetime.now()
                
                if not results.empty:
                    st.success(f"✅ Tarama tamamlandı! {len(results)} sinyal bulundu")
                else:
                    st.warning("🤔 Hiç sinyal bulunamadı. Ayarları kontrol edin.")
        
        # Sonuçları göster
        display_results()
        
    except Exception as e:
        st.error(f"Uygulama hatası: {str(e)}")
        logger.error(f"Main app error: {e}")

def display_results():
    """Sonuçları göster"""
    df = st.session_state.scan_results
    
    if df.empty:
        show_welcome_message()
        return
    
    # İstatistikler
    total_signals = len(df)
    strong_buy = len(df[df['signal'] == 'GÜÇLÜ AL'])
    buy = len(df[df['signal'] == 'AL'])
    sell = len(df[df['signal'] == 'SAT'])
    strong_sell = len(df[df['signal'] == 'GÜÇLÜ SAT'])
    
    st.markdown("### 📊 Tarama Sonuçları")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Toplam Sinyal", total_signals)
    with col2:
        st.metric("🚀 Güçlü Al", strong_buy)
    with col3:
        st.metric("📈 Al", buy)
    with col4:
        st.metric("📉 Sat", sell)
    with col5:
        st.metric("🔻 Güçlü Sat", strong_sell)
    
    # Filtreler
    st.markdown("### 🎯 Sinyal Listesi")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        signal_filter = st.selectbox(
            "Sinyal Filtresi",
            ["Tümü", "GÜÇLÜ AL", "AL", "NÖTR", "SAT", "GÜÇLÜ SAT"]
        )
    
    with col2:
        min_score = st.slider("Min Skor", -100, 100, -100)
    
    # Filtrelenmiş sinyaller
    filtered_df = df.copy()
    if signal_filter != "Tümü":
        filtered_df = filtered_df[filtered_df['signal'] == signal_filter]
    filtered_df = filtered_df[abs(filtered_df['score']) >= abs(min_score)]
    
    if filtered_df.empty:
        st.info("🤔 Filtrelerinize uygun sinyal bulunamadı")
        return
    
    # Sinyal kartlarını göster
    for _, row in filtered_df.iterrows():
        display_signal_card(row)
    
    # Detaylı analiz
    if st.session_state.get('selected_symbol'):
        display_symbol_details()

def display_signal_card(row):
    """Sinyal kartını göster"""
    symbol = row['symbol']
    signal = row['signal']
    score = row['score']
    price = row['price']
    timeframe = row['timeframe']
    rsi = row.get('rsi', 50)
    volume_ratio = row.get('volume_ratio', 1)
    
    # Sinyal tipine göre stil
    signal_class = "signal-neutral"
    emoji = "⚪"
    
    if signal == "GÜÇLÜ AL":
        signal_class = "signal-strong-long"
        emoji = "🚀"
    elif signal == "AL":
        signal_class = "signal-long"
        emoji = "📈"
    elif signal == "SAT":
        signal_class = "signal-short"
        emoji = "📉"
    elif signal == "GÜÇLÜ SAT":
        signal_class = "signal-strong-short"
        emoji = "🔻"
    
    st.markdown(f"""
    <div class='signal-card {signal_class}'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
            <h4 style='margin: 0; color: white;'>{emoji} {symbol}</h4>
            <div style='background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 15px; font-size: 12px;'>
                <strong>{signal}</strong>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; color: #cbd5e1;'>
            <div>⏰ Zaman: <strong>{timeframe}</strong></div>
            <div>💎 Skor: <strong>{score}</strong></div>
            <div>💰 Fiyat: <strong>${price:.4f}</strong></div>
            <div>📊 RSI: <strong>{rsi:.1f}</strong></div>
            <div>🔊 Hacim: <strong>x{volume_ratio:.1f}</strong></div>
            <div>🎯 Güven: <strong>{abs(score)}%</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detay butonu
    if st.button("📈 Detaylı Analiz", key=f"btn_{symbol}", use_container_width=True):
        st.session_state.selected_symbol = symbol

def display_symbol_details():
    """Sembol detaylarını göster"""
    symbol = st.session_state.selected_symbol
    df = st.session_state.scan_results
    
    if not symbol or df.empty:
        return
        
    symbol_data = df[df['symbol'] == symbol].iloc[0]
    
    st.markdown("---")
    st.markdown(f"### 📊 {symbol} Detaylı Analiz")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # TradingView grafiği
        timeframe = symbol_data['timeframe']
        interval_tv = '15' if timeframe == '15m' else '60' if timeframe == '1h' else '240'
        
        html_code = f"""
        <div class="tradingview-widget-container" style="height:500px;">
          <div id="tradingview_{symbol}"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "container_id": "tradingview_{symbol}",
            "symbol": "BINANCE:{symbol}",
            "interval": "{interval_tv}",
            "timezone": "Europe/Istanbul",
            "theme": "dark",
            "style": "1",
            "locale": "tr",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies"]
          }});
          </script>
        </div>
        """
        components.html(html_code, height=500)
    
    with col2:
        st.markdown("#### 🎯 Sinyal Detayları")
        
        st.metric("Sembol", symbol_data['symbol'])
        st.metric("Sinyal", symbol_data['signal'])
        st.metric("Skor", symbol_data['score'])
        st.metric("Zaman Dilimi", symbol_data['timeframe'])
        st.metric("Fiyat", f"${symbol_data['price']:.4f}")
        st.metric("RSI", f"{symbol_data.get('rsi', 0):.1f}")
        st.metric("Hacim Çarpanı", f"x{symbol_data.get('volume_ratio', 0):.1f}")
        
        # Sinyal nedenleri
        st.markdown("#### 📋 Sinyal Nedenleri")
        for reason in symbol_data['reasons']:
            st.write(f"• {reason}")

def show_welcome_message():
    """Hoş geldin mesajı"""
    st.info("""
    ## 🎯 MEXC PRO SINYAL TERMİNALİ
    
    **Özellikler:**
    - 🚀 **Hızlı Tarama** - 50+ coin simultane analiz
    - 📊 **Çoklu İndikatör** - EMA, RSI, MACD, Volume
    - ⏰ **Multi-Timeframe** - 15m, 1h, 4h analiz
    - 🎯 **Net Sinyaller** - GÜÇLÜ AL, AL, SAT, GÜÇLÜ SAT
    - 📈 **Gerçek Zamanlı** - Canlı piyasa verileri
    
    **Başlamak için:**
    1. Sol menüden sembolleri seçin (veya butonlarla)
    2. Zaman dilimlerini belirleyin
    3. "TARAMAYI BAŞLAT" butonuna tıklayın
    
    **Önerilen Ayarlar:**
    - Semboller: Major Coins veya Top 50
    - Zaman Dilimleri: 15m, 1h, 4h
    - Tarama Limiti: 20-30 coin
    """)

if __name__ == "__main__":
    main()
