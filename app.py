# app.py
# Gelişmiş MEXC Vadeli Sinyal Uygulaması - Specter Trend Cloud & AI Hibrit Motor

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
import ai_engine  # <-- Gelişmiş AI motorumuz
import trend_cloud  # <-- Yeni Specter Trend Cloud indikatörü
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Gelişmiş temalar ve görsel öğeler
st.set_page_config(
    page_title="MEXC Pro Sinyal Terminali", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🚀"
)

# ---------------- CONFIG & STYLING ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']

# Gelişmiş CSS Styling
st.markdown("""
<style>
    /* Ana tema */
    .main {
        background: linear-gradient(135deg, #0c1426 0%, #0a0f1d 50%, #070b16 100%);
        color: #e6eef6;
    }
    
    /* Header ve kart stilleri */
    .header-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .signal-card {
        background: rgba(15, 23, 42, 0.8);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 8px 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .signal-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.2);
    }
    
    /* Sinyal etiketleri */
    .signal-long {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    
    .signal-short {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    
    /* Metrik kartlar */
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.05);
        text-align: center;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f172a 0%, #0a0f1d 100%);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- API FONKSİYONLARI ----------------
def fetch_json(url, params=None, timeout=10):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_contract_ticker():
    url = f"{CONTRACT_BASE}/contract/ticker"
    try:
        j = fetch_json(url)
        return j.get('data', [])
    except Exception:
        return []

def get_top_contracts_by_volume(limit=200):
    data = fetch_contract_ticker()
    def vol(x):
        return float(x.get('volume24') or x.get('amount24') or 0)
    items = sorted(data, key=vol, reverse=True)
    syms = [it.get('symbol') for it in items[:limit]]
    return [s.replace('_','') for s in syms if s]

def mexc_symbol_from(symbol: str) -> str:
    s = symbol.strip().upper()
    if '_' in s: return s
    if s.endswith('USDT'): return s[:-4] + "_USDT"
    return s[:-4] + "_" + s[-4:]

def fetch_contract_klines(symbol_mexc, interval_mexc, limit=200):
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    try:
        j = fetch_json(url, params={'interval': interval_mexc, 'limit': limit})
        d = j.get('data') or {}
        times = d.get('time', [])
        if not times:
            return pd.DataFrame()
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(d.get('time'), unit='s'),
            'open': d.get('open'),
            'high': d.get('high'), 
            'low': d.get('low'),
            'close': d.get('close'),
            'volume': d.get('vol')
        })
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def fetch_contract_funding_rate(symbol_mexc):
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"
    try:
        j = fetch_json(url)
        data = j.get('data') or {}
        return {'fundingRate': float(data.get('fundingRate') or 0)}
    except Exception:
        return {'fundingRate': 0.0}

# ---------------- SPECTER TREND CLOUD ANALİZİ ----------------
def analyze_with_specter_trend(df, symbol, timeframe):
    """Specter Trend Cloud analizi yapar"""
    if df.empty or len(df) < 50:
        return None
    
    try:
        # Trend Cloud hesaplamaları
        specter_data = trend_cloud.calculate_specter_cloud(df)
        
        if specter_data is None:
            return None
            
        # Son sinyali al
        latest = specter_data.iloc[-1]
        prev = specter_data.iloc[-2] if len(specter_data) > 1 else latest
        
        # Trend analizi
        current_trend = "BULLISH" if latest['trend'] == 1 else "BEARISH"
        trend_strength = abs(latest['momentum_strength'])
        
        # Retest sinyalleri
        retest_signals = []
        if latest['bullish_retest']:
            retest_signals.append("BULLISH_RETEST")
        if latest['bearish_retest']:
            retest_signals.append("BEARISH_RETEST")
            
        # Seviye analizi
        levels = {
            'current_price': latest['close'],
            'cloud_top': latest['ma_upper'],
            'cloud_bottom': latest['ma_lower'],
            'short_ma': latest['ma_short'],
            'long_ma': latest['ma_long']
        }
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'trend': current_trend,
            'trend_strength': trend_strength,
            'retest_signals': retest_signals,
            'levels': levels,
            'momentum': latest['momentum_strength'],
            'cloud_data': specter_data
        }
        
    except Exception as e:
        st.error(f"Specter analiz hatası {symbol}: {str(e)}")
        return None

# ---------------- GELİŞMİŞ TARAMA MOTORU ----------------
@st.cache_data(ttl=120)
def run_advanced_scan(symbols, timeframes, gemini_api_key, top_n=100):
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, sym in enumerate(symbols[:top_n]):
        status_text.text(f"📊 Analiz ediliyor: {sym} ({idx+1}/{top_n})")
        progress_bar.progress((idx + 1) / top_n)
        
        entry = {
            'symbol': sym, 
            'details': {},
            'specter_analysis': {},
            'ai_predictions': {}
        }
        
        mexc_sym = mexc_symbol_from(sym)
        funding = fetch_contract_funding_rate(mexc_sym)
        
        best_tf_score = -1
        best_tf = None
        
        for tf in timeframes:
            interval = INTERVAL_MAP.get(tf)
            if interval is None:
                continue
                
            # Kline verilerini al
            df = fetch_contract_klines(mexc_sym, interval, limit=200)
            if df.empty or len(df) < 50:
                continue
            
            # 1. Specter Trend Cloud Analizi
            specter_analysis = analyze_with_specter_trend(df, sym, tf)
            
            # 2. AI Analizi için indikatör snapshot'ı oluştur
            indicators_snapshot = trend_cloud.create_ai_snapshot(df, funding)
            
            # 3. Gelişmiş AI Analizi
            ai_analysis = ai_engine.get_ai_prediction(
                indicators_snapshot, 
                specter_analysis,
                api_key=gemini_api_key
            )
            
            # 4. Kombine skor hesapla
            combined_score = ai_engine.calculate_combined_score(ai_analysis, specter_analysis)
            
            entry['details'][tf] = {
                'price': float(df['close'].iloc[-1]),
                'specter': specter_analysis,
                'ai_analysis': ai_analysis,
                'combined_score': combined_score,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # En iyi timeframe'i güncelle
            if combined_score > best_tf_score:
                best_tf_score = combined_score
                best_tf = tf
                entry['best_analysis'] = {
                    'timeframe': tf,
                    'specter': specter_analysis,
                    'ai_analysis': ai_analysis,
                    'combined_score': combined_score
                }
        
        if entry['best_analysis']:
            results.append(entry)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# ---------------- GELİŞMİŞ GÖRSELLEŞTİRME ----------------
def create_specter_chart(df, specter_data, symbol, timeframe):
    """Specter Trend Cloud grafiği oluşturur"""
    
    fig = make_subplots(rows=2, cols=1, shared_x=True,
                       vertical_spacing=0.05,
                       subplot_titles=(f'{symbol} - {timeframe} Price Action with Specter Cloud', 
                                     'Momentum Strength'),
                       row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df['timestamp'],
                                open=df['open'],
                                high=df['high'],
                                low=df['low'],
                                close=df['close'],
                                name='Price'),
                 row=1, col=1)
    
    # Specter Cloud
    fig.add_trace(go.Scatter(x=specter_data['timestamp'],
                            y=specter_data['ma_upper'],
                            line=dict(color='rgba(0,0,0,0)'),
                            showlegend=False,
                            name='Cloud Upper'),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=specter_data['timestamp'],
                            y=specter_data['ma_lower'],
                            line=dict(color='rgba(0,0,0,0)'),
                            fill='tonexty',
                            fillcolor='rgba(0, 255, 255, 0.2)' if specter_data['trend'].iloc[-1] == 1 else 'rgba(255, 165, 0, 0.2)',
                            name='Trend Cloud'),
                 row=1, col=1)
    
    # Moving averages
    fig.add_trace(go.Scatter(x=specter_data['timestamp'],
                            y=specter_data['ma_short'],
                            line=dict(color='#3B82F6', width=2),
                            name='Short MA'),
                 row=1, col=1)
    
    fig.add_trace(go.Scatter(x=specter_data['timestamp'],
                            y=specter_data['ma_long'],
                            line=dict(color='#EF4444', width=2),
                            name='Long MA'),
                 row=1, col=1)
    
    # Retest markers
    bullish_retests = specter_data[specter_data['bullish_retest'] == True]
    bearish_retests = specter_data[specter_data['bearish_retest'] == True]
    
    if not bullish_retests.empty:
        fig.add_trace(go.Scatter(x=bullish_retests['timestamp'],
                                y=bullish_retests['low'] * 0.998,
                                mode='markers',
                                marker=dict(symbol='diamond', size=10, color='#10B981'),
                                name='Bullish Retest'),
                     row=1, col=1)
    
    if not bearish_retests.empty:
        fig.add_trace(go.Scatter(x=bearish_retests['timestamp'],
                                y=bearish_retests['high'] * 1.002,
                                mode='markers',
                                marker=dict(symbol='diamond', size=10, color='#EF4444'),
                                name='Bearish Retest'),
                     row=1, col=1)
    
    # Momentum
    fig.add_trace(go.Scatter(x=specter_data['timestamp'],
                            y=specter_data['momentum_strength'],
                            line=dict(color='#8B5CF6', width=2),
                            name='Momentum'),
                 row=2, col=1)
    
    fig.update_layout(height=600, 
                     template='plotly_dark',
                     showlegend=True,
                     xaxis_rangeslider_visible=False)
    
    return fig

# ---------------- ANA UYGULAMA ----------------
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class='header-card'>
            <h1 style='text-align: center; margin: 0; color: white;'>🚀 MEXC PRO SINYAL TERMİNALİ</h1>
            <p style='text-align: center; margin: 5px 0 0 0; color: #cbd5e1;'>Specter Trend Cloud & AI Hibrit Analiz Sistemi</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("### ⚙️ Tarama Ayarları")
    
    # API Key
    gemini_api_key = st.sidebar.text_input(
        "🔑 Gemini API Anahtarı", 
        type="password",
        help="Gelişmiş AI analizleri için Gemini API anahtarınızı girin"
    )
    
    # Sembol seçimi
    mode = st.sidebar.selectbox(
        "📊 Sembol Kaynağı", 
        ["Top by volume (200)", "Custom list"]
    )
    
    if mode == "Custom list":
        custom = st.sidebar.text_area(
            "📝 Özel Sembol Listesi", 
            value="BTCUSDT,ETHUSDT,ADAUSDT",
            help="Virgülle ayırarak sembolleri girin"
        )
        symbols = [s.strip().upper() for s in custom.split(',') if s.strip()]
    else:
        symbols = get_top_contracts_by_volume(200)
    
    if not symbols:
        st.sidebar.error("❌ Sembol listesi boş.")
        st.stop()
    
    # Zaman dilimleri
    timeframes = st.sidebar.multiselect(
        "⏰ Zaman Dilimleri", 
        options=ALL_TFS, 
        default=DEFAULT_TFS
    )
    
    top_n = st.sidebar.slider(
        "🔢 İlk N Coin Taransın", 
        min_value=5, 
        max_value=min(100, len(symbols)), 
        value=min(30, len(symbols))
    )
    
    # Specter ayarları
    with st.sidebar.expander("⚡ Specter Trend Cloud Ayarları"):
        ma_type = st.selectbox("MA Tipi", ["SMA", "EMA", "WMA", "DEMA"], index=1)
        base_length = st.slider("Temel Uzunluk", 5, 50, 20)
        atr_multiplier = st.slider("ATR Çarpanı", 0.5, 3.0, 1.5)
    
    # Tarama butonu
    scan_clicked = st.sidebar.button(
        "🔍 Gelişmiş Tarama Başlat", 
        type="primary",
        use_container_width=True
    )
    
    # Otomatik yenileme
    auto_refresh = st.sidebar.checkbox("🔄 Otomatik Yenileme (120sn)", value=False)
    
    # Session state yönetimi
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = pd.DataFrame()
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None
    if 'last_scan' not in st.session_state:
        st.session_state.last_scan = None
    
    # Tarama işlemi
    if scan_clicked or (auto_refresh and st.session_state.last_scan and 
                       (datetime.utcnow() - st.session_state.last_scan).seconds > 120):
        
        with st.spinner("🚀 Gelişmiş tarama çalışıyor... Specter Cloud ve AI analizleri yapılıyor"):
            st.session_state.scan_results = run_advanced_scan(
                symbols, timeframes, gemini_api_key, top_n
            )
            st.session_state.last_scan = datetime.utcnow()
            st.rerun()
    
    # Sonuçları göster
    df = st.session_state.scan_results
    
    if df.empty:
        st.info("""
        ## 📋 Başlarken
        
        **MEXC Pro Sinyal Terminali'ne hoş geldiniz!**
        
        🎯 **Özellikler:**
        - 🤖 **Gelişmiş AI Analiz** - Gemini AI entegrasyonu
        - ☁️ **Specter Trend Cloud** - Profesyonel trend takip sistemi
        - 📊 **Çoklu Zaman Dilimi** - 1m'den 1güne kadar analiz
        - ⚡ **Gerçek Zamanlı Sinyaller** - Anlık al/sat önerileri
        - 🎨 **Gelişmiş Görselleştirme** - Interaktif grafikler
        
        **Başlamak için sol menüden ayarları yapıp 'Gelişmiş Tarama Başlat' butonuna tıklayın.**
        """)
        
    else:
        # AI Sinyal Listesi
        display_ai_signals(df)
        
        # Seçili Sembol Detayları
        display_selected_symbol_details(df, gemini_api_key)

def display_ai_signals(df):
    """AI sinyallerini göster"""
    
    st.markdown("### 🔥 AI Sinyal Listesi")
    
    # Filtreler
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        filter_signal = st.selectbox("Sinyal Türü", ["Tümü", "LONG", "SHORT", "NEUTRAL"])
    
    with col2:
        min_confidence = st.slider("Min Güven", 0, 100, 70)
    
    with col3:
        min_strength = st.slider("Min Trend Gücü", 0, 100, 50)
    
    with col4:
        sort_by = st.selectbox("Sırala", ["Güven", "Trend Gücü", "Kombine Skor"])
    
    # Sinyal kartları
    signals = []
    for _, row in df.iterrows():
        if not row.get('best_analysis'):
            continue
            
        analysis = row['best_analysis']
        ai_analysis = analysis.get('ai_analysis', {})
        specter = analysis.get('specter', {})
        
        signal_info = {
            'symbol': row['symbol'],
            'timeframe': analysis['timeframe'],
            'signal': ai_analysis.get('signal', 'NEUTRAL'),
            'confidence': ai_analysis.get('confidence', 0),
            'trend_strength': specter.get('trend_strength', 0) * 100,
            'combined_score': analysis.get('combined_score', 0),
            'price': analysis.get('ai_analysis', {}).get('entry', 0),
            'explanation': ai_analysis.get('explanation', ''),
            'targets': ai_analysis.get('take_profit'),
            'stop_loss': ai_analysis.get('stop_loss')
        }
        signals.append(signal_info)
    
    # Filtreleme
    filtered_signals = [s for s in signals if s['confidence'] >= min_confidence]
    
    if filter_signal != "Tümü":
        filtered_signals = [s for s in filtered_signals if s['signal'] == filter_signal]
    
    # Sıralama
    if sort_by == "Güven":
        filtered_signals.sort(key=lambda x: x['confidence'], reverse=True)
    elif sort_by == "Trend Gücü":
        filtered_signals.sort(key=lambda x: x['trend_strength'], reverse=True)
    else:
        filtered_signals.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Sinyal kartlarını göster
    cols = st.columns(3)
    for idx, signal in enumerate(filtered_signals[:12]):  # Max 12 sinyal
        with cols[idx % 3]:
            display_signal_card(signal, idx)

def display_signal_card(signal, idx):
    """Tekil sinyal kartını göster"""
    
    signal_class = {
        'LONG': 'signal-long',
        'SHORT': 'signal-short',
        'NEUTRAL': 'signal-neutral'
    }.get(signal['signal'], 'signal-neutral')
    
    emoji = {
        'LONG': '🚀',
        'SHORT': '🔻', 
        'NEUTRAL': '⚪'
    }.get(signal['signal'], '⚪')
    
    st.markdown(f"""
    <div class='signal-card'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
            <h4 style='margin: 0; color: white;'>{emoji} {signal['symbol']}</h4>
            <span class='{signal_class}'>{signal['signal']}</span>
        </div>
        
        <div style='font-size: 12px; color: #cbd5e1;'>
            <div>⏰ TF: {signal['timeframe']}</div>
            <div>🎯 Güven: <strong>{signal['confidence']}%</strong></div>
            <div>💪 Trend: <strong>{signal['trend_strength']:.0f}%</strong></div>
            <div>💰 Fiyat: ${signal['price']:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📈 Detaylı Analiz", key=f"btn_{idx}", use_container_width=True):
        st.session_state.selected_symbol = signal['symbol']

def display_selected_symbol_details(df, api_key):
    """Seçili sembolün detaylı analizini göster"""
    
    st.markdown("---")
    st.markdown("### 📊 Detaylı Teknik Analiz")
    
    # Sembol seçimi
    symbols = [row['symbol'] for _, row in df.iterrows() if row.get('best_analysis')]
    selected_symbol = st.session_state.get('selected_symbol') 
    
    if not selected_symbol and symbols:
        selected_symbol = symbols[0]
    
    if not selected_symbol:
        st.warning("⚠️ Analiz edilecek sembol bulunamadı")
        return
    
    # Sembol seçicisi
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_symbol = st.selectbox("Sembol Seçin:", symbols, index=symbols.index(selected_symbol) if selected_symbol in symbols else 0)
    
    # Seçili sembolün verilerini al
    symbol_data = next((row for _, row in df.iterrows() if row['symbol'] == selected_symbol), None)
    if not symbol_data or not symbol_data.get('best_analysis'):
        st.error("❌ Seçili sembol için analiz bulunamadı")
        return
    
    analysis = symbol_data['best_analysis']
    ai_analysis = analysis.get('ai_analysis', {})
    specter = analysis.get('specter', {})
    
    # Ana metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 AI Sinyali", ai_analysis.get('signal', 'N/A'), 
                 f"%{ai_analysis.get('confidence', 0)} Güven")
    
    with col2:
        st.metric("☁️ Trend", specter.get('trend', 'N/A'),
                 f"%{specter.get('trend_strength', 0) * 100:.0f} Güç")
    
    with col3:
        current_price = ai_analysis.get('entry', 0)
        st.metric("💰 Mevcut Fiyat", f"${current_price:.4f}")
    
    with col4:
        st.metric("🎯 Kombine Skor", f"{analysis.get('combined_score', 0):.0f}")
    
    # Grafik ve analiz bölümü
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Specter Trend Cloud Grafiği
        if specter.get('cloud_data') is not None:
            fig = create_specter_chart(
                specter['cloud_data'], 
                specter['cloud_data'],
                selected_symbol, 
                analysis['timeframe']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("📊 Grafik verisi mevcut değil")
    
    with col2:
        # Ticaret planı
        st.markdown("#### 🎯 Ticaret Planı")
        
        entry = ai_analysis.get('entry', 0)
        stop_loss = ai_analysis.get('stop_loss', 0)
        take_profit = ai_analysis.get('take_profit', 0)
        
        if entry and stop_loss and take_profit:
            risk = abs(entry - stop_loss)
            reward = abs(take_profit - entry)
            rr_ratio = reward / risk if risk > 0 else 0
            
            st.metric("🎯 Giriş", f"${entry:.4f}")
            st.metric("🛑 Stop Loss", f"${stop_loss:.4f}")
            st.metric("🎯 Take Profit", f"${take_profit:.4f}")
            st.metric("⚖️ R/R Oranı", f"{rr_ratio:.2f}:1")
            
            # Kazanç hesaplama
            if st.button("📈 Potansiyel Kazanç Hesapla", use_container_width=True):
                calculate_profit_potential(entry, stop_loss, take_profit)
        
        # AI Açıklaması
        st.markdown("#### 🤖 AI Analizi")
        explanation = ai_analysis.get('explanation', 'Açıklama mevcut değil.')
        st.write(explanation)
        
        # Retest sinyalleri
        if specter.get('retest_signals'):
            st.markdown("#### 📍 Retest Sinyalleri")
            for retest in specter['retest_signals']:
                if 'BULLISH' in retest:
                    st.success("🟢 Bullish Retest - ALIM fırsatı")
                elif 'BEARISH' in retest:
                    st.error("🔴 Bearish Retest - SATIM fırsatı")
    
    # Ek bilgiler
    with st.expander("📋 Detaylı İndikatör Verileri"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Specter Trend Cloud")
            if specter:
                st.json({
                    'Trend': specter.get('trend'),
                    'Trend Gücü': f"{specter.get('trend_strength', 0) * 100:.1f}%",
                    'Momentum': f"{specter.get('momentum', 0):.2f}",
                    'Retest Sinyalleri': specter.get('retest_signals', [])
                })
        
        with col2:
            st.markdown("##### Seviye Analizi")
            if specter.get('levels'):
                levels = specter['levels']
                st.json({
                    'Mevcut Fiyat': f"${levels.get('current_price', 0):.4f}",
                    'Cloud Üstü': f"${levels.get('cloud_top', 0):.4f}",
                    'Cloud Altı': f"${levels.get('cloud_bottom', 0):.4f}",
                    'Kısa MA': f"${levels.get('short_ma', 0):.4f}",
                    'Uzun MA': f"${levels.get('long_ma', 0):.4f}"
                })

def calculate_profit_potential(entry, stop_loss, take_profit):
    """Potansiyel kazanç hesaplama"""
    
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    
    if risk == 0:
        st.error("❌ Risk hesaplanamıyor")
        return
    
    rr_ratio = reward / risk
    
    st.success(f"""
    **💰 Potansiyel Kazanç Analizi:**
    
    - ⚠️ **Risk:** ${risk:.4f} (%{(risk/entry)*100:.2f})
    - 🎯 **Ödül:** ${reward:.4f} (%{(reward/entry)*100:.2f})  
    - ⚖️ **Risk/Ödül Oranı:** {rr_ratio:.2f}:1
    - 📈 **Başarı Oranı Gereksinimi:** %{100/(1+rr_ratio):.1f}
    
    💡 **Öneri:** {'Mükemmel trade!' if rr_ratio >= 2 else 'İyi trade' if rr_ratio >= 1.5 else 'Dikkatli olun'}
    """)

if __name__ == "__main__":
    main()
