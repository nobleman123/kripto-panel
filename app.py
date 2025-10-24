# app.py
# MEXC Pro AI Trading Terminal - Premium Sinyal Sistemi

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime, timedelta
import ai_engine
import technical_indicators
import market_analysis
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
from typing import Dict, Any, List
import hashlib
import hmac

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gelişmiş temalar ve görsel öğeler
st.set_page_config(
    page_title="MEXC PRO AI TRADER", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🚀"
)

# ---------------- GELİŞMİŞ STYLING ----------------
st.markdown("""
<style>
    /* Premium Dark Theme */
    .main {
        background: linear-gradient(135deg, #0a0f1d 0%, #071018 50%, #050a14 100%);
        color: #ffffff;
    }
    
    /* Header Styles */
    .premium-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 20px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        text-align: center;
    }
    
    .market-sentiment-bullish {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        padding: 15px;
        border-radius: 15px;
        border: 2px solid #10b981;
        text-align: center;
        margin: 10px 0;
    }
    
    .market-sentiment-bearish {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        padding: 15px;
        border-radius: 15px;
        border: 2px solid #ef4444;
        text-align: center;
        margin: 10px 0;
    }
    
    .market-sentiment-neutral {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        padding: 15px;
        border-radius: 15px;
        border: 2px solid #9ca3af;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Signal Cards */
    .signal-card-premium {
        background: rgba(15, 23, 42, 0.9);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid;
        margin: 12px 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .signal-card-premium:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    }
    
    .signal-strong-long {
        border-color: #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(15, 23, 42, 0.9) 100%);
    }
    
    .signal-strong-short {
        border-color: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(15, 23, 42, 0.9) 100%);
    }
    
    .signal-moderate {
        border-color: #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(15, 23, 42, 0.9) 100%);
    }
    
    .signal-neutral {
        border-color: #6b7280;
        background: linear-gradient(135deg, rgba(107, 114, 128, 0.1) 0%, rgba(15, 23, 42, 0.9) 100%);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        text-align: center;
        margin: 5px;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 30%, #10b981 70%, #3b82f6 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f172a 0%, #0a0f1d 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIGURATION ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']

# ---------------- SECURE API FUNCTIONS ----------------
def secure_fetch_json(url, params=None, timeout=8):
    """Güvenli API çağrısı"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API Error {url}: {str(e)}")
        return {}

def get_all_contract_symbols():
    """Tüm vadeli işlem sembollerini getir"""
    try:
        url = f"{CONTRACT_BASE}/contract/ticker"
        data = secure_fetch_json(url)
        symbols = [item['symbol'].replace('_', '') for item in data.get('data', []) if 'symbol' in item]
        return sorted(list(set(symbols)))  # Benzersiz semboller
    except Exception as e:
        logger.error(f"Symbol fetch error: {str(e)}")
        return ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT"]

def fetch_klines_with_retry(symbol, interval, limit=300):
    """Kline verilerini güvenli şekilde getir"""
    mexc_symbol = symbol.replace('USDT', '_USDT') if not '_' in symbol else symbol
    url = f"{CONTRACT_BASE}/contract/kline/{mexc_symbol}"
    
    try:
        data = secure_fetch_json(url, params={'interval': interval, 'limit': limit})
        if not data or 'data' not in data:
            return pd.DataFrame()
            
        kline_data = data['data']
        if not kline_data or 'time' not in kline_data:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(kline_data['time'], unit='s'),
            'open': kline_data['open'],
            'high': kline_data['high'],
            'low': kline_data['low'],
            'close': kline_data['close'],
            'volume': kline_data['vol']
        })
        
        # Veri tipi dönüşümü
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.dropna()
        
    except Exception as e:
        logger.error(f"Kline error {symbol}: {str(e)}")
        return pd.DataFrame()

# ---------------- ENHANCED SCANNING ENGINE ----------------
@st.cache_data(ttl=150, show_spinner=False)
def run_premium_scan(selected_symbols, timeframes, trading_style, gemini_api_key):
    """Premium tarama motoru"""
    results = []
    total_symbols = len(selected_symbols)
    
    if total_symbols == 0:
        return pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(selected_symbols):
        try:
            status_text.text(f"🔍 Analiz: {symbol} ({idx+1}/{total_symbols})")
            progress_bar.progress((idx + 1) / total_symbols)
            
            entry = {
                'symbol': symbol,
                'details': {},
                'best_analysis': None,
                'scan_time': datetime.utcnow().isoformat()
            }
            
            best_score = -1
            best_tf_data = None
            
            for tf in timeframes:
                interval = INTERVAL_MAP.get(tf)
                if not interval:
                    continue
                    
                # Verileri getir
                df = fetch_klines_with_retry(symbol, interval, 300)
                if df.empty or len(df) < 100:
                    continue
                
                # Teknik analiz
                tech_analysis = technical_indicators.calculate_all_indicators(df)
                if tech_analysis is None:
                    continue
                    
                # Piyasa verileri
                market_data = market_analysis.get_market_snapshot(symbol)
                
                # AI analizi
                try:
                    ai_result = ai_engine.get_ai_prediction(
                        tech_analysis, 
                        market_data, 
                        trading_style,
                        gemini_api_key
                    )
                except Exception as e:
                    logger.warning(f"AI analysis failed for {symbol}: {str(e)}")
                    continue
                
                # Skor hesaplama
                score = ai_engine.calculate_premium_score(ai_result, tech_analysis, trading_style)
                
                entry['details'][tf] = {
                    'technical': tech_analysis,
                    'ai_analysis': ai_result,
                    'score': score,
                    'price': float(df['close'].iloc[-1])
                }
                
                # En iyi zaman dilimini güncelle
                if score > best_score:
                    best_score = score
                    best_tf_data = {
                        'timeframe': tf,
                        'technical': tech_analysis,
                        'ai_analysis': ai_result,
                        'score': score,
                        'price': float(df['close'].iloc[-1])
                    }
            
            if best_tf_data:
                entry['best_analysis'] = best_tf_data
                results.append(entry)
                
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results) if results else pd.DataFrame()

# ---------------- ENHANCED VISUALIZATION ----------------
def create_premium_chart(tech_data, symbol, timeframe):
    """Premium teknik analiz grafiği"""
    try:
        fig = make_subplots(
            rows=4, cols=1,
            shared_x=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f'🎯 {symbol} - {timeframe} Premium Analysis',
                '📊 Momentum Indicators',
                '📈 Volume Analysis', 
                '⚡ Oscillators'
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price and MAs
        fig.add_trace(go.Candlestick(
            x=tech_data['timestamp'],
            open=tech_data['open'],
            high=tech_data['high'],
            low=tech_data['low'],
            close=tech_data['close'],
            name='Price'
        ), row=1, col=1)
        
        # EMA'lar
        for ma_period in [20, 50, 200]:
            if f'ema_{ma_period}' in tech_data.columns:
                fig.add_trace(go.Scatter(
                    x=tech_data['timestamp'],
                    y=tech_data[f'ema_{ma_period}'],
                    name=f'EMA {ma_period}',
                    line=dict(width=2)
                ), row=1, col=1)
        
        # RSI
        if 'rsi_14' in tech_data.columns:
            fig.add_trace(go.Scatter(
                x=tech_data['timestamp'],
                y=tech_data['rsi_14'],
                name='RSI',
                line=dict(color='#F59E0B', width=2)
            ), row=4, col=1)
            
            # RSI seviyeleri
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=4, col=1)
        
        # MACD
        if all(col in tech_data.columns for col in ['macd', 'macd_signal']):
            fig.add_trace(go.Scatter(
                x=tech_data['timestamp'],
                y=tech_data['macd'],
                name='MACD',
                line=dict(color='#10B981', width=2)
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=tech_data['timestamp'],
                y=tech_data['macd_signal'],
                name='MACD Signal',
                line=dict(color='#EF4444', width=2)
            ), row=2, col=1)
        
        # Volume
        colors = ['red' if tech_data['close'].iloc[i] < tech_data['open'].iloc[i] else 'green' 
                 for i in range(len(tech_data))]
        
        fig.add_trace(go.Bar(
            x=tech_data['timestamp'],
            y=tech_data['volume'],
            name='Volume',
            marker_color=colors
        ), row=3, col=1)
        
        fig.update_layout(
            height=900,
            template='plotly_dark',
            showlegend=True,
            xaxis_rangeslider_visible=False,
            title_font_size=20
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Chart error: {str(e)}")
        return None

# ---------------- MAIN APPLICATION ----------------
def main():
    try:
        # Premium Header
        st.markdown("""
        <div class='premium-header'>
            <h1 style='margin:0; color:white; font-size:2.5em;'>🚀 MEXC PRO AI TRADER</h1>
            <p style='margin:0; color:#e0f2fe; font-size:1.2em;'>Premium Sinyal Sistemi • Gerçek Zamanlı AI Analiz</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Piyasa Sentiment Gösterge
        display_market_sentiment()
        
        # Sidebar - Güvenli Ayarlar
        with st.sidebar:
            st.markdown("### ⚙️ Premium Ayarlar")
            
            # Güvenli API Key Girişi
            gemini_api_key = st.text_input(
                "🔐 Gemini API Key",
                type="password",
                help="AI analizleri için API keyinizi girin",
                placeholder="sk-proj-xxxxxxxxxxxxxxxx"
            )
            
            # İşlem Stili Seçimi
            trading_style = st.selectbox(
                "🎯 İşlem Stili",
                ["SCALP", "SWING", "POSITION"],
                help="SCALP: 1m-15m, SWING: 1h-4h, POSITION: 4h-1d"
            )
            
            # Zaman Dilimleri
            timeframe_map = {
                "SCALP": ['1m', '5m', '15m'],
                "SWING": ['15m', '1h', '4h'], 
                "POSITION": ['4h', '1d']
            }
            
            timeframes = st.multiselect(
                "⏰ Zaman Dilimleri",
                options=ALL_TFS,
                default=timeframe_map.get(trading_style, ['15m', '1h', '4h'])
            )
            
            # Sembol Seçimi
            st.markdown("### 📊 Sembol Seçimi")
            all_symbols = get_all_contract_symbols()
            
            # Hızlı seçim butonları
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🎯 Major Coins", use_container_width=True):
                    st.session_state.selected_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT"]
            with col2:
                if st.button("🔥 Trend Coins", use_container_width=True):
                    st.session_state.selected_symbols = ["AVAXUSDT", "MATICUSDT", "LINKUSDT", "ATOMUSDT", "NEARUSDT"]
            with col3:
                if st.button("📈 All Coins", use_container_width=True):
                    st.session_state.selected_symbols = all_symbols[:50]
            
            # Multi-select sembol seçimi
            selected_symbols = st.multiselect(
                "💰 Semboller Seçin",
                options=all_symbols,
                default=st.session_state.get('selected_symbols', ["BTCUSDT", "ETHUSDT"]),
                help="Analiz edilecek sembolleri seçin"
            )
            
            # Tarama butonu
            scan_clicked = st.button(
                "🚀 PREMIUM TARAMA BAŞLAT",
                type="primary",
                use_container_width=True,
                help="Seçili sembolleri AI ile tarar"
            )
        
        # Session state yönetimi
        if 'scan_results' not in st.session_state:
            st.session_state.scan_results = pd.DataFrame()
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = None
        if 'saved_signals' not in st.session_state:
            st.session_state.saved_signals = []
        
        # Tarama işlemi
        if scan_clicked and selected_symbols:
            with st.spinner("🚀 Premium tarama çalışıyor... AI derin analiz yapıyor"):
                try:
                    st.session_state.scan_results = run_premium_scan(
                        selected_symbols, timeframes, trading_style, gemini_api_key
                    )
                    st.session_state.last_scan = datetime.utcnow()
                    st.rerun()
                except Exception as e:
                    st.error(f"Tarama hatası: {str(e)}")
        
        # Sonuçları göster
        display_results()
        
    except Exception as e:
        logger.error(f"Main app error: {str(e)}")
        st.error("Uygulamada bir hata oluştu. Lütfen sayfayı yenileyin.")

def display_market_sentiment():
    """Piyasa sentiment göstergesi"""
    try:
        sentiment = market_analysis.get_market_sentiment()
        
        if sentiment['sentiment'] == "BULLISH":
            st.markdown(f"""
            <div class='market-sentiment-bullish'>
                <h3>🎯 PİYASA TAHMİNİ: GÜÇLÜ AL</h3>
                <p>💰 Fear & Greed: {sentiment['fear_greed']} | 📊 Trend: {sentiment['trend_strength']}%</p>
                <p>🔍 {sentiment['analysis']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif sentiment['sentiment'] == "BEARISH":
            st.markdown(f"""
            <div class='market-sentiment-bearish'>
                <h3>🎯 PİYASA TAHMİNİ: GÜÇLÜ SAT</h3>
                <p>💰 Fear & Greed: {sentiment['fear_greed']} | 📊 Trend: {sentiment['trend_strength']}%</p>
                <p>🔍 {sentiment['analysis']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='market-sentiment-neutral'>
                <h3>🎯 PİYASA TAHMİNİ: NÖTR</h3>
                <p>💰 Fear & Greed: {sentiment['fear_greed']} | 📊 Trend: {sentiment['trend_strength']}%</p>
                <p>🔍 {sentiment['analysis']}</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        logger.error(f"Sentiment error: {str(e)}")

def display_results():
    """Tarama sonuçlarını göster"""
    df = st.session_state.scan_results
    
    if df.empty:
        show_welcome_screen()
        return
    
    # Premium Sinyal Listesi
    st.markdown("### 🎯 PREMIUM AI SİNYALLERİ")
    
    # Filtreler
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        signal_filter = st.selectbox("Sinyal Tipi", ["TÜMÜ", "GÜÇLÜ AL", "AL", "NÖTR", "SAT", "GÜÇLÜ SAT"])
    
    with col2:
        min_confidence = st.slider("Min Güven", 0, 100, 75)
    
    with col3:
        min_score = st.slider("Min Skor", 0, 100, 60)
    
    with col4:
        sort_by = st.selectbox("Sırala", ["SKOR", "GÜVEN", "TREND"])
    
    # Sinyalleri filtrele ve göster
    filtered_signals = filter_and_sort_signals(df, signal_filter, min_confidence, min_score, sort_by)
    display_signal_cards(filtered_signals)
    
    # Seçili sembol detayları
    if st.session_state.get('selected_symbol'):
        display_symbol_details()

def show_welcome_screen():
    """Hoş geldin ekranı"""
    st.markdown("""
    <div style='text-align: center; padding: 50px 20px;'>
        <h1 style='color: #3b82f6; font-size: 3em;'>🎯 MEXC PRO TRADER</h1>
        <p style='color: #9ca3af; font-size: 1.3em;'>Premium AI Sinyal Sistemi</p>
        
        <div style='margin: 40px 0;'>
            <div style='display: inline-block; margin: 10px; padding: 20px; background: rgba(30, 41, 59, 0.8); border-radius: 15px; width: 200px;'>
                <h3>🚀 SCALP</h3>
                <p>1m-15m timeframe<br>Hızlı işlemler</p>
            </div>
            <div style='display: inline-block; margin: 10px; padding: 20px; background: rgba(30, 41, 59, 0.8); border-radius: 15px; width: 200px;'>
                <h3>📊 SWING</h3>
                <p>1h-4h timeframe<br>Orta vadeli</p>
            </div>
            <div style='display: inline-block; margin: 10px; padding: 20px; background: rgba(30, 41, 59, 0.8); border-radius: 15px; width: 200px;'>
                <h3>💎 POSITION</h3>
                <p>4h-1d timeframe<br>Uzun vadeli</p>
            </div>
        </div>
        
        <p style='color: #6b7280;'>Başlamak için sol menüden sembolleri seçin ve Premium Tarama başlatın</p>
    </div>
    """, unsafe_allow_html=True)

def filter_and_sort_signals(df, signal_filter, min_confidence, min_score, sort_by):
    """Sinyalleri filtrele ve sırala"""
    signals = []
    
    for _, row in df.iterrows():
        if not row.get('best_analysis'):
            continue
            
        analysis = row['best_analysis']
        ai_analysis = analysis.get('ai_analysis', {})
        
        signal_info = {
            'symbol': row['symbol'],
            'timeframe': analysis['timeframe'],
            'signal': ai_analysis.get('signal', 'NEUTRAL'),
            'confidence': ai_analysis.get('confidence', 0),
            'score': analysis.get('score', 0),
            'price': analysis.get('price', 0),
            'explanation': ai_analysis.get('explanation', ''),
            'entry': ai_analysis.get('entry'),
            'stop_loss': ai_analysis.get('stop_loss'),
            'take_profit': ai_analysis.get('take_profit'),
            'risk_reward': ai_analysis.get('risk_reward', 0)
        }
        
        # Filtreleme
        if signal_filter != "TÜMÜ" and signal_info['signal'] != signal_filter:
            continue
            
        if signal_info['confidence'] < min_confidence:
            continue
            
        if signal_info['score'] < min_score:
            continue
            
        signals.append(signal_info)
    
    # Sıralama
    if sort_by == "SKOR":
        signals.sort(key=lambda x: x['score'], reverse=True)
    elif sort_by == "GÜVEN":
        signals.sort(key=lambda x: x['confidence'], reverse=True)
    else:
        signals.sort(key=lambda x: x['score'], reverse=True)
    
    return signals

def display_signal_cards(signals):
    """Sinyal kartlarını göster"""
    if not signals:
        st.info("🤔 Filtrelerinize uygun sinyal bulunamadı")
        return
    
    # Sinyal kartları için grid
    cols = st.columns(2)
    
    for idx, signal in enumerate(signals):
        with cols[idx % 2]:
            display_single_signal_card(signal, idx)

def display_single_signal_card(signal, idx):
    """Tek sinyal kartını göster"""
    # Sinyal tipine göre stil
    signal_class = "signal-neutral"
    signal_emoji = "⚪"
    
    if signal['signal'] == 'GÜÇLÜ AL':
        signal_class = "signal-strong-long"
        signal_emoji = "🚀"
    elif signal['signal'] == 'AL':
        signal_class = "signal-moderate"
        signal_emoji = "📈"
    elif signal['signal'] == 'SAT':
        signal_class = "signal-moderate" 
        signal_emoji = "📉"
    elif signal['signal'] == 'GÜÇLÜ SAT':
        signal_class = "signal-strong-short"
        signal_emoji = "🔻"
    
    st.markdown(f"""
    <div class='signal-card-premium {signal_class}'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
            <h3 style='margin: 0; color: white;'>{signal_emoji} {signal['symbol']}</h3>
            <div style='background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 20px;'>
                <strong>{signal['signal']}</strong>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 14px;'>
            <div>⏰ Zaman: <strong>{signal['timeframe']}</strong></div>
            <div>🎯 Güven: <strong>{signal['confidence']}%</strong></div>
            <div>💎 Skor: <strong>{signal['score']}/100</strong></div>
            <div>💰 Fiyat: <strong>${signal['price']:.4f}</strong></div>
        </div>
        
        <div style='margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;'>
            <div style='font-size: 12px; color: #cbd5e1;'>
                🎯 Giriş: <strong>${signal['entry']:.4f if signal['entry'] else 'N/A'}</strong> | 
                🛑 Stop: <strong>${signal['stop_loss']:.4f if signal['stop_loss'] else 'N/A'}</strong> |
                🎯 Hedef: <strong>${signal['take_profit']:.4f if signal['take_profit'] else 'N/A'}</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Butonlar
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Detaylı Analiz", key=f"detail_{idx}", use_container_width=True):
            st.session_state.selected_symbol = signal['symbol']
    with col2:
        if st.button("💾 Sinyali Kaydet", key=f"save_{idx}", use_container_width=True):
            save_signal_to_history(signal)

def display_symbol_details():
    """Seçili sembolün detaylarını göster"""
    symbol = st.session_state.selected_symbol
    df = st.session_state.scan_results
    
    if not symbol or df.empty:
        return
        
    symbol_data = next((row for _, row in df.iterrows() if row['symbol'] == symbol), None)
    if not symbol_data:
        return
        
    st.markdown("---")
    st.markdown(f"### 📊 {symbol} Detaylı Analiz")
    
    analysis = symbol_data['best_analysis']
    ai_analysis = analysis.get('ai_analysis', {})
    technical = analysis.get('technical', {})
    
    # Metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🤖 AI Sinyali", ai_analysis.get('signal', 'N/A'), 
                 f"%{ai_analysis.get('confidence', 0)} Güven")
    
    with col2:
        st.metric("💎 AI Skoru", f"{analysis.get('score', 0)}/100")
    
    with col3:
        st.metric("💰 Mevcut Fiyat", f"${analysis.get('price', 0):.4f}")
    
    with col4:
        st.metric("⏰ Zaman Dilimi", analysis.get('timeframe', 'N/A'))
    
    # Grafik ve detaylar
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not technical.empty:
            fig = create_premium_chart(technical, symbol, analysis['timeframe'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Ticaret planı
        st.markdown("#### 🎯 Ticaret Planı")
        
        if all(k in ai_analysis for k in ['entry', 'stop_loss', 'take_profit']):
            display_trade_plan(ai_analysis)
        
        # AI Analizi
        st.markdown("#### 🤖 AI Analiz Raporu")
        st.write(ai_analysis.get('explanation', 'Analiz mevcut değil'))
        
        # Sinyal kaydetme
        if st.button("💾 Bu Sinyali Kaydet", use_container_width=True):
            save_signal_to_history({
                'symbol': symbol,
                'timeframe': analysis['timeframe'],
                'signal': ai_analysis.get('signal'),
                'confidence': ai_analysis.get('confidence'),
                'entry': ai_analysis.get('entry'),
                'stop_loss': ai_analysis.get('stop_loss'),
                'take_profit': ai_analysis.get('take_profit'),
                'timestamp': datetime.utcnow().isoformat()
            })
            st.success("✅ Sinyal kaydedildi!")

def display_trade_plan(ai_analysis):
    """Ticaret planını göster"""
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
        st.metric("⚖️ Risk/Ödül", f"{rr_ratio:.2f}:1")
        
        # Risk hesaplama
        if st.button("📈 Risk Hesapla", use_container_width=True):
            calculate_risk_management(entry, stop_loss, take_profit)

def calculate_risk_management(entry, stop_loss, take_profit):
    """Risk yönetimi hesaplamaları"""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    rr_ratio = reward / risk if risk > 0 else 0
    
    st.success(f"""
    **💰 Risk Yönetimi:**
    
    - ⚠️ **Risk:** ${risk:.4f} (%{(risk/entry)*100:.2f})
    - 🎯 **Ödül:** ${reward:.4f} (%{(reward/entry)*100:.2f})  
    - ⚖️ **Risk/Ödül Oranı:** {rr_ratio:.2f}:1
    - 📈 **Minimum Başarı Oranı:** %{100/(1+rr_ratio):.1f}
    
    💡 **Değerlendirme:** {'🚀 MÜKEMMEL!' if rr_ratio >= 2 else '✅ İYİ' if rr_ratio >= 1.5 else '⚠️ DİKKAT'}
    """)

def save_signal_to_history(signal):
    """Sinyali geçmişe kaydet"""
    if 'saved_signals' not in st.session_state:
        st.session_state.saved_signals = []
    
    signal['saved_at'] = datetime.utcnow().isoformat()
    st.session_state.saved_signals.append(signal)
    
    # En fazla 50 sinyal sakla
    if len(st.session_state.saved_signals) > 50:
        st.session_state.saved_signals = st.session_state.saved_signals[-50:]

if __name__ == "__main__":
    main()
