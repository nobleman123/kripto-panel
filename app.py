# app.py
# MEXC Profesyonel Sinyal Paneli - Çoklu İndikatör & AI Hibrit Sistem

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
import ai_engine
import technical_indicators
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':7,'vol':10,'funding':30,'nw':8}

# Modern CSS Styling
st.markdown("""
<style>
    /* Ana tema */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Sinyal kartları */
    .signal-card {
        background: rgba(30, 41, 59, 0.9);
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid;
        margin: 8px 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .signal-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
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
        background: linear-gradient(135deg, rgba(107, 114, 128, 0.1), rgba(30, 41, 59, 0.9));
    }
    
    .signal-short {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(30, 41, 59, 0.9));
    }
    
    .signal-strong-short {
        border-left-color: #dc2626;
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.1), rgba(30, 41, 59, 0.9));
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.05);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- API FONKSİYONLARI ----------------
def fetch_json(url, params=None, timeout=8):
    """Güvenli API çağrısı"""
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API Error {url}: {str(e)}")
        return {}

def fetch_contract_ticker():
    """Contract ticker verilerini getir"""
    url = f"{CONTRACT_BASE}/contract/ticker"
    try:
        data = fetch_json(url)
        return data.get('data', [])
    except Exception as e:
        logger.error(f"Ticker fetch error: {str(e)}")
        return []

def get_top_contracts_by_volume(limit=200):
    """Hacme göre en iyi coinleri getir"""
    try:
        data = fetch_contract_ticker()
        if not data:
            return ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT", "LINKUSDT"]
        
        def vol(x):
            return float(x.get('volume24') or x.get('amount24') or 0)
        
        items = sorted(data, key=vol, reverse=True)
        syms = [it.get('symbol') for it in items[:limit]]
        return [s.replace('_', '') for s in syms if s]
    except Exception as e:
        logger.error(f"Top contracts error: {str(e)}")
        return ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]

def mexc_symbol_from(symbol: str) -> str:
    """Sembol formatını dönüştür"""
    s = symbol.strip().upper()
    if '_' in s: 
        return s
    if s.endswith('USDT'): 
        return s[:-4] + "_USDT"
    return s

def fetch_contract_klines(symbol_mexc, interval_mexc, limit=200):
    """Kline verilerini getir"""
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    try:
        data = fetch_json(url, params={'interval': interval_mexc, 'limit': limit})
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
        logger.error(f"Kline error {symbol_mexc}: {str(e)}")
        return pd.DataFrame()

def fetch_contract_funding_rate(symbol_mexc):
    """Funding rate verilerini getir"""
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"
    try:
        data = fetch_json(url)
        funding_data = data.get('data', {})
        return {'fundingRate': float(funding_data.get('fundingRate') or 0)}
    except Exception:
        return {'fundingRate': 0.0}

# ---------------- İNDİKATÖR HESAPLAMA ----------------
def compute_indicators(df):
    """Temel indikatörleri hesapla"""
    if df.empty or len(df) < 50:
        return pd.DataFrame()
        
    df = df.copy()
    
    try:
        # Moving Averages
        df['ema20'] = ta.ema(df['close'], length=20)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['ema200'] = ta.ema(df['close'], length=200)
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_histogram'] = macd['MACDh_12_26_9']
        
        # RSI
        df['rsi14'] = ta.rsi(df['close'], length=14)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        if bb is not None:
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0'] 
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ADX
        adx = ta.adx(df['high'], df['low'], df['close'])
        if adx is not None:
            df['adx'] = adx['ADX_14']
        
        # ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Volume indicators
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['momentum'] = ta.mom(df['close'], length=10)
        
        return df.dropna()
        
    except Exception as e:
        logger.error(f"Indicator error: {str(e)}")
        return pd.DataFrame()

# ---------------- SKORLAMA SİSTEMİ ----------------
def score_signals(latest, prev, funding, weights):
    """Sinyalleri skorla"""
    per = {}
    reasons = []
    total = 0
    
    try:
        # EMA Skoru
        w = weights.get('ema', 20)
        contrib = 0
        if latest['ema20'] > latest['ema50'] > latest['ema200']:
            contrib = +w
            reasons.append("EMA yükseliş dizilimi")
        elif latest['ema20'] < latest['ema50'] < latest['ema200']:
            contrib = -w
            reasons.append("EMA düşüş dizilimi")
        per['ema'] = contrib
        total += contrib
    except:
        per['ema'] = 0
    
    try:
        # MACD Skoru
        w = weights.get('macd', 15)
        p_hist = float(prev.get('macd_histogram', 0))
        l_hist = float(latest.get('macd_histogram', 0))
        contrib = 0
        if p_hist < 0 and l_hist > 0:
            contrib = w
            reasons.append("MACD yukarı kesişim")
        elif p_hist > 0 and l_hist < 0:
            contrib = -w
            reasons.append("MACD aşağı kesişim")
        elif l_hist > 0:
            contrib = w * 0.5
        elif l_hist < 0:
            contrib = -w * 0.5
        per['macd'] = contrib
        total += contrib
    except:
        per['macd'] = 0
    
    try:
        # RSI Skoru
        w = weights.get('rsi', 12)
        rsi = float(latest.get('rsi14', 50))
        contrib = 0
        if rsi < 30:
            contrib = w
            reasons.append("RSI aşırı satım")
        elif rsi > 70:
            contrib = -w
            reasons.append("RSI aşırı alım")
        elif rsi < 45:
            contrib = w * 0.5
        elif rsi > 55:
            contrib = -w * 0.5
        per['rsi'] = contrib
        total += contrib
    except:
        per['rsi'] = 0
    
    try:
        # Bollinger Bands Skoru
        w = weights.get('bb', 8)
        bb_pos = latest.get('bb_position', 0.5)
        contrib = 0
        if bb_pos > 0.8:
            contrib = -w
            reasons.append("BB üst bandına dokundu")
        elif bb_pos < 0.2:
            contrib = w
            reasons.append("BB alt bandına dokundu")
        per['bb'] = contrib
        total += contrib
    except:
        per['bb'] = 0
    
    try:
        # Volume Skoru
        w = weights.get('vol', 6)
        vol_ratio = float(latest.get('volume_ratio', 1))
        contrib = 0
        if vol_ratio > 1.5:
            contrib = w
            reasons.append("Hacim patlaması")
        elif vol_ratio < 0.5:
            contrib = -w
            reasons.append("Hacim düşüşü")
        per['vol'] = contrib
        total += contrib
    except:
        per['vol'] = 0
    
    try:
        # Funding Rate Skoru
        w = weights.get('funding', 20)
        fr = funding.get('fundingRate', 0.0)
        contrib = 0
        if fr > 0.0005:
            contrib = -w
            reasons.append("Pozitif funding - Short baskı")
        elif fr < -0.0005:
            contrib = w
            reasons.append("Negatif funding - Long baskı")
        per['funding'] = contrib
        total += contrib
    except:
        per['funding'] = 0
    
    try:
        # Momentum Skoru
        w = weights.get('nw', 8)
        momentum = float(latest.get('momentum', 0))
        contrib = 0
        if momentum > 0:
            contrib = w * 0.5
        elif momentum < 0:
            contrib = -w * 0.5
        per['momentum'] = contrib
        total += contrib
    except:
        per['momentum'] = 0
    
    total = int(max(min(total, 100), -100))
    return total, per, reasons

def label_from_score(score, thresholds):
    """Skora göre sinyal etiketi belirle"""
    if score is None:
        return "NÖTR"
    
    strong_buy_t, buy_t, sell_t, strong_sell_t = thresholds
    
    if score >= strong_buy_t:
        return "GÜÇLÜ AL"
    if score >= buy_t:
        return "AL"
    if score <= strong_sell_t:
        return "GÜÇLÜ SAT"
    if score <= sell_t:
        return "SAT"
    return "NÖTR"

# ---------------- TARAMA MOTORU ----------------
@st.cache_data(ttl=120, show_spinner=False)
def run_scan(symbols, timeframes, weights, thresholds, gemini_api_key, top_n=100):
    """Ana tarama motoru"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, sym in enumerate(symbols[:top_n]):
        status_text.text(f"🔍 Analiz: {sym} ({idx+1}/{len(symbols[:top_n])})")
        progress_bar.progress((idx + 1) / len(symbols[:top_n]))
        
        entry = {
            'symbol': sym,
            'details': {},
            'best_score': -1000,
            'best_timeframe': None,
            'buy_count': 0,
            'strong_buy_count': 0,
            'sell_count': 0
        }
        
        mexc_sym = mexc_symbol_from(sym)
        funding = fetch_contract_funding_rate(mexc_sym)
        
        for tf in timeframes:
            interval = INTERVAL_MAP.get(tf)
            if not interval:
                continue
                
            df = fetch_contract_klines(mexc_sym, interval, 200)
            if df.empty or len(df) < 50:
                continue
            
            df_ind = compute_indicators(df)
            if df_ind.empty or len(df_ind) < 3:
                continue
            
            latest = df_ind.iloc[-1]
            prev = df_ind.iloc[-2] if len(df_ind) > 1 else latest
            
            # Temel skorlama
            score, per_scores, reasons = score_signals(latest, prev, funding, weights)
            label = label_from_score(score, thresholds)
            
            # AI Analizi (opsiyonel)
            ai_analysis = None
            if gemini_api_key:
                try:
                    indicators_snapshot = {
                        'score': int(score),
                        'price': float(latest['close']),
                        'rsi14': float(latest.get('rsi14', 50)),
                        'macd_histogram': float(latest.get('macd_histogram', 0)),
                        'volume_ratio': float(latest.get('volume_ratio', 1)),
                        'atr': float(latest.get('atr', 0)),
                        'ema_alignment': 1 if latest['ema20'] > latest['ema50'] > latest['ema200'] else -1
                    }
                    ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=gemini_api_key)
                except Exception as e:
                    logger.warning(f"AI analysis failed for {sym}: {str(e)}")
            
            entry['details'][tf] = {
                'score': int(score),
                'label': label,
                'price': float(latest['close']),
                'per_scores': per_scores,
                'reasons': reasons,
                'ai_analysis': ai_analysis
            }
            
            # En iyi skoru güncelle
            if score > entry['best_score']:
                entry['best_score'] = score
                entry['best_timeframe'] = tf
            
            # Sinyal sayılarını güncelle
            if label in ['AL', 'GÜÇLÜ AL']:
                entry['buy_count'] += 1
            if label == 'GÜÇLÜ AL':
                entry['strong_buy_count'] += 1
            if label in ['SAT', 'GÜÇLÜ SAT']:
                entry['sell_count'] += 1
        
        if entry['details']:
            results.append(entry)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

# ---------------- GÖRSELLEŞTİRME ----------------
def show_tradingview(symbol: str, interval_tv: str, height: int = 450):
    """TradingView widget göster"""
    uid = f"tv_{symbol.replace('/', '_')}_{interval_tv}"
    
    html_code = f"""
    <div class="tradingview-widget-container" style="height:{height}px;">
      <div id="{uid}"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "container_id": "{uid}",
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
    components.html(html_code, height=height)

def create_indicator_chart(per_scores):
    """İndikatör skor grafiği"""
    if not per_scores:
        return None
        
    try:
        df_plot = pd.DataFrame(list(per_scores.items()), columns=['Indicator', 'Score'])
        df_plot = df_plot.sort_values('Score', ascending=True)
        
        fig = go.Figure()
        
        colors = ['red' if x < 0 else 'green' for x in df_plot['Score']]
        
        fig.add_trace(go.Bar(
            y=df_plot['Indicator'],
            x=df_plot['Score'],
            orientation='h',
            marker_color=colors,
            text=df_plot['Score'],
            textposition='auto'
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            template='plotly_dark',
            showlegend=False,
            xaxis_range=[-30, 30]
        )
        
        return fig
    except Exception as e:
        logger.error(f"Chart error: {str(e)}")
        return None

# ---------------- ANA UYGULAMA ----------------
def main():
    try:
        # Header
        st.markdown("""
        <div class='main-header'>
            <h1 style='margin:0; color:white;'>🚀 MEXC Pro Sinyal Terminali</h1>
            <p style='margin:0; color:#e0f2fe;'>Çoklu İndikatör & AI Hibrit Sinyal Sistemi</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("### ⚙️ Tarama Ayarları")
            
            # API Key
            gemini_api_key = st.text_input(
                "🔑 Gemini API Key (Opsiyonel)",
                type="password",
                help="Daha gelişmiş analiz için API key girin"
            )
            
            # Sembol seçimi
            mode = st.selectbox(
                "📊 Sembol Kaynağı",
                ["Top 200 by volume", "Custom list"]
            )
            
            if mode == "Custom list":
                custom_text = st.text_area(
                    "📝 Sembol Listesi",
                    value="BTCUSDT,ETHUSDT,ADAUSDT,SOLUSDT,MATICUSDT,AVAXUSDT,DOTUSDT,LINKUSDT",
                    help="Virgülle ayırarak sembolleri girin"
                )
                symbols = [s.strip().upper() for s in custom_text.split(',') if s.strip()]
            else:
                symbols = get_top_contracts_by_volume(200)
            
            if not symbols:
                st.error("❌ Sembol listesi boş")
                return
            
            # Zaman dilimleri
            timeframes = st.multiselect(
                "⏰ Zaman Dilimleri",
                options=ALL_TFS,
                default=DEFAULT_TFS
            )
            
            if not timeframes:
                st.error("❌ En az bir zaman dilimi seçin")
                return
            
            # Tarama limiti
            top_n = st.slider(
                "🔢 İlk N Coin Taransın",
                min_value=5,
                max_value=min(100, len(symbols)),
                value=min(50, len(symbols))
            )
            
            # Ağırlık ayarları
            with st.expander("🎯 İndikatör Ağırlıkları"):
                w_ema = st.slider("EMA", 0, 30, 20)
                w_macd = st.slider("MACD", 0, 25, 15)
                w_rsi = st.slider("RSI", 0, 20, 12)
                w_bb = st.slider("BB", 0, 15, 8)
                w_vol = st.slider("VOL", 0, 15, 6)
                w_funding = st.slider("FUNDING", 0, 40, 25)
                w_momentum = st.slider("MOMENTUM", 0, 15, 8)
            
            weights = {
                'ema': w_ema, 'macd': w_macd, 'rsi': w_rsi, 
                'bb': w_bb, 'vol': w_vol, 'funding': w_funding, 
                'nw': w_momentum
            }
            
            # Eşik ayarları
            with st.expander("📈 Sinyal Eşikleri"):
                strong_buy_t = st.slider("GÜÇLÜ AL ≥", 10, 100, 60)
                buy_t = st.slider("AL ≥", 0, 80, 20)
                sell_t = st.slider("SAT ≤", -80, 0, -20)
                strong_sell_t = st.slider("GÜÇLÜ SAT ≤", -100, -10, -60)
            
            thresholds = (strong_buy_t, buy_t, sell_t, strong_sell_t)
            
            # Tarama butonu
            scan_clicked = st.button(
                "🚀 TARAMA BAŞLAT",
                type="primary",
                use_container_width=True
            )
        
        # Session state
        if 'scan_results' not in st.session_state:
            st.session_state.scan_results = pd.DataFrame()
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = None
        
        # Tarama işlemi
        if scan_clicked:
            with st.spinner("🔍 Coinler taranıyor... Bu işlem birkaç dakika sürebilir"):
                try:
                    st.session_state.scan_results = run_scan(
                        symbols, timeframes, weights, thresholds, gemini_api_key, top_n
                    )
                    st.session_state.last_scan = datetime.now()
                    st.success(f"✅ Tarama tamamlandı! {len(st.session_state.scan_results)} coin analiz edildi")
                except Exception as e:
                    st.error(f"❌ Tarama hatası: {str(e)}")
        
        # Sonuçları göster
        display_results()
        
    except Exception as e:
        logger.error(f"Main app error: {str(e)}")
        st.error("Uygulamada bir hata oluştu. Lütfen sayfayı yenileyin.")

def display_results():
    """Tarama sonuçlarını göster"""
    df = st.session_state.scan_results
    
    if df.empty:
        show_welcome_message()
        return
    
    # İstatistikler
    total_coins = len(df)
    strong_buy_signals = sum(df['strong_buy_count'])
    buy_signals = sum(df['buy_count'])
    sell_signals = sum(df['sell_count'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Toplam Coin", total_coins)
    with col2:
        st.metric("🚀 Güçlü Al Sinyali", strong_buy_signals)
    with col3:
        st.metric("📈 Al Sinyali", buy_signals)
    with col4:
        st.metric("📉 Sat Sinyali", sell_signals)
    
    # Sinyal listesi
    st.markdown("### 🎯 Aktif Sinyaller")
    
    # Filtreler
    col1, col2 = st.columns([2, 1])
    with col1:
        filter_signal = st.selectbox(
            "Sinyal Filtresi",
            ["Tümü", "GÜÇLÜ AL", "AL", "NÖTR", "SAT", "GÜÇLÜ SAT"]
        )
    with col2:
        min_confidence = st.slider("Min Skor", -100, 100, -100)
    
    # Sinyal kartları
    display_signal_cards(df, filter_signal, min_confidence)
    
    # Seçili sembol detayları
    if st.session_state.get('selected_symbol'):
        display_symbol_details()

def show_welcome_message():
    """Hoş geldin mesajı"""
    st.info("""
    ## 🎯 MEXC Pro Sinyal Terminali
    
    **Özellikler:**
    - 🤖 **AI Destekli Analiz** - Gemini AI entegrasyonu
    - 📊 **Çoklu İndikatör** - EMA, MACD, RSI, Bollinger Bands
    - ⏰ **Multi-Timeframe** - 1m'den 1güne kadar analiz
    - 🎯 **Güçlü Sinyaller** - Net AL/SAT önerileri
    - 📈 **Gerçek Zamanlı** - Anlık piyasa verileri
    
    **Başlamak için:**
    1. Sol menüden sembol kaynağını seçin
    2. Zaman dilimlerini belirleyin  
    3. Tarama ayarlarını yapın
    4. "TARAMA BAŞLAT" butonuna tıklayın
    """)

def display_signal_cards(df, filter_signal, min_confidence):
    """Sinyal kartlarını göster"""
    signals_found = 0
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        best_tf = row.get('best_timeframe')
        best_score = row.get('best_score', 0)
        
        if best_tf is None:
            continue
            
        details = row['details'].get(best_tf, {})
        label = details.get('label', 'NÖTR')
        price = details.get('price', 0)
        
        # Filtreleme
        if filter_signal != "Tümü" and label != filter_signal:
            continue
            
        if best_score < min_confidence:
            continue
        
        # Sinyal kartı
        display_signal_card(symbol, best_tf, label, best_score, price, details, signals_found)
        signals_found += 1
    
    if signals_found == 0:
        st.warning("🤔 Filtrelerinize uygun sinyal bulunamadı")

def display_signal_card(symbol, timeframe, label, score, price, details, idx):
    """Tek sinyal kartını göster"""
    # Sinyal tipine göre stil
    signal_class = "signal-neutral"
    emoji = "⚪"
    
    if label == "GÜÇLÜ AL":
        signal_class = "signal-strong-long"
        emoji = "🚀"
    elif label == "AL":
        signal_class = "signal-long" 
        emoji = "📈"
    elif label == "SAT":
        signal_class = "signal-short"
        emoji = "📉"
    elif label == "GÜÇLÜ SAT":
        signal_class = "signal-strong-short"
        emoji = "🔻"
    
    st.markdown(f"""
    <div class='signal-card {signal_class}'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
            <h4 style='margin: 0; color: white;'>{emoji} {symbol}</h4>
            <div style='background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 15px; font-size: 12px;'>
                <strong>{label}</strong>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; color: #cbd5e1;'>
            <div>⏰ Zaman: <strong>{timeframe}</strong></div>
            <div>💎 Skor: <strong>{score}</strong></div>
            <div>💰 Fiyat: <strong>${price:.4f}</strong></div>
            <div>📊 Güven: <strong>{abs(score)}%</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detay butonu
    if st.button("📊 Detaylı Analiz", key=f"btn_{idx}", use_container_width=True):
        st.session_state.selected_symbol = symbol

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
    
    best_tf = symbol_data.get('best_timeframe')
    details = symbol_data['details'].get(best_tf, {})
    ai_analysis = details.get('ai_analysis', {})
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # TradingView grafiği
        if best_tf:
            interval_tv = TV_INTERVAL_MAP.get(best_tf, '60')
            show_tradingview(symbol, interval_tv)
    
    with col2:
        # Sinyal bilgileri
        st.markdown("#### 🎯 Sinyal Bilgisi")
        
        st.metric("Sinyal", details.get('label', 'NÖTR'))
        st.metric("Skor", details.get('score', 0))
        st.metric("Fiyat", f"${details.get('price', 0):.4f}")
        st.metric("Zaman Dilimi", best_tf)
        
        # İndikatör grafiği
        per_scores = details.get('per_scores', {})
        if per_scores:
            fig = create_indicator_chart(per_scores)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # AI Analizi
        if ai_analysis:
            st.markdown("#### 🤖 AI Analizi")
            st.write(ai_analysis.get('explanation', 'AI analizi mevcut değil'))
            
            if ai_analysis.get('entry'):
                st.markdown("**🎯 Ticaret Seviyeleri:**")
                st.write(f"Giriş: ${ai_analysis['entry']:.4f}")
                st.write(f"Stop: ${ai_analysis.get('stop_loss', 0):.4f}")
                st.write(f"Hedef: ${ai_analysis.get('take_profit', 0):.4f}")
    
    # Sinyal nedenleri
    reasons = details.get('reasons', [])
    if reasons:
        st.markdown("#### 📋 Sinyal Nedenleri")
        for reason in reasons:
            st.write(f"• {reason}")

if __name__ == "__main__":
    main()
