# app.py
# Gelişmiş MEXC Vadeli Sinyal Uygulaması - Specter Trend Cloud & AI Hibrit Motor

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
import ai_engine
import trend_cloud
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging

st.set_page_config(page_title="MEXC Pro Sinyal Terminali", layout="wide", page_icon="🚀")

CONTRACT_BASE = "https://contract.mexc.com/api/v1"
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']

# Basit CSS (ayrıntıları önceki versiyonla benzer)
st.markdown("""
<style>
/* küçük stil */
body { background: #071029; color: #e6eef6; }
</style>
""", unsafe_allow_html=True)

# ---------------- API yardımcı fonksiyonlar ----------------
def fetch_json(url, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.warning(f"HTTP hata: {e} - {url} - {params}")
        return {}

def fetch_contract_ticker():
    url = f"{CONTRACT_BASE}/contract/ticker"
    j = fetch_json(url)
    return j.get('data', []) if isinstance(j, dict) else []

def get_top_contracts_by_volume(limit=200):
    data = fetch_contract_ticker()
    def vol(x):
        try:
            return float(x.get('volume24') or x.get('amount24') or 0)
        except Exception:
            return 0
    items = sorted(data, key=vol, reverse=True)
    syms = [it.get('symbol') for it in items[:limit] if it.get('symbol')]
    return [s.replace('_','') for s in syms]

def mexc_symbol_from(symbol: str) -> str:
    s = symbol.strip().upper()
    if '_' in s:
        return s
    if s.endswith('USDT'):
        return s[:-4] + "_USDT"
    # fallback
    return s

def fetch_contract_klines(symbol_mexc, interval_mexc, limit=500):
    """
    MEXC contract kline endpoint kullanımı.
    interval_mexc örn: '1m','5m','15m' vb.
    """
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
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        logging.warning(f"Kline fetch hata: {e}")
        return pd.DataFrame()

def fetch_contract_funding_rate(symbol_mexc):
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"
    j = fetch_json(url)
    data = j.get('data') or {}
    try:
        return {'fundingRate': float(data.get('fundingRate') or 0)}
    except Exception:
        return {'fundingRate': 0.0}

# ---------------- Specter analiz wrapper ----------------
def analyze_with_specter_trend(df, symbol, timeframe, ma_type="EMA", base_length=20, atr_multiplier=1.5):
    if df is None or df.empty or len(df) < 20:
        return None
    try:
        specter_df = trend_cloud.calculate_specter_cloud(df, ma_type, base_length, atr_multiplier)
        if specter_df is None or specter_df.empty:
            return None
        latest = specter_df.iloc[-1]
        prev = specter_df.iloc[-2] if len(specter_df) > 1 else latest
        current_trend = "BULLISH" if latest['trend'] == 1 else "BEARISH"
        trend_strength = abs(latest.get('momentum_strength', 0))
        retest_signals = []
        if latest.get('bullish_retest', False):
            retest_signals.append("BULLISH_RETEST")
        if latest.get('bearish_retest', False):
            retest_signals.append("BEARISH_RETEST")
        levels = {
            'current_price': float(latest['close']),
            'cloud_top': float(latest['ma_upper']),
            'cloud_bottom': float(latest['ma_lower']),
            'short_ma': float(latest['ma_short']),
            'long_ma': float(latest['ma_long'])
        }
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'trend': current_trend,
            'trend_strength': trend_strength,
            'retest_signals': retest_signals,
            'levels': levels,
            'momentum': float(latest.get('momentum_strength', 0)),
            'cloud_data': specter_df
        }
    except Exception as e:
        logging.error(f"Specter analiz hatası {symbol}: {e}")
        return None

# ---------------- Tarama motoru ----------------
@st.cache_data(ttl=120)
def run_advanced_scan(symbols, timeframes, gemini_api_key, top_n=100, ma_type="EMA", base_length=20, atr_multiplier=1.5):
    results = []
    for idx, sym in enumerate(symbols[:top_n]):
        mexc_sym = mexc_symbol_from(sym)
        funding = fetch_contract_funding_rate(mexc_sym)
        best_tf_score = -1
        best_tf = None
        entry = {'symbol': sym, 'details': {}, 'best_analysis': None}
        for tf in timeframes:
            # interval olarak tf gönderiyoruz (ör: '15m', '1h')
            df = fetch_contract_klines(mexc_sym, tf, limit=500)
            if df.empty or len(df) < 30:
                continue
            specter_analysis = analyze_with_specter_trend(df, sym, tf, ma_type, base_length, atr_multiplier)
            indicators_snapshot = trend_cloud.create_ai_snapshot(df, funding)
            ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, specter_analysis or {}, api_key=gemini_api_key)
            combined_score = ai_engine.calculate_combined_score(ai_analysis or {}, specter_analysis or {})
            entry['details'][tf] = {
                'price': float(df['close'].iloc[-1]),
                'specter': specter_analysis,
                'ai_analysis': ai_analysis,
                'combined_score': combined_score,
                'cloud_data': specter_analysis.get('cloud_data') if specter_analysis else None,
                'kline_df_preview': df.tail(3).to_dict(orient='records')  # hızlı inceleme
            }
            if combined_score > best_tf_score:
                best_tf_score = combined_score
                best_tf = tf
                entry['best_analysis'] = {
                    'timeframe': tf,
                    'specter': specter_analysis,
                    'ai_analysis': ai_analysis,
                    'combined_score': combined_score,
                    'cloud_data': specter_analysis.get('cloud_data') if specter_analysis else None
                }
        if entry.get('best_analysis'):
            results.append(entry)
    return pd.DataFrame(results)

# ---------------- Grafik oluşturma ----------------
def create_specter_chart(kline_df: pd.DataFrame, specter_df: pd.DataFrame, symbol, timeframe):
    if kline_df is None or kline_df.empty or specter_df is None or specter_df.empty:
        return None
    fig = make_subplots(rows=2, cols=1, shared_x=True, vertical_spacing=0.06, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=kline_df['timestamp'], open=kline_df['open'], high=kline_df['high'], low=kline_df['low'], close=kline_df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df['ma_upper'], line=dict(width=1, dash='dash'), name='Cloud Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df['ma_lower'], line=dict(width=1, dash='dash'), name='Cloud Lower', fill='tonexty', fillcolor='rgba(0,150,150,0.12)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df['ma_short'], line=dict(width=2), name='Short MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df['ma_long'], line=dict(width=2), name='Long MA'), row=1, col=1)
    bull_retest = specter_df[specter_df.get('bullish_retest', False)==True]
    bear_retest = specter_df[specter_df.get('bearish_retest', False)==True]
    if not bull_retest.empty:
        fig.add_trace(go.Scatter(x=bull_retest['timestamp'], y=bull_retest['low'] * 0.998, mode='markers', marker=dict(symbol='diamond', size=9), name='Bullish Retest'), row=1, col=1)
    if not bear_retest.empty:
        fig.add_trace(go.Scatter(x=bear_retest['timestamp'], y=bear_retest['high'] * 1.002, mode='markers', marker=dict(symbol='diamond', size=9), name='Bearish Retest'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df.get('momentum_strength', [0]*len(specter_df)), name='Momentum'), row=2, col=1)
    fig.update_layout(template='plotly_dark', height=600, showlegend=True, xaxis_rangeslider_visible=False)
    return fig

# ---------------- UI / Main ----------------
def main():
    st.title("🚀 MEXC Pro Sinyal Terminali - (Güncellenmiş)")
    st.sidebar.header("Tarama Ayarları")
    gemini_api_key = st.sidebar.text_input("Gemini API Key (opsiyonel)", type="password")
    mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top by volume (200)", "Custom list"])
    if mode == "Custom list":
        custom = st.sidebar.text_area("Özel Sembol Listesi (virgülle ayır)", value="BTCUSDT,ETHUSDT,ADAUSDT")
        symbols = [s.strip().upper() for s in custom.split(',') if s.strip()]
    else:
        symbols = get_top_contracts_by_volume(200)
    if not symbols:
        st.sidebar.error("Sembol listesi boş.")
        st.stop()
    timeframes = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS)
    top_n = st.sidebar.slider("İlk N Coin Tara", 5, min(200, len(symbols)), value=min(30, len(symbols)))
    # Specter parametreleri
    with st.sidebar.expander("Specter Ayarları"):
        ma_type = st.selectbox("MA Tipi", ["SMA","EMA","WMA","DEMA"], index=1)
        base_length = st.slider("Base Length", 5, 50, 20)
        atr_multiplier = st.slider("ATR Multiplier", 0.5, 3.0, 1.5)
    scan_clicked = st.sidebar.button("🔍 Tarama Başlat")
    if 'scan_results' not in st.session_state:
        st.session_state.scan_results = pd.DataFrame()
    if scan_clicked:
        with st.spinner("Tarama çalıştırılıyor..."):
            st.session_state.scan_results = run_advanced_scan(symbols, timeframes, gemini_api_key, top_n, ma_type, base_length, atr_multiplier)
            st.success("Tarama tamamlandı.")
    df = st.session_state.scan_results
    if df is None or df.empty:
        st.info("Tarama sonucu boş. Sol menüden tarama parametrelerini ayarlayıp 'Tarama Başlat' butonuna tıklayın.")
        return
    # Sinyal gösterimi
    st.header("🔥 AI Sinyal Listesi")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        filter_signal = st.selectbox("Sinyal Türü", ["Tümü","LONG","SHORT","NEUTRAL"])
    with col2:
        min_conf = st.slider("Min Güven", 0, 100, 60)
    with col3:
        sort_by = st.selectbox("Sırala", ["Güven","Kombine Skor"], index=0)
    signals = []
    for _, row in df.iterrows():
        ba = row.get('best_analysis') or {}
        ai = ba.get('ai_analysis') or {}
        specter = ba.get('specter') or {}
        sig = {
            'symbol': row.get('symbol'),
            'timeframe': ba.get('timeframe'),
            'signal': ai.get('signal','NEUTRAL'),
            'confidence': ai.get('confidence', 0),
            'trend_strength': specter.get('trend_strength',0)*100 if specter else 0,
            'combined_score': ba.get('combined_score',0),
            'price': ai.get('entry', 0),
            'explanation': ai.get('explanation',''),
            'cloud_data': ba.get('cloud_data'),
            'ai_analysis': ai
        }
        signals.append(sig)
    filtered = [s for s in signals if s['confidence'] >= min_conf]
    if filter_signal != "Tümü":
        filtered = [s for s in filtered if s['signal'] == filter_signal]
    if sort_by == "Güven":
        filtered.sort(key=lambda x: x['confidence'], reverse=True)
    else:
        filtered.sort(key=lambda x: x['combined_score'], reverse=True)
    # kartlar
    cols = st.columns(3)
    for idx, s in enumerate(filtered[:12]):
        with cols[idx % 3]:
            st.markdown(f"**{s['symbol']} — {s['timeframe']}**")
            st.write(f"**Sinyal:** {s['signal']}  |  **Güven:** {s['confidence']}%")
            st.write(f"**Trend Gücü:** {s['trend_strength']:.0f}%  |  **Fiyat:** {s['price']:.6f}")
            if st.button("Detay", key=f"det_{idx}"):
                st.session_state.selected_symbol = s['symbol']
                st.session_state.selected_entry = s
    # detay paneli
    if 'selected_symbol' in st.session_state:
        sel_symbol = st.session_state.selected_symbol
        st.markdown("---")
        st.subheader(f"Detaylı Analiz — {sel_symbol}")
        sel_entry = st.session_state.get('selected_entry')
        if sel_entry:
            st.write("**AI Açıklama:**")
            st.write(sel_entry.get('explanation','Açıklama yok.'))
            cloud = sel_entry.get('cloud_data')
            # grafiği oluşturmak için kline yeniden çek (daha temiz veri)
            mexc_sym = mexc_symbol_from(sel_symbol)
            tf = sel_entry.get('timeframe') or DEFAULT_TFS[0]
            kline_df = fetch_contract_klines(mexc_sym, tf, limit=500)
            if cloud is None or cloud.empty:
                # olası durum: scan sırasında cloud kaydedilmemiş olabilir, yeniden hesapla
                cloud = trend_cloud.calculate_specter_cloud(kline_df, ma_type, base_length, atr_multiplier)
            fig = create_specter_chart(kline_df, cloud, sel_symbol, tf)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            # ticaret planı
            ai = sel_entry.get('ai_analysis') or {}
            entry = ai.get('entry', 0)
            stop = ai.get('stop_loss', 0)
            tp = ai.get('take_profit', 0)
            if entry and stop and tp:
                risk = abs(entry - stop)
                reward = abs(tp - entry)
                rr = reward / risk if risk != 0 else 0
                st.metric("Giriş", f"{entry:.6f}")
                st.metric("Stop Loss", f"{stop:.6f}")
                st.metric("Take Profit", f"{tp:.6f}")
                st.metric("R/R", f"{rr:.2f}:1")
            else:
                st.warning("AI tarafından net trade seviyleri belirlenemedi.")
    # footer - performans
    st.markdown("---")
    st.write("🔧 Bu araç bir öneri sağlar; ticaret kararı her zaman kullanıcıya aittir.")

if __name__ == "__main__":
    main()
