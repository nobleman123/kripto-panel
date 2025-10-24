# app.py
# Streamlit arayüzü - Scalp / Swing / Long-Term uyumlu, Gemini destekli tarayıcı

import streamlit as st
import pandas as pd
import requests
import time
from typing import List
import trend_cloud
import ai_engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Sinyal Terminali - Strateji Ayrımlı", layout="wide", page_icon="⚡")

CONTRACT_BASE = "https://contract.mexc.com/api/v1"

# --- Helpers ---
def fetch_json(url, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.session_state.last_http_error = str(e)
        return {}

def mexc_symbol_from(symbol: str) -> str:
    s = symbol.strip().upper()
    if '_' in s:
        return s
    if s.endswith('USDT'):
        return s[:-4] + "_USDT"
    return s

def fetch_contract_klines(symbol_mexc: str, interval_mexc: str, limit: int = 500):
    """
    MEXC contract kline endpoint; interval mapping allowed
    """
    # Basit mapping: '1m' -> 'Min1', '15m'->'Min15', '1h'->'Hour1', '1d'->'Day1'
    mapping = {
        '1m':'Min1','3m':'Min3','5m':'Min5','15m':'Min15','30m':'Min30',
        '1h':'Hour1','4h':'Hour4','1d':'Day1','1w':'Week1'
    }
    im = mapping.get(interval_mexc, interval_mexc)
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    j = fetch_json(url, params={'interval': im, 'limit': limit})
    data = j.get('data') or {}
    times = data.get('time', [])
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(data.get('time'), unit='s'),
        'open': data.get('open'),
        'high': data.get('high'),
        'low': data.get('low'),
        'close': data.get('close'),
        'volume': data.get('vol')
    })
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna().reset_index(drop=True)

def fetch_contract_ticker():
    url = f"{CONTRACT_BASE}/contract/ticker"
    j = fetch_json(url)
    return j.get('data', []) if isinstance(j, dict) else []

def get_top_contracts_by_volume(limit=200):
    data = fetch_contract_ticker()
    def vol(x):
        try:
            return float(x.get('volume24') or x.get('amount24') or 0)
        except:
            return 0
    items = sorted(data, key=vol, reverse=True)
    syms = [it.get('symbol') for it in items[:limit] if it.get('symbol')]
    return [s.replace('_','') for s in syms]

# Plot helper
def create_specter_chart(kline_df, specter_df, title="Chart"):
    if kline_df is None or kline_df.empty or specter_df is None or specter_df.empty:
        return None
    fig = make_subplots(rows=2, cols=1, shared_x=True, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=kline_df['timestamp'], open=kline_df['open'], high=kline_df['high'], low=kline_df['low'], close=kline_df['close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df['ma_upper'], mode='lines', name='Cloud Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df['ma_lower'], mode='lines', fill='tonexty', name='Cloud Lower'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df['ma_short'], mode='lines', name='Short MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df['ma_long'], mode='lines', name='Long MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=specter_df['timestamp'], y=specter_df['momentum_strength'], mode='lines', name='Momentum'), row=2, col=1)
    fig.update_layout(template='plotly_dark', height=600, title=title, xaxis_rangeslider_visible=False)
    return fig

# --- UI ---
st.title("📡 Sinyal Terminali — Scalp / Swing / Long-Term")
st.sidebar.header("Ayarlar")

# Gemini key (prefilled with user-provided example)
gemini_key = st.sidebar.text_input("Gemini API Key (opsiyonel)", value="AIzaSyBn9dEGq1cTQqwUq6bnoQSEgmbaiJiRnks", type="password")

strategy_choice = st.sidebar.selectbox("Strateji Tipi (varsayılan tarama)", ["SCALP","SWING","LONG"])
scan_top_n = st.sidebar.slider("Tarama: İlk N Coin", 5, 200, 30)
timeframe_override = st.sidebar.multiselect("Zaman dilimleri (opsiyonel, boş=profile görecek)", options=['1m','3m','5m','15m','30m','1h','4h','1d','1w'], default=[])
scan_mode = st.sidebar.radio("Tarama Hızı", ["Yavaş - Tam Analiz","Hızlı - Hafif Analiz"], index=0)

# Risk / R/R (kullanıcı değiştirebilir)
st.sidebar.subheader("Risk Ayarları (varsayılan agresif)")
risk_scalp = st.sidebar.number_input("Scalp risk fraction (örn 0.04 = %4)", value=0.04, step=0.005, format="%.4f")
rr_scalp = st.sidebar.number_input("Scalp target R/R", value=1.8, step=0.1, format="%.2f")
risk_swing = st.sidebar.number_input("Swing risk fraction", value=0.025, step=0.005, format="%.4f")
rr_swing = st.sidebar.number_input("Swing target R/R", value=2.5, step=0.1, format="%.2f")
risk_long = st.sidebar.number_input("Long risk fraction", value=0.02, step=0.005, format="%.4f")
rr_long = st.sidebar.number_input("Long target R/R", value=3.5, step=0.1, format="%.2f")

apply_defaults = st.sidebar.button("Varsayılan profil uygula")
if apply_defaults:
    st.experimental_rerun()

# Symbols
mode = st.sidebar.radio("Sembol kaynağı", ["Top by volume", "Custom list"], index=0)
if mode == "Custom list":
    custom = st.sidebar.text_area("Özel semboller (virgülle ayrılmış)", value="BTCUSDT,ETHUSDT")
    symbols = [s.strip().upper() for s in custom.split(',') if s.strip()]
else:
    symbols = get_top_contracts_by_volume(200)

if not symbols:
    st.warning("Sembol listesi boş. Ayarları kontrol et.")
    st.stop()

# Scan button
if 'results' not in st.session_state:
    st.session_state.results = []
scan_btn = st.button("🔍 Tarama Başlat (yavaş tam analiz için biraz zaman alır)")
if scan_btn:
    st.session_state.results = []  # reset
    selected_profile = strategy_choice
    timeframes_to_use = timeframe_override or ai_engine.StrategyProfile.PROFILES[selected_profile]['timeframes']
    # update profile with UI risk/RR if user changed
    p = ai_engine.StrategyProfile.PROFILES
    p['SCALP']['risk_fraction'] = float(risk_scalp); p['SCALP']['target_rr'] = float(rr_scalp)
    p['SWING']['risk_fraction'] = float(risk_swing); p['SWING']['target_rr'] = float(rr_swing)
    p['LONG']['risk_fraction'] = float(risk_long); p['LONG']['target_rr'] = float(rr_long)

    status_bar = st.progress(0)
    total = min(len(symbols), scan_top_n)
    for idx, sym in enumerate(symbols[:scan_top_n]):
        status_bar.progress(int((idx/total)*100))
        mexc_sym = mexc_symbol_from(sym)
        best = None
        for tf in timeframes_to_use:
            df = fetch_contract_klines(mexc_sym, tf, limit=500)
            if df.empty or len(df) < 30:
                continue
            # specter
            profile = ai_engine.StrategyProfile.PROFILES[selected_profile]
            specter_df = trend_cloud.calculate_specter_cloud(df, ma_type="EMA", base_length=profile['ma_base'], atr_multiplier=profile['atr_mult'])
            specter_snapshot = {}
            if specter_df is not None and not specter_df.empty:
                last = specter_df.iloc[-1]
                specter_snapshot = {
                    'trend': "BULLISH" if last['trend']==1 else "BEARISH",
                    'trend_strength': abs(last.get('momentum_strength',0)),
                    'has_bullish_retest': bool(last.get('bullish_retest',False)),
                    'has_bearish_retest': bool(last.get('bearish_retest',False)),
                    'atr': float(last.get('atr', 0))
                }
            # indicators snapshot
            indicators = {}
            try:
                indicators['price'] = float(df['close'].iloc[-1])
                indicators['rsi'] = float(ta_rsi(df['close'], 14))
                macd_hist = ta_macd_hist(df['close'])
                indicators['macd_histogram'] = float(macd_hist)
                indicators['atr_percent'] = float((specter_snapshot.get('atr',0) / indicators['price'])*100 if indicators['price'] else 0)
                indicators['volume_ratio'] = float(df['volume'].iloc[-1] / (df['volume'].rolling(20).mean().iloc[-1] if df['volume'].rolling(20).mean().iloc[-1] else 1))
            except Exception:
                indicators = {'price': df['close'].iloc[-1]}
            # combined decision
            decision = ai_engine.get_combined_decision(indicators, specter_snapshot, selected_profile, api_key=gemini_key if gemini_key else None)
            # attach tf info
            decision_entry = {
                'symbol': sym,
                'mexc_symbol': mexc_sym,
                'timeframe': tf,
                'decision': decision,
                'indicators': indicators,
                'specter': specter_snapshot
            }
            # choose best by confidence
            conf = decision.get('confidence', 0)
            if best is None or conf > best.get('decision',{}).get('confidence', -1):
                best = decision_entry
            # if scan_mode is fast, break early
            if scan_mode.startswith("Hızlı"):
                time.sleep(0.05)
        if best:
            st.session_state.results.append(best)
        time.sleep(0.2)  # be polite to API
    status_bar.progress(100)
    st.success("Tarama tamamlandı.")

# Utility: small wrappers for pandas_ta indicators to avoid import at top
def ta_rsi(series, length=14):
    import pandas_ta as ta
    r = ta.rsi(series, length=length)
    return r.iloc[-1] if not r.empty else 50.0

def ta_macd_hist(series):
    import pandas_ta as ta
    macd = ta.macd(series)
    if macd is None or macd.empty:
        return 0.0
    vals = macd.iloc[-1].to_list()
    return vals[2] if len(vals) > 2 else 0.0

# Results display
if st.session_state.results:
    st.header("🔎 Sinyal Sonuçları")
    df_res = st.session_state.results
    # simple filter
    min_conf = st.slider("Min confidence", 0, 100, 60)
    filtered = [r for r in df_res if r['decision'].get('confidence',0) >= min_conf]
    # sort by confidence
    filtered = sorted(filtered, key=lambda x: x['decision'].get('confidence',0), reverse=True)
    for idx, item in enumerate(filtered[:50]):
        dec = item['decision']
        st.subheader(f"{item['symbol']} — {item['timeframe']} — {dec.get('signal')} ({dec.get('confidence')}%)")
        st.write(f"Strategy: {strategy_choice} | Source: {dec.get('ai_source')}")
        st.write(dec.get('explanation',''))
        lv = dec.get('levels', {})
        e = lv.get('entry') or lv.get('entry',0)
        sl = lv.get('stop_loss',0)
        tp = lv.get('take_profit', lv.get('take_profit',0))
        rr = lv.get('rr', 0)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Giriş", f"{e:.6f}")
        col2.metric("Stop", f"{sl:.6f}")
        col3.metric("TP", f"{tp:.6f}")
        col4.metric("R/R", f"{rr:.2f}:1")
        if st.button(f"Grafik göster - {item['symbol']}_{idx}", key=f"g_{idx}"):
            df = fetch_contract_klines(item['mexc_symbol'], item['timeframe'], limit=500)
            profile = ai_engine.StrategyProfile.PROFILES[strategy_choice]
            specter_df = trend_cloud.calculate_specter_cloud(df, ma_type="EMA", base_length=profile['ma_base'], atr_multiplier=profile['atr_mult'])
            fig = create_specter_chart(df, specter_df, title=f"{item['symbol']} {item['timeframe']}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.write("AI Açıklama:")
            st.write(dec.get('explanation',''))
            # save record button
            if st.button(f"Kayıt et - {item['symbol']}_{idx}", key=f"save_{idx}"):
                to_save = {
                    'symbol': item['symbol'],
                    'timeframe': item['timeframe'],
                    'decision': dec,
                    'indicators': item['indicators'],
                    'specter': item['specter']
                }
                ai_engine.save_record(to_save)
                st.success("Kayıt edildi.")
else:
    st.info("Henüz tarama yapılmadı. Sol menüden ayarları yapıp 'Tarama Başlat' butonuna tıklayın.")
