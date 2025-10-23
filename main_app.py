# main_app.py
# Main Streamlit app for MEXC contract scanning + AI suggestions.
import time
import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
import ai_engine

# optional plotly
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

st.set_page_config(page_title="MEXC Vadeli — Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="expanded")

# ---------------- CONFIG ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','8h':'Hour8','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','8h':'480','1d':'D'}
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','8h','1d']
DEFAULT_WEIGHTS = {'ema':25, 'macd':20, 'rsi':15, 'bb':10, 'adx':7, 'vol':10, 'funding':30, 'nw':8}

# ---------------- UI CSS ----------------
st.markdown("""
<style>
body { background: #0b0f14; color: #e6eef6; }
.block { background: linear-gradient(180deg,#0c1116,#071018); padding:10px; border-radius:10px; border:1px solid rgba(255,255,255,0.04); margin-bottom:8px;}
.small-muted { color:#9aa3b2; font-size:12px; }
.score-big { font-size:18px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers / MEXC ----------------
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
    return [s.replace('_','') for s in syms]

def mexc_symbol_from(symbol: str) -> str:
    s = symbol.strip().upper()
    if '_' in s: return s
    if s.endswith('USDT'): return s[:-4] + "_USDT"
    return s[:-4] + "_" + s[-4:]

def fetch_contract_klines(symbol_mexc, interval_mexc):
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    try:
        j = fetch_json(url, params={'interval': interval_mexc})
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
        return df
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

# ---------------- Indicators & scoring ----------------
def nw_smooth(series, bandwidth=8):
    y = np.asarray(series)
    n = len(y)
    if n == 0: return np.array([])
    sm = np.zeros(n)
    for i in range(n):
        distances = np.arange(n) - i
        bw = max(1, bandwidth)
        weights = np.exp(-0.5 * (distances / bw)**2)
        sm[i] = np.sum(weights * y) / (np.sum(weights) + 1e-12)
    return sm

def compute_indicators(df):
    df = df.copy()
    try:
        df['ema20'] = ta.ema(df['close'], length=20)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['ema200'] = ta.ema(df['close'], length=200)
    except Exception:
        df[['ema20','ema50','ema200']] = np.nan
    try:
        macd = ta.macd(df['close'])
        df['macd_hist'] = macd.iloc[:,1] if isinstance(macd, pd.DataFrame) and macd.shape[1]>=2 else np.nan
    except Exception:
        df['macd_hist'] = np.nan
    try:
        df['rsi14'] = ta.rsi(df['close'], length=14)
    except Exception:
        df['rsi14'] = np.nan
    try:
        bb = ta.bbands(df['close'])
        if isinstance(bb, pd.DataFrame) and bb.shape[1]>=3:
            df['bb_lower'] = bb.iloc[:,0]; df['bb_mid'] = bb.iloc[:,1]; df['bb_upper'] = bb.iloc[:,2]
        else:
            df[['bb_lower','bb_mid','bb_upper']] = np.nan
    except Exception:
        df[['bb_lower','bb_mid','bb_upper']] = np.nan
    try:
        adx = ta.adx(df['high'], df['low'], df['close'])
        df['adx14'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx.columns else np.nan
    except Exception:
        df['adx14'] = np.nan
    try:
        df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    except Exception:
        df['atr14'] = np.nan
    try:
        df['vol_ma_short'] = ta.sma(df['volume'], length=20)
        df['vol_ma_long'] = ta.sma(df['volume'], length=50)
        df['vol_osc'] = (df['vol_ma_short'] - df['vol_ma_long']) / (df['vol_ma_long'] + 1e-9)
    except Exception:
        df['vol_osc'] = np.nan
    try:
        sm = nw_smooth(df['close'].values, bandwidth=8)
        if len(sm) == len(df):
            df['nw_smooth'] = sm
            df['nw_slope'] = pd.Series(sm).diff().fillna(0)
        else:
            df['nw_smooth'] = np.nan; df['nw_slope'] = np.nan
    except Exception:
        df['nw_smooth'] = np.nan; df['nw_slope'] = np.nan
    df = df.dropna()
    return df

def score_signals(latest, prev, funding, weights):
    per = {}
    reasons = []
    total = 0
    try:
        atr = float(latest.get('atr14', 0)) if not pd.isna(latest.get('atr14', 0)) else 0.0
    except Exception:
        atr = 0.0
    try:
        price = float(latest.get('close', 0)) if not pd.isna(latest.get('close', 0)) else 0.0
    except Exception:
        price = 0.0

    # EMA
    try:
        w = weights.get('ema', 20)
        contrib = 0
        if latest['ema20'] > latest['ema50'] > latest['ema200']:
            contrib = +w; reasons.append("EMA alignment bullish")
        elif latest['ema20'] < latest['ema50'] < latest['ema200']:
            contrib = -w; reasons.append("EMA alignment bearish")
        per['ema'] = contrib; total += contrib
    except Exception:
        per['ema'] = 0

    # MACD
    try:
        w = weights.get('macd', 15)
        p_h = float(prev.get('macd_hist', 0)); l_h = float(latest.get('macd_hist', 0))
        contrib = 0
        if p_h < 0 and l_h > 0:
            contrib = w; reasons.append("MACD crossover bullish")
        elif p_h > 0 and l_h < 0:
            contrib = -w; reasons.append("MACD crossover bearish")
        per['macd'] = contrib; total += contrib
    except Exception:
        per['macd'] = 0

    # RSI
    try:
        w = weights.get('rsi', 12)
        rsi = float(latest.get('rsi14', np.nan))
        if rsi < 30:
            contrib = w; reasons.append("RSI oversold")
        elif rsi > 70:
            contrib = -w; reasons.append("RSI overbought")
        else:
            contrib = 0
        per['rsi'] = contrib; total += contrib
    except Exception:
        per['rsi'] = 0

    # Bollinger
    try:
        w = weights.get('bb', 8)
        if latest['close'] > latest['bb_upper']: contrib = w; reasons.append("Price above BB upper")
        elif latest['close'] < latest['bb_lower']: contrib = -w; reasons.append("Price below BB lower")
        else: contrib = 0
        per['bb'] = contrib; total += contrib
    except Exception:
        per['bb'] = 0

    # ADX
    try:
        w = weights.get('adx', 6)
        adx = float(latest.get('adx14', 0))
        if adx > 35: contrib = w; reasons.append("ADX strong trend")
        elif adx > 25: contrib = int(w*0.6)
        else: contrib = 0
        per['adx'] = contrib; total += contrib
    except Exception:
        per['adx'] = 0

    # Volume
    try:
        w = weights.get('vol', 6)
        vol_osc = float(latest.get('vol_osc', 0))
        if vol_osc > 0.5: contrib = w; reasons.append("Volume spike")
        elif vol_osc < -0.5: contrib = -w; reasons.append("Volume drop")
        else: contrib = 0
        per['vol'] = contrib; total += contrib
    except Exception:
        per['vol'] = 0

    # NW slope
    try:
        w = weights.get('nw', 8)
        nw_s = float(latest.get('nw_slope', 0))
        if nw_s > 0: contrib = w; reasons.append("NW slope +")
        elif nw_s < 0: contrib = -w; reasons.append("NW slope -")
        else:
