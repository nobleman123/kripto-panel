# app.py  -- MEXC Futures (Contract) Streamlit scanner
import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="MEXC Vadeli Sinyal Paneli", layout="wide", initial_sidebar_state="expanded")

# ----------------- CONFIG -----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
# interval mapping user TF -> MEXC API interval
INTERVAL_MAP = {
    '1m': 'Min1', '5m': 'Min5', '15m': 'Min15', '30m': 'Min30',
    '1h': 'Min60', '4h': 'Hour4', '8h': 'Hour8', '1d': 'Day1'
}
DEFAULT_TFS = ['15m', '1h', '4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','8h','1d']

# ----------------- HELPERS -----------------
def mexc_symbol_from(symbol):
    """Convert BTCUSDT -> BTC_USDT (MEXC contract format)."""
    s = symbol.upper()
    if '_' in s:
        return s
    # assume quote is USDT typically
    if s.endswith('USDT'):
        return s[:-4] + "_USDT"
    # fallback: insert underscore before last 4 chars
    return s[:-4] + "_"+ s[-4:]

def safe_int_or_dash(v):
    if v is None: return '-'
    try:
        if pd.isna(v): return '-'
    except:
        pass
    try:
        return int(v)
    except:
        return '-'

# ----------------- MEXC contract market functions -----------------
def fetch_contract_ticker():
    """Return list of contract tickers (market snapshot)."""
    url = f"{CONTRACT_BASE}/contract/ticker"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    j = r.json()
    # docs: response { success:true, code:0, data: [ {symbol, lastPrice, volume24, ...}, ... ] }
    return j.get('data', [])

def get_top_contracts_by_volume(limit=100):
    """Return top symbols by 24h volume (contract market)."""
    data = fetch_contract_ticker()
    # some entries may have 'volume24' or 'amount24' depending on docs; try both
    def vol(x):
        return float(x.get('volume24') or x.get('amount24') or 0)
    sorted_symbols = sorted(data, key=vol, reverse=True)
    return [item.get('symbol') for item in sorted_symbols[:limit]]

def fetch_contract_klines(symbol_mexc, interval_mexc):
    """
    Get kline for contract symbol (MEXC format e.g. BTC_USDT).
    Uses endpoint: GET /api/v1/contract/kline/{symbol}?interval=Min15
    Returns DataFrame with columns timestamp, open, high, low, close, vol
    """
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    params = {'interval': interval_mexc}
    r = requests.get(url, params=params, timeout=12)
    r.raise_for_status()
    j = r.json()
    # docs: j['data'] contains arrays: time, open, close, high, low, vol, amount
    d = j.get('data') or {}
    times = d.get('time', [])
    opens = d.get('open', [])
    closes = d.get('close', [])
    highs = d.get('high', [])
    lows = d.get('low', [])
    vols = d.get('vol', [])
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(times, unit='s'),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': vols
    })
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def fetch_contract_funding_rate(symbol_mexc):
    """GET /api/v1/contract/funding_rate/{symbol}"""
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    j = r.json()
    data = j.get('data') or {}
    # sample keys: fundingRate, nextSettleTime
    return {
        'fundingRate': float(data.get('fundingRate') or 0),
        'nextSettleTime': data.get('nextSettleTime')
    }

# ----------------- Indicators & scoring (reuse your previous logic, defensive) -----------------
def compute_indicators(df):
    df = df.copy()
    # short, defensive indicator set
    try:
        df['ema20'] = ta.ema(df['close'], length=20)
        df['ema50'] = ta.ema(df['close'], length=50)
        df['ema200'] = ta.ema(df['close'], length=200)
    except Exception:
        df['ema20']=df['ema50']=df['ema200']=np.nan
    try:
        macd = ta.macd(df['close'])
        if isinstance(macd, pd.DataFrame) and macd.shape[1] >= 2:
            df['macd_hist'] = macd.iloc[:,1]
        else:
            df['macd_hist'] = np.nan
    except Exception:
        df['macd_hist'] = np.nan
    try:
        df['rsi14'] = ta.rsi(df['close'], length=14)
    except Exception:
        df['rsi14']=np.nan
    try:
        bb = ta.bbands(df['close'])
        if isinstance(bb, pd.DataFrame) and bb.shape[1] >= 3:
            df['bb_lower'] = bb.iloc[:,0]; df['bb_mid'] = bb.iloc[:,1]; df['bb_upper'] = bb.iloc[:,2]
        else:
            df['bb_lower']=df['bb_mid']=df['bb_upper']=np.nan
    except Exception:
        df['bb_lower']=df['bb_mid']=df['bb_upper']=np.nan
    try:
        df['adx14'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    except Exception:
        df['adx14'] = np.nan
    try:
        df['vol_ma_short'] = ta.sma(df['volume'], length=20)
        df['vol_ma_long'] = ta.sma(df['volume'], length=50)
        df['vol_osc'] = (df['vol_ma_short'] - df['vol_ma_long']) / (df['vol_ma_long'] + 1e-9)
    except Exception:
        df['vol_osc'] = np.nan
    # drop rows missing indicators
    df = df.dropna()
    return df

def score_signals(latest, prev, funding, weights):
    """Return total score (-100..100), per-indicator contributions, reasons list"""
    per = {}
    reasons = []
    total = 0
    # EMA alignment
    try:
        if latest['ema20'] > latest['ema50'] > latest['ema200']:
            per['ema'] = weights['ema']; total += per['ema']; reasons.append('EMA ↑')
        elif latest['ema20'] < latest['ema50'] < latest['ema200']:
            per['ema'] = -weights['ema']; total += per['ema']; reasons.append('EMA ↓')
        else:
            per['ema'] = 0
    except Exception:
        per['ema'] = 0
    # MACD hist crossover
    try:
        if prev.get('macd_hist', np.nan) < 0 and latest.get('macd_hist', np.nan) > 0:
            per['macd'] = weights['macd']; total += per['macd']; reasons.append('MACD ↑')
        elif prev.get('macd_hist', np.nan) > 0 and latest.get('macd_hist', np.nan) < 0:
            per['macd'] = -weights['macd']; total += per['macd']; reasons.append('MACD ↓')
        else:
            per['macd'] = 0
    except Exception:
        per['macd'] = 0
    # RSI
    try:
        if latest.get('rsi14', np.nan) < 30:
            per['rsi'] = int(weights['rsi'] * 0.9); total += per['rsi']; reasons.append('RSI oversold')
        elif latest.get('rsi14', np.nan) > 70:
            per['rsi'] = -int(weights['rsi']*0.9); total += per['rsi']; reasons.append('RSI overbought')
        else:
            per['rsi'] = 0
    except Exception:
        per['rsi'] = 0
    # Bollinger
    try:
        if latest.get('bb_upper', np.nan) is not np.nan and latest['close'] > latest['bb_upper']:
            per['bb'] = weights['bb']; total += per['bb']; reasons.append('BB upper')
        elif latest.get('bb_lower', np.nan) is not np.nan and latest['close'] < latest['bb_lower']:
            per['bb'] = -weights['bb']; total += per['bb']; reasons.append('BB lower')
        else:
            per['bb'] = 0
    except Exception:
        per['bb'] = 0
    # ADX
    try:
        per['adx'] = int(weights.get('adx',7) * 0.5) if latest.get('adx14',0) > 25 else 0
        total += per['adx']
    except Exception:
        per['adx'] = 0
    # Volume oscillator
    try:
        if latest.get('vol_osc',0) > 0.4:
            per['vol'] = weights.get('vol',10); total += per['vol']; reasons.append('Vol spike')
        elif latest.get('vol_osc',0) < -0.4:
            per['vol'] = -weights.get('vol',10); total += per['vol']; reasons.append('Vol drop')
        else:
            per['vol'] = 0
    except Exception:
        per['vol'] = 0
    # Funding (from MEXC funding endpoint)
    try:
        fr = funding.get('fundingRate', 0)
        if fr > 0.0006:
            per['funding'] = -weights.get('funding', 30); total += per['funding']; reasons.append('Funding + -> bearish')
        elif fr < -0.0006:
            per['funding'] = weights.get('funding', 30); total += per['funding']; reasons.append('Funding - -> bullish')
        else:
            per['funding'] = 0
    except Exception:
        per['funding'] = 0

    total = int(max(min(total, 100), -100))
    return total, per, reasons

def label_from_score(score, thresholds):
    strong_buy_t, buy_t, sell_t, strong_sell_t = thresholds
    if score is None: return "NO DATA"
    if score >= strong_buy_t: return "GÜÇLÜ AL"
    if score >= buy_t: return "AL"
    if score <= strong_sell_t: return "GÜÇLÜ SAT"
    if score <= sell_t: return "SAT"
    return "TUT"

# ----------------- SCAN (synchronous, defensive) -----------------
@st.cache_data(ttl=120)
def run_scan(symbols, timeframes, weights, thresholds, top_n=100):
    results = []
    # symbols are in user format like BTCUSDT or BTC_USDT; normalize to MEXC
    for sym in symbols[:top_n]:
        entry = {'symbol': sym, 'details': {}}
        best_score = None; best_tf = None
        buy_count = strong_buy = sell_count = 0
        mexc_sym = mexc_symbol_from(sym)
        for tf in timeframes:
            interval = INTERVAL_MAP.get(tf)
            if interval is None:
                entry['details'][tf] = None
                continue
            try:
                df = fetch_contract_klines(mexc_sym, interval)
                if df is None or df.empty or len(df) < 50:
                    entry['details'][tf] = None
                    continue
                df_ind = compute_indicators(df)
                if df_ind is None or len(df_ind) < 3:
                    entry['details'][tf] = None
                    continue
                latest = df_ind.iloc[-1]
                prev = df_ind.iloc[-2]
                funding = fetch_contract_funding_rate(mexc_sym)
                score, per_scores, reasons = score_signals(latest, prev, funding, weights)
                label = label_from_score(score, thresholds)
                entry['details'][tf] = {'score': int(score), 'label': label, 'price': float(latest['close']), 'per_scores': per_scores, 'reasons': reasons}
                if best_score is None or score > best_score:
                    best_score = score; best_tf = tf
                if label in ['AL','GÜÇLÜ AL']: buy_count += 1
                if label == 'GÜÇLÜ AL': strong_buy += 1
                if label in ['SAT','GÜÇLÜ SAT']: sell_count += 1
            except Exception:
                entry['details'][tf] = None
                continue
        entry['best_timeframe'] = best_tf
        entry['best_score'] = int(best_score) if best_score is not None else None
        entry['buy_count'] = buy_count
        entry['strong_buy_count'] = strong_buy
        entry['sell_count'] = sell_count
        results.append(entry)
    return pd.DataFrame(results)

# ----------------- UI -----------------
st.title("🔥 MEXC Vadeli (Contract) Sinyal Paneli")
st.write("Not: MEXC contract API kullanılarak piyasa verileri çekiliyor (kline, funding, ticker gibi endpointler).")

# Sidebar controls
st.sidebar.header("Tarama Ayarları")
mode = st.sidebar.selectbox("Sembol kaynağı", ["Top 50 by vol","Top 100 by vol","Custom list"])
if mode == "Custom list":
    custom = st.sidebar.text_area("Virgülle ayrılmış semboller (ör: BTCUSDT,ETHUSDT)", value="BTCUSDT,ETHUSDT")
    symbols = [s.strip().upper() for s in custom.split(',') if s.strip()]
elif mode == "Top 50 by vol":
    try:
        top = get_top_contracts_by_volume(50)
        symbols = [s.replace('_','') for s in top]  # keep user friendly BTCUSDT
    except Exception:
        symbols = ["BTCUSDT","ETHUSDT"]
else:
    try:
        top = get_top_contracts_by_volume(100)
        symbols = [s.replace('_','') for s in top]
    except Exception:
        symbols = ["BTCUSDT","ETHUSDT"]

timeframes = st.sidebar.multiselect("Zaman dilimleri", options=ALL_TFS, default=DEFAULT_TFS)
top_n = st.sidebar.slider("İlk N coin taransın", min_value=5, max_value=min(200,len(symbols)), value=min(50,len(symbols)))

with st.sidebar.expander("Ağırlıklar (gelişmiş)"):
    w_ema = st.number_input("EMA", value=25)
    w_macd = st.number_input("MACD", value=20)
    w_rsi = st.number_input("RSI", value=15)
    w_bb = st.number_input("Bollinger", value=10)
    w_adx = st.number_input("ADX", value=7)
    w_vol = st.number_input("Volume", value=10)
    w_funding = st.number_input("Funding", value=30)
weights = {'ema': w_ema, 'macd': w_macd, 'rsi': w_rsi, 'bb': w_bb, 'adx': w_adx, 'vol': w_vol, 'funding': w_funding}

with st.sidebar.expander("Sinyal eşikleri"):
    strong_buy_t = st.slider("GÜÇLÜ AL ≥", 10, 100, 60)
    buy_t = st.slider("AL ≥", 0, 80, 20)
    sell_t = st.slider("SAT ≤", -80, 0, -20)
    strong_sell_t = st.slider("GÜÇLÜ SAT ≤", -100, -10, -60)
thresholds = (strong_buy_t, buy_t, sell_t, strong_sell_t)

scan = st.sidebar.button("🔍 Tara / Yenile")

# session
if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'open_symbol' not in st.session_state: st.session_state.open_symbol = None
if scan:
    with st.spinner("MEXC contract piyasası taranıyor..."):
        st.session_state.scan_results = run_scan(symbols, timeframes, weights, thresholds, top_n=top_n)
        st.session_state.last_scan = datetime.utcnow()

df = st.session_state.scan_results
if df is None or df.empty:
    st.info("Henüz tarama yok veya sonuç boş. Yan panelden Tarayıcıyı çalıştırın.")
else:
    # small header
    st.write(f"Son tarama: {st.session_state.get('last_scan','-')}")
    # simple table view
    cols = st.columns([2,1,1,3,1,1])
    cols[0].markdown("**Coin**"); cols[1].markdown("**Best TF**"); cols[2].markdown("**Skor**")
    cols[3].markdown("**TF Etiketleri**"); cols[4].markdown("**SB**"); cols[5].markdown("**Aç**")

    for _, row in df.iterrows():
        c = st.columns([2,1,1,3,1,1])
        c[0].markdown(f"**{row['symbol']}**")
        c[1].markdown(f"{row.get('best_timeframe','-') or '-'}")
        c[2].markdown(f"<b style='font-size:18px'>{safe_int_or_dash(row.get('best_score'))}</b>", unsafe_allow_html=True)
        # tf labels compact
        tf_lines = []
        dets = row.get('details') or {}
        for tf in timeframes:
            d = dets.get(tf) if dets else None
            lbl = d.get('label') if d else "NO DATA"
            tf_lines.append(f"`{tf}`: **{lbl}**")
        c[3].write("  \n".join(tf_lines))
        c[4].markdown(f"**SB: {row.get('strong_buy_count',0)}**")
        if c[5].button("Aç", key=f"open_{row['symbol']}"):
            st.session_state.open_symbol = row['symbol']; st.session_state.open_details = dets

        if st.session_state.open_symbol == row['symbol']:
            with st.expander(f"Detaylar — {row['symbol']}", expanded=True):
                dets_local = st.session_state.open_details or {}
                for tf in timeframes:
                    cell = dets_local.get(tf) if dets_local else None
                    if not cell:
                        st.write(f"**{tf}** — Veri yok veya yetersiz.")
                        continue
                    st.markdown(f"#### {tf} — {cell.get('label','-')} (Skor: {cell.get('score','-')})")
                    st.write(f"Fiyat: {cell.get('price','-')}")
                    ps = pd.Series(cell.get('per_scores', {})).rename('points').to_frame()
                    if not ps.empty:
                        st.table(ps)
                    # direction
                    per_scores = cell.get('per_scores', {})
                    pos = sum(v for v in per_scores.values() if v>0)
                    neg = sum(-v for v in per_scores.values() if v<0)
                    net = pos - neg
                    total_abs = pos + neg if (pos+neg)>0 else 1
                    strength_pct = (net/total_abs)*100
                    direction = 'Bullish' if net>0 else ('Bearish' if net<0 else 'Neutral')
                    st.markdown(f"**Direction:** {direction} | Strength: {strength_pct:.1f}%")

st.markdown("---")
st.caption("Bilgilendirme: Bu uygulama eğitim/deneme amaçlıdır. Yatırım tavsiyesi değildir.")
