# app.py - Düzeltilmiş ve NaN-safe sürüm
import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Pro Vadeli Sinyal Paneli", layout="wide", initial_sidebar_state="expanded")

# ---------------- CSS
st.markdown("""
<style>
body { background: #0b0f14; color: #d6d9de; }
.panel-card { background: #101820; padding:12px; border-radius:12px; border:1px solid rgba(255,255,255,0.05); }
.coin-row:hover { background: rgba(255,255,255,0.03); border-radius:8px; }
.small-muted { color:#9aa3b2; font-size:12px; }
.score-big { font-size:20px; font-weight:800; }
.badge { padding:6px 8px; border-radius:8px; font-weight:700; }
.green { background:#064e3b; color:#66ffb2; }
.red { background:#5b1220; color:#ff9b9b; }
.yellow { background:#5a4b0c; color:#ffe48d; }
</style>
""", unsafe_allow_html=True)

# ---------------- Config
DEFAULT_TIMEFRAMES = ['15m', '1h', '4h']
ALL_TIMEFRAMES = ['15m','30m','1h','4h','1d']
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':7,'vol':10,'funding':30}

def safe_int_or_dash(val):
    """Return int string if val is finite number, else '-'"""
    if val is None:
        return '-'
    try:
        if pd.isna(val):
            return '-'
    except Exception:
        pass
    try:
        return str(int(val))
    except Exception:
        return '-'

def get_coin_logo(symbol):
    base = symbol.replace('USDT','').lower()
    try:
        url = f"https://assets.coincap.io/assets/icons/{base}@2x.png"
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return url
    except Exception:
        pass
    return None

@st.cache_resource
def get_binance_client(api_key, api_secret):
    try:
        if api_key and api_secret:
            c = Client(api_key, api_secret)
        else:
            c = Client()
        c.ping()
        return c
    except Exception as e:
        st.error(f"Binance bağlantısında hata: {e}")
        return None

# ---------------- Cache-uyumlu fetcher'lar (parametre isimleri _client olacak)
@st.cache_data(ttl=300)
def get_all_futures_symbols(_client):
    try:
        info = _client.futures_exchange_info()
        syms = [s['symbol'] for s in info['symbols']
                if s['status']=='TRADING' and s['quoteAsset']=='USDT' and s['contractType']=='PERPETUAL']
        return sorted(syms)
    except Exception:
        return ['BTCUSDT','ETHUSDT']

@st.cache_data(ttl=60)
def get_top_by_volume(_client, limit=100):
    try:
        tickers = _client.futures_ticker()
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        sorted_t = sorted(usdt, key=lambda x: float(x.get('quoteVolume',0)), reverse=True)
        return [t['symbol'] for t in sorted_t[:limit]]
    except Exception:
        return get_all_futures_symbols(_client)[:limit]

@st.cache_data(ttl=60)
def fetch_klines(_client, symbol, interval, limit=500):
    try:
        kl = _client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=['timestamp','open','high','low','close','volume','close_time','qav','n_trades','taker_buy_base','taker_buy_quote','ignore'])
        df = df[['timestamp','open','high','low','close','volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_sentiment(_client, symbol):
    data = {}
    try:
        fr = _client.futures_funding_rate(symbol=symbol, limit=1)[0]
        data['fundingRate'] = float(fr.get('fundingRate',0))
    except Exception:
        data['fundingRate'] = 0.0
    try:
        oi = _client.futures_open_interest(symbol=symbol)
        data['openInterest'] = float(oi.get('openInterest',0))
    except Exception:
        data['openInterest'] = 0.0
    try:
        ls = _client.futures_top_long_short_account_ratio(symbol=symbol, period='5m', limit=1)[0]
        data['ls_long'] = float(ls.get('longAccount',0.5))
        data['ls_short'] = float(ls.get('shortAccount',0.5))
    except Exception:
        data['ls_long'] = 0.5
        data['ls_short'] = 0.5
    return data

# ---------------- Indicators (defensive)
def compute_indicators(df):
    df = df.copy()
    # compute indicators defensively
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
        df['rsi14'] = np.nan
    try:
        bb = ta.bbands(df['close'])
        if isinstance(bb, pd.DataFrame) and bb.shape[1] >= 3:
            df['bb_lower'] = bb.iloc[:,0]; df['bb_mid'] = bb.iloc[:,1]; df['bb_upper'] = bb.iloc[:,2]
        else:
            df['bb_lower']=df['bb_mid']=df['bb_upper']=np.nan
    except Exception:
        df['bb_lower']=df['bb_mid']=df['bb_upper']=np.nan
    try:
        adx = ta.adx(df['high'], df['low'], df['close'])
        df['adx14'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx.columns else np.nan
    except Exception:
        df['adx14'] = np.nan
    try:
        df['mfi14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
    except Exception:
        df['mfi14'] = np.nan
    try:
        df['vol_ma_short'] = ta.sma(df['volume'], length=20)
        df['vol_ma_long'] = ta.sma(df['volume'], length=50)
        df['vol_osc'] = (df['vol_ma_short'] - df['vol_ma_long']) / (df['vol_ma_long'] + 1e-9)
    except Exception:
        df['vol_osc'] = np.nan
    # dropna to keep rows where indicators exist
    df = df.dropna()
    return df

def label_from_score(score, thresholds):
    strong_buy_t, buy_t, sell_t, strong_sell_t = thresholds
    if score is None:
        return "NO DATA"
    if score >= strong_buy_t: return "GÜÇLÜ AL"
    if score >= buy_t: return "AL"
    if score <= strong_sell_t: return "GÜÇLÜ SAT"
    if score <= sell_t: return "SAT"
    return "TUT"

def score_signals(latest, prev, sentiment, weights):
    scores = {}
    reasons = []
    total = 0
    # EMA
    try:
        if latest['ema20'] > latest['ema50'] > latest['ema200']:
            scores['ema'] = weights['ema']; reasons.append('EMA ↑'); total += scores['ema']
        elif latest['ema20'] < latest['ema50'] < latest['ema200']:
            scores['ema'] = -weights['ema']; reasons.append('EMA ↓'); total += scores['ema']
        else:
            scores['ema'] = 0
    except Exception:
        scores['ema'] = 0
    # MACD hist
    try:
        if prev.get('macd_hist', np.nan) < 0 and latest.get('macd_hist', np.nan) > 0:
            scores['macd'] = weights['macd']; reasons.append('MACD ↑'); total += scores['macd']
        elif prev.get('macd_hist', np.nan) > 0 and latest.get('macd_hist', np.nan) < 0:
            scores['macd'] = -weights['macd']; reasons.append('MACD ↓'); total += scores['macd']
        else:
            scores['macd'] = 0
    except Exception:
        scores['macd'] = 0
    # RSI
    try:
        if latest.get('rsi14', np.nan) < 30:
            scores['rsi'] = weights['rsi']; reasons.append('RSI oversold'); total += scores['rsi']
        elif latest.get('rsi14', np.nan) > 70:
            scores['rsi'] = -weights['rsi']; reasons.append('RSI overbought'); total += scores['rsi']
        else:
            scores['rsi'] = 0
    except Exception:
        scores['rsi'] = 0
    # Bollinger
    try:
        if not pd.isna(latest.get('bb_upper')) and latest['close'] > latest['bb_upper']:
            scores['bb'] = weights['bb']; reasons.append('BB upper'); total += scores['bb']
        elif not pd.isna(latest.get('bb_lower')) and latest['close'] < latest['bb_lower']:
            scores['bb'] = -weights['bb']; reasons.append('BB lower'); total += scores['bb']
        else:
            scores['bb'] = 0
    except Exception:
        scores['bb'] = 0
    # ADX
    try:
        if latest.get('adx14', 0) > 25:
            scores['adx'] = int(weights['adx'] * 0.5); total += scores['adx']
        else:
            scores['adx'] = 0
    except Exception:
        scores['adx'] = 0
    # Volume oscillator
    try:
        if latest.get('vol_osc', 0) > 0.4:
            scores['vol'] = weights['vol']; total += scores['vol']
        elif latest.get('vol_osc', 0) < -0.4:
            scores['vol'] = -weights['vol']; total += scores['vol']
        else:
            scores['vol'] = 0
    except Exception:
        scores['vol'] = 0
    # MFI
    try:
        if latest.get('mfi14', np.nan) < 20:
            scores['mfi'] = int(weights['rsi']*0.5); total += scores['mfi']
        elif latest.get('mfi14', np.nan) > 80:
            scores['mfi'] = -int(weights['rsi']*0.5); total += scores['mfi']
        else:
            scores['mfi'] = 0
    except Exception:
        scores['mfi'] = 0
    # Sentiment (funding)
    try:
        fr = sentiment.get('fundingRate', 0)
        if fr > 0.0006:
            scores['funding'] = -weights['funding']; total += scores['funding']
        elif fr < -0.0006:
            scores['funding'] = weights['funding']; total += scores['funding']
        else:
            scores['funding'] = 0
    except Exception:
        scores['funding'] = 0

    total = int(max(min(total, 100), -100))
    return total, scores, reasons

# ---------------- Scan engine (defensive checks)
@st.cache_data(ttl=120)
def run_scan(_client, symbols, timeframes, weights, thresholds, top_n=100):
    results = []
    for sym in symbols[:top_n]:
        entry = {'symbol': sym, 'details': {}}
        best_score = None
        best_tf = None
        buy_count = 0; strong_buy_count = 0; sell_count = 0

        for tf in timeframes:
            try:
                df = fetch_klines(_client=_client, symbol=sym, interval=tf, limit=400)
                if df is None or df.empty or len(df) < 30:
                    entry['details'][tf] = None
                    continue
                df_ind = compute_indicators(df)
                if df_ind is None or len(df_ind) < 3:
                    entry['details'][tf] = None
                    continue
                latest = df_ind.iloc[-1]
                prev = df_ind.iloc[-2]
                sentiment = fetch_sentiment(_client=_client, symbol=sym)
                score, per_scores, reasons = score_signals(latest, prev, sentiment, weights)
                label = label_from_score(score, thresholds)
                entry['details'][tf] = {'score': int(score), 'label': label, 'price': float(latest['close']), 'per_scores': per_scores, 'reasons': reasons}
                if best_score is None or score > best_score:
                    best_score = score; best_tf = tf
                if label in ['AL','GÜÇLÜ AL']: buy_count += 1
                if label == 'GÜÇLÜ AL': strong_buy_count += 1
                if label in ['SAT','GÜÇLÜ SAT']: sell_count += 1
            except Exception:
                entry['details'][tf] = None
                continue

        entry['best_timeframe'] = best_tf
        entry['best_score'] = int(best_score) if best_score is not None else None
        entry['buy_count'] = buy_count
        entry['strong_buy_count'] = strong_buy_count
        entry['sell_count'] = sell_count
        results.append(entry)
    return pd.DataFrame(results)

# ---------------- Sidebar / inputs
st.sidebar.title("⚙️ Ayarlar")
api_key = st.sidebar.text_input("Binance API Key (opsiyonel)", type="password")
api_secret = st.sidebar.text_input("Binance Secret (opsiyonel)", type="password")
client = get_binance_client(api_key.strip(), api_secret.strip())
if client is None:
    st.stop()

col_choice = st.sidebar.radio("Coin listesi", ["Top 50","Top 100","All USDT Perp","Custom list"])
if col_choice == "Custom list":
    custom_input = st.sidebar.text_area("Virgülle ayrılmış coinler", value="BTCUSDT,ETHUSDT")
    symbols = [s.strip().upper() for s in custom_input.split(',') if s.strip()]
elif col_choice == "Top 50":
    symbols = get_top_by_volume(_client=client, limit=50)
elif col_choice == "Top 100":
    symbols = get_top_by_volume(_client=client, limit=100)
else:
    symbols = get_all_futures_symbols(_client=client)

timeframes = st.sidebar.multiselect("Zaman dilimleri", ALL_TIMEFRAMES, DEFAULT_TIMEFRAMES)

max_possible = min(500, max(10, len(symbols)))
top_n = st.sidebar.slider("Kaç coin taransın (ilk N)", min_value=10, max_value=max_possible, value=min(100, max_possible))

with st.sidebar.expander("Gösterge ağırlıkları"):
    w_ema = st.number_input("EMA", value=DEFAULT_WEIGHTS['ema'], step=1)
    w_macd = st.number_input("MACD", value=DEFAULT_WEIGHTS['macd'], step=1)
    w_rsi = st.number_input("RSI", value=DEFAULT_WEIGHTS['rsi'], step=1)
    w_bb = st.number_input("Bollinger", value=DEFAULT_WEIGHTS['bb'], step=1)
    w_adx = st.number_input("ADX", value=DEFAULT_WEIGHTS['adx'], step=1)
    w_vol = st.number_input("Volume", value=DEFAULT_WEIGHTS['vol'], step=1)
    w_funding = st.number_input("Funding/OI", value=DEFAULT_WEIGHTS['funding'], step=1)
weights = {'ema': w_ema, 'macd': w_macd, 'rsi': w_rsi, 'bb': w_bb, 'adx': w_adx, 'vol': w_vol, 'funding': w_funding}

with st.sidebar.expander("Sinyal eşikleri"):
    strong_buy_t = st.slider("GÜÇLÜ AL ≥", 10, 100, 60)
    buy_t = st.slider("AL ≥", 0, 80, 20)
    sell_t = st.slider("SAT ≤", -80, 0, -20)
    strong_sell_t = st.slider("GÜÇLÜ SAT ≤", -100, -10, -60)
thresholds = (strong_buy_t, buy_t, sell_t, strong_sell_t)

scan_button = st.sidebar.button("🔍 Tara / Yenile")

# ---------------- Main UI
st.title("📊 Pro Vadeli Sinyal Paneli")
st.caption("SB = Strong Buy sayısı (kaç TF'de 'GÜÇLÜ AL' olduğu).")

if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'open_symbol' not in st.session_state: st.session_state.open_symbol = None
if 'open_details' not in st.session_state: st.session_state.open_details = None

if scan_button:
    with st.spinner("Piyasa taranıyor — lütfen bekleyin..."):
        st.session_state.scan_results = run_scan(_client=client, symbols=symbols, timeframes=timeframes, weights=weights, thresholds=thresholds, top_n=top_n)
        st.session_state.last_scan = datetime.utcnow()

dfres = st.session_state.scan_results
if dfres is None or dfres.empty:
    st.info("Henüz tarama yapılmadı (veya sonuç boş). Yan panelden parametreleri seçip 'Tara / Yenile' butonuna basın.")
else:
    # header
    hdr = st.columns([2,1,1,3,1,1])
    hdr[0].markdown("**Coin**"); hdr[1].markdown("**TF**"); hdr[2].markdown("**Skor**")
    hdr[3].markdown("**Zaman Dilimleri**"); hdr[4].markdown("**SB**"); hdr[5].markdown("**Detay**")

    sort_by = st.selectbox("Sırala", options=["Best Score","Strong Buy Count","Buy Count","Sell Count","Symbol"], index=0)
    desc = st.checkbox("Azalan sırada", value=True)
    if sort_by == "Best Score":
        dfres = dfres.sort_values(by='best_score', ascending=not desc, na_position='last')
    elif sort_by == "Strong Buy Count":
        dfres = dfres.sort_values(by='strong_buy_count', ascending=not desc)
    elif sort_by == "Buy Count":
        dfres = dfres.sort_values(by='buy_count', ascending=not desc)
    elif sort_by == "Sell Count":
        dfres = dfres.sort_values(by='sell_count', ascending=not desc)
    else:
        dfres = dfres.sort_values(by='symbol', ascending=not desc)

    max_show = st.number_input("Bir sayfada göster", min_value=10, max_value=min(500, len(dfres)), value=min(100, len(dfres)))
    shown = dfres.head(int(max_show))

    for idx, row in shown.iterrows():
        cols = st.columns([2,1,1,3,1,1])
        # logo
        logo = get_coin_logo(row['symbol'])
        if logo:
            try:
                r = requests.get(logo, timeout=2)
                if r.status_code == 200:
                    img = Image.open(BytesIO(r.content)); cols[0].image(img, width=32)
            except Exception:
                pass
        cols[0].markdown(f"**{row['symbol']}**")
        cols[1].markdown(f"**{row.get('best_timeframe','-') or '-'}**")
        sc_val = row.get('best_score')
        cols[2].markdown(f"<div class='score-big'>{safe_int_or_dash(sc_val)}</div>", unsafe_allow_html=True)
        # per-TF labels
        tf_lines = []
        details = row.get('details') or {}
        for tf in timeframes:
            d = details.get(tf) if details else None
            lbl = d.get('label') if d else "NO DATA"
            tf_lines.append(f"`{tf}`: **{lbl}**")
        cols[3].write("  \n".join(tf_lines))
        cols[4].markdown(f"**SB: {safe_int_or_dash(row.get('strong_buy_count',0))}**")
        btn = cols[5].button("Aç", key=f"open_{row['symbol']}")
        if btn:
            st.session_state.open_symbol = row['symbol']
            st.session_state.open_details = details

        # inline detail (only for selected)
        if st.session_state.open_symbol == row['symbol']:
            with st.container():
                with st.expander(f"Detaylar — {row['symbol']}", expanded=True):
                    details_local = st.session_state.open_details or details or {}
                    for tf in timeframes:
                        cell = details_local.get(tf) if details_local else None
                        if not cell:
                            st.write(f"**{tf}** — Veri yok veya yetersiz.")
                            continue
                        st.markdown(f"#### {tf} — {cell.get('label','-')} (Skor: {safe_int_or_dash(cell.get('score'))})")
                        st.write(f"Fiyat: {cell.get('price','-')}")
                        ps = pd.Series(cell.get('per_scores', {})).rename('points').to_frame()
                        if not ps.empty:
                            st.table(ps)
                        else:
                            st.write("Gösterge puanları yok.")
                        per_scores = cell.get('per_scores', {})
                        pos_sum = sum(v for v in per_scores.values() if v>0)
                        neg_sum = sum(-v for v in per_scores.values() if v<0)
                        net = pos_sum - neg_sum
                        total_abs = pos_sum + neg_sum if (pos_sum+neg_sum)>0 else 1
                        strength_pct = (net / total_abs) * 100
                        direction = 'Bullish' if net>0 else ('Bearish' if net<0 else 'Neutral')
                        st.markdown(f"**Direction:** **{direction}**  |  Strength: {strength_pct:.1f}%")
                    sel_tf = row.get('best_timeframe') or (timeframes[0] if timeframes else '1h')
                    interval_map = {'15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
                    tv_interval = interval_map.get(sel_tf, '60')
                    tv_html = f"""
                    <div class="tradingview-widget-container" style="height:480px; width:100%">
                      <div id="tv_{row['symbol']}"></div>
                      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                      <script type="text/javascript">
                      new TradingView.widget({{
                        "container_id": "tv_{row['symbol']}",
                        "symbol": "BINANCE:{row['symbol']}",
                        "interval": "{tv_interval}",
                        "timezone": "Europe/Istanbul",
                        "theme": "dark",
                        "style": "1",
                        "locale": "tr",
                        "toolbar_bg": "#0b0f14",
                        "enable_publishing": false,
                        "hide_legend": true
                      }});
                      </script>
                    </div>
                    """
                    st.components.v1.html(tv_html, height=480)

st.markdown("---")
st.caption("Bu uygulama yatırım tavsiyesi değildir. Lütfen kendi risk yönetiminizi uygulayın.")
