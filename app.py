import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
from binance.client import Client
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="Kripto Spot Sinyal Paneli", layout="wide", initial_sidebar_state="expanded")

# ---------------- CSS ----------------
st.markdown("""
<style>
body { background: #0b0f14; color: #d6d9de; }
.coin-row:hover { background: rgba(255,255,255,0.03); border-radius:8px; }
.score-big { font-size:20px; font-weight:800; }
</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
DEFAULT_TIMEFRAMES = ['15m', '1h', '4h']
ALL_TIMEFRAMES = ['15m','30m','1h','4h','1d']
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':10,'vol':10}

def safe_int_or_dash(val):
    if val is None or pd.isna(val): return '-'
    try: return str(int(val))
    except: return '-'

# ---------------- Binance Client ----------------
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

# ---------------- Spot Data ----------------
@st.cache_data(ttl=300)
def get_all_spot_symbols(_client):
    try:
        info = _client.get_exchange_info()
        syms = [s['symbol'] for s in info['symbols']
                if s['status']=='TRADING' and s['quoteAsset']=='USDT']
        return sorted(syms)
    except Exception:
        return ['BTCUSDT','ETHUSDT']

@st.cache_data(ttl=60)
def get_top_by_volume(_client, limit=50):
    try:
        tickers = _client.get_ticker()
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        sorted_t = sorted(usdt, key=lambda x: float(x.get('quoteVolume',0)), reverse=True)
        return [t['symbol'] for t in sorted_t[:limit]]
    except Exception:
        return get_all_spot_symbols(_client)[:limit]

@st.cache_data(ttl=60)
def fetch_klines(_client, symbol, interval, limit=500):
    try:
        kl = _client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=['timestamp','open','high','low','close','volume','close_time','qav','n_trades','taker_buy_base','taker_buy_quote','ignore'])
        df = df[['timestamp','open','high','low','close','volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()

# ---------------- Indicator Calculations ----------------
def compute_indicators(df):
    df = df.copy()
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    macd = ta.macd(df['close'])
    if isinstance(macd, pd.DataFrame): df['macd_hist'] = macd.iloc[:,1]
    df['rsi14'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'])
    if isinstance(bb, pd.DataFrame):
        df['bb_lower'] = bb.iloc[:,0]; df['bb_upper'] = bb.iloc[:,2]
    df['adx14'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    df['vol_ma_short'] = ta.sma(df['volume'], length=20)
    df['vol_ma_long'] = ta.sma(df['volume'], length=50)
    df.dropna(inplace=True)
    return df

# ---------------- Label & Scoring ----------------
def label_from_score(score, thresholds):
    strong_buy_t, buy_t, sell_t, strong_sell_t = thresholds
    if score >= strong_buy_t: return "GÜÇLÜ AL"
    if score >= buy_t: return "AL"
    if score <= strong_sell_t: return "GÜÇLÜ SAT"
    if score <= sell_t: return "SAT"
    return "TUT"

def score_signals(latest, prev, weights):
    scores, reasons = {}, []
    try:
        if latest['ema20'] > latest['ema50'] > latest['ema200']:
            scores['ema']=weights['ema']; reasons.append("EMA ↑")
        elif latest['ema20'] < latest['ema50'] < latest['ema200']:
            scores['ema']=-weights['ema']; reasons.append("EMA ↓")
        else: scores['ema']=0
    except: scores['ema']=0

    try:
        if prev['macd_hist']<0 and latest['macd_hist']>0:
            scores['macd']=weights['macd']; reasons.append("MACD ↑")
        elif prev['macd_hist']>0 and latest['macd_hist']<0:
            scores['macd']=-weights['macd']; reasons.append("MACD ↓")
        else: scores['macd']=0
    except: scores['macd']=0

    try:
        if latest['rsi14']<30: scores['rsi']=weights['rsi']
        elif latest['rsi14']>70: scores['rsi']=-weights['rsi']
        else: scores['rsi']=0
    except: scores['rsi']=0

    try:
        if latest['close']>latest['bb_upper']: scores['bb']=weights['bb']
        elif latest['close']<latest['bb_lower']: scores['bb']=-weights['bb']
        else: scores['bb']=0
    except: scores['bb']=0

    try:
        if latest['adx14']>25: scores['adx']=weights['adx']
        else: scores['adx']=0
    except: scores['adx']=0

    try:
        if latest['vol_ma_short']>latest['vol_ma_long']: scores['vol']=weights['vol']
        elif latest['vol_ma_short']<latest['vol_ma_long']: scores['vol']=-weights['vol']
        else: scores['vol']=0
    except: scores['vol']=0

    total = sum(scores.values())
    return total, scores, reasons

# ---------------- Scan ----------------
@st.cache_data(ttl=120)
def run_scan(_client, symbols, timeframes, weights, thresholds, top_n=50):
    results=[]
    for sym in symbols[:top_n]:
        entry={'symbol':sym,'details':{}}
        best_score=-999; best_tf=None
        buy_count=0; strong_buy_count=0; sell_count=0
        for tf in timeframes:
            df=fetch_klines(_client=_client, symbol=sym, interval=tf, limit=400)
            if df.empty or len(df)<50: continue
            df=compute_indicators(df)
            latest,prev=df.iloc[-1],df.iloc[-2]
            score,per_scores,reasons=score_signals(latest,prev,weights)
            label=label_from_score(score,thresholds)
            entry['details'][tf]={'score':score,'label':label,'price':float(latest['close']),'per_scores':per_scores}
            if score>best_score: best_score=score; best_tf=tf
            if label in ['AL','GÜÇLÜ AL']: buy_count+=1
            if label=='GÜÇLÜ AL': strong_buy_count+=1
            if label in ['SAT','GÜÇLÜ SAT']: sell_count+=1
        entry['best_timeframe']=best_tf
        entry['best_score']=int(best_score)
        entry['buy_count']=buy_count
        entry['strong_buy_count']=strong_buy_count
        entry['sell_count']=sell_count
        results.append(entry)
    return pd.DataFrame(results)

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Ayarlar")
api_key=st.sidebar.text_input("Binance API Key (opsiyonel)",type="password")
api_secret=st.sidebar.text_input("Binance Secret (opsiyonel)",type="password")
client=get_binance_client(api_key.strip(),api_secret.strip())
if client is None: st.stop()

col_choice=st.sidebar.radio("Coin listesi",["Top 50","Top 100","All USDT","Custom"])
if col_choice=="Custom":
    custom_input=st.sidebar.text_area("Virgülle ayrılmış coinler",value="BTCUSDT,ETHUSDT")
    symbols=[s.strip().upper() for s in custom_input.split(',') if s.strip()]
elif col_choice=="Top 50": symbols=get_top_by_volume(_client=client,limit=50)
elif col_choice=="Top 100": symbols=get_top_by_volume(_client=client,limit=100)
else: symbols=get_all_spot_symbols(_client=client)

timeframes=st.sidebar.multiselect("Zaman dilimleri",ALL_TIMEFRAMES,DEFAULT_TIMEFRAMES)
top_n=st.sidebar.slider("Kaç coin taransın",10,min(200,len(symbols)),50)
strong_buy_t=st.sidebar.slider("GÜÇLÜ AL ≥",10,100,60)
buy_t=st.sidebar.slider("AL ≥",0,80,20)
sell_t=st.sidebar.slider("SAT ≤",-80,0,-20)
strong_sell_t=st.sidebar.slider("GÜÇLÜ SAT ≤",-100,-10,-60)
thresholds=(strong_buy_t,buy_t,sell_t,strong_sell_t)
scan_button=st.sidebar.button("🔍 Tara")

# ---------------- Main ----------------
st.title("📊 Binance Spot Kripto Sinyal Paneli")
st.caption("SB = Strong Buy sayısı (kaç zaman diliminde 'GÜÇLÜ AL' olduğu).")

if 'scan_results' not in st.session_state: st.session_state.scan_results=pd.DataFrame()
if 'open_symbol' not in st.session_state: st.session_state.open_symbol=None

if scan_button:
    with st.spinner("Taranıyor..."):
        st.session_state.scan_results=run_scan(_client=client, symbols=symbols, timeframes=timeframes, weights=DEFAULT_WEIGHTS, thresholds=thresholds, top_n=top_n)

dfres=st.session_state.scan_results
if dfres.empty:
    st.info("Henüz tarama yapılmadı.")
else:
    hdr=st.columns([2,1,1,3,1,1])
    hdr[0].markdown("**Coin**"); hdr[1].markdown("**TF**"); hdr[2].markdown("**Skor**")
    hdr[3].markdown("**Zaman Dilimleri**"); hdr[4].markdown("**SB**"); hdr[5].markdown("**Detay**")
    for i,row in dfres.iterrows():
        cols=st.columns([2,1,1,3,1,1])
        cols[0].markdown(f"**{row['symbol']}**")
        cols[1].markdown(f"{row['best_timeframe']}")
        cols[2].markdown(f"<div class='score-big'>{safe_int_or_dash(row['best_score'])}</div>",unsafe_allow_html=True)
        tf_labels=[]
        for tf,det in row['details'].items():
            tf_labels.append(f"`{tf}`: **{det['label']}**")
        cols[3].write("  \n".join(tf_labels))
        cols[4].markdown(f"**SB: {row['strong_buy_count']}**")
        btn=cols[5].button("Aç",key=f"open_{row['symbol']}")
        if btn:
            st.session_state.open_symbol=row['symbol']
            st.session_state.open_details=row['details']
        if st.session_state.open_symbol==row['symbol']:
            with st.expander(f"Detaylar — {row['symbol']}",expanded=True):
                details=st.session_state.open_details
                for tf,cell in details.items():
                    st.markdown(f"#### {tf} — {cell['label']} (Skor: {cell['score']})")
                    ps=pd.Series(cell['per_scores']).rename('points').to_frame()
                    st.table(ps)
                sel_tf=row['best_timeframe'] or '1h'
                interval_map={'15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
                tv_html=f"""
                <div class="tradingview-widget-container" style="height:480px; width:100%">
                  <div id="tv_{row['symbol']}"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                  <script type="text/javascript">
                  new TradingView.widget({{
                    "container_id": "tv_{row['symbol']}",
                    "symbol": "BINANCE:{row['symbol']}",
                    "interval": "{interval_map.get(sel_tf,'60')}",
                    "timezone": "Europe/Istanbul",
                    "theme": "dark",
                    "style": "1",
                    "locale": "tr"
                  }});
                  </script>
                </div>
                """
                st.components.v1.html(tv_html,height=480)

st.markdown("---")
st.caption("Bu uygulama yatırım tavsiyesi değildir.")
