import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
from PIL import Image
from io import BytesIO
import time

st.set_page_config(page_title="MEXC Futures Sinyal Paneli", layout="wide", initial_sidebar_state="expanded")

# ---------------- STYLE
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

# ---------------- DEFAULTS
DEFAULT_TIMEFRAMES = ['15m','1h','4h']
ALL_TIMEFRAMES = ['5m','15m','30m','1h','4h','1d']
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':7,'vol':10,'funding':30}

# ---------------- HELPERS
def get_coin_logo(symbol):
    base = symbol.replace('USDT','').lower()
    try:
        url = f"https://assets.coincap.io/assets/icons/{base}@2x.png"
        r = requests.get(url, timeout=3)
        if r.status_code == 200:
            return url
    except: pass
    return None

def safe_int_or_dash(val):
    if val is None or pd.isna(val): return '-'
    try: return str(int(val))
    except: return '-'

# ---------------- MEXC Futures API
MEXC_BASE = "https://fapi.mexc.com"

def get_futures_symbols():
    try:
        r = requests.get(f"{MEXC_BASE}/api/v1/contract/detail", timeout=5)
        data = r.json()
        return sorted([i['symbol'] for i in data if i.get('quoteCoin') == 'USDT'])
    except:
        return ['BTC_USDT','ETH_USDT']

def fetch_klines(symbol, interval, limit=500):
    try:
        r = requests.get(f"{MEXC_BASE}/api/v1/contract/kline/{symbol}?interval={interval}&limit={limit}", timeout=5)
        data = r.json().get('data', [])
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','closeTime'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()

# ---------------- COINGLASS API (Funding Rate, Long/Short, OI)
def fetch_sentiment(symbol):
    try:
        coin = symbol.replace("_USDT", "").upper()
        r = requests.get(f"https://open-api.coinglass.com/api/futures/fundingRate?symbol={coin}&exchange=MEXC",
                         headers={'coinglassSecret': 'demo-key'}, timeout=5)
        data = r.json().get('data', [])
        funding = float(data[0]['fundingRate']) if data else 0.0
        return {'fundingRate': funding, 'openInterest': 0.0}
    except:
        return {'fundingRate': 0.0, 'openInterest': 0.0}

# ---------------- INDICATORS
def compute_indicators(df):
    df = df.copy()
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    macd = ta.macd(df['close'])
    df['macd_hist'] = macd.iloc[:,1] if isinstance(macd, pd.DataFrame) else np.nan
    df['rsi14'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'])
    if isinstance(bb, pd.DataFrame):
        df['bb_lower'] = bb.iloc[:,0]; df['bb_upper'] = bb.iloc[:,2]
    df['adx14'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14']
    df.dropna(inplace=True)
    return df

def label_from_score(score, t):
    if score >= t[0]: return "GÜÇLÜ AL"
    if score >= t[1]: return "AL"
    if score <= t[3]: return "GÜÇLÜ SAT"
    if score <= t[2]: return "SAT"
    return "TUT"

def score_signals(latest, prev, sentiment, w):
    s = {}; total = 0
    if latest['ema20']>latest['ema50']>latest['ema200']: s['ema']=w['ema']
    elif latest['ema20']<latest['ema50']<latest['ema200']: s['ema']=-w['ema']
    else: s['ema']=0
    if prev['macd_hist']<0 and latest['macd_hist']>0: s['macd']=w['macd']
    elif prev['macd_hist']>0 and latest['macd_hist']<0: s['macd']=-w['macd']
    else: s['macd']=0
    if latest['rsi14']<30: s['rsi']=w['rsi']
    elif latest['rsi14']>70: s['rsi']=-w['rsi']
    else: s['rsi']=0
    if latest['close']>latest['bb_upper']: s['bb']=w['bb']
    elif latest['close']<latest['bb_lower']: s['bb']=-w['bb']
    else: s['bb']=0
    s['adx'] = w['adx'] if latest['adx14']>25 else 0
    fr = sentiment.get('fundingRate',0)
    s['funding'] = w['funding'] if fr<0 else -w['funding'] if fr>0 else 0
    total = sum(s.values())
    return int(total), s

# ---------------- SCANNER
def run_scan(symbols, timeframes, weights, thresholds, top_n=50):
    results=[]
    for sym in symbols[:top_n]:
        entry={'symbol':sym,'details':{}}
        best_score=-999; best_tf=None; sb=0
        for tf in timeframes:
            df=fetch_klines(sym, tf, 400)
            if df.empty or len(df)<50: continue
            df=compute_indicators(df)
            if len(df)<5: continue
            latest,prev=df.iloc[-1],df.iloc[-2]
            sent=fetch_sentiment(sym)
            score,subs=score_signals(latest,prev,sent,weights)
            label=label_from_score(score,thresholds)
            if label=="GÜÇLÜ AL": sb+=1
            if score>best_score: best_score=score; best_tf=tf
            entry['details'][tf]={'score':score,'label':label,'per_scores':subs,'price':float(latest['close'])}
        entry['best_timeframe']=best_tf
        entry['best_score']=best_score
        entry['strong_buy_count']=sb
        results.append(entry)
    return pd.DataFrame(results)

# ---------------- SIDEBAR
st.sidebar.title("⚙️ Ayarlar")
coin_option=st.sidebar.radio("Coin listesi",["Top 50","All","Custom"])
if coin_option=="Custom":
    custom_input=st.sidebar.text_area("Coinler (BTC_USDT, ETH_USDT)", value="BTC_USDT,ETH_USDT")
    symbols=[s.strip().upper() for s in custom_input.split(',')]
else:
    symbols=get_futures_symbols()

timeframes=st.sidebar.multiselect("Zaman Dilimleri",ALL_TIMEFRAMES,DEFAULT_TIMEFRAMES)
top_n=st.sidebar.slider("Kaç coin taransın",10,min(200,len(symbols)),50)
strong_buy_t=st.sidebar.slider("GÜÇLÜ AL ≥",10,100,60)
buy_t=st.sidebar.slider("AL ≥",0,80,20)
sell_t=st.sidebar.slider("SAT ≤",-80,0,-20)
strong_sell_t=st.sidebar.slider("GÜÇLÜ SAT ≤",-100,-10,-60)
thresholds=(strong_buy_t,buy_t,sell_t,strong_sell_t)
scan_button=st.sidebar.button("🔍 Tara")

# ---------------- MAIN UI
st.title("📊 MEXC Futures Kripto Sinyal Paneli")
st.caption("Veriler MEXC ve Coinglass API'lerinden alınır. Türkiye’den tamamen erişilebilir.")

if 'results' not in st.session_state: st.session_state.results=pd.DataFrame()
if 'open_symbol' not in st.session_state: st.session_state.open_symbol=None

if scan_button:
    with st.spinner("Taranıyor..."):
        st.session_state.results=run_scan(symbols,timeframes,DEFAULT_WEIGHTS,thresholds,top_n)

dfres=st.session_state.results
if dfres.empty:
    st.info("Henüz tarama yapılmadı.")
else:
    hdr=st.columns([2,1,1,3,1])
    hdr[0].markdown("**Coin**"); hdr[1].markdown("**TF**"); hdr[2].markdown("**Skor**")
    hdr[3].markdown("**Detay**"); hdr[4].markdown("**SB**")

    for i,row in dfres.iterrows():
        cols=st.columns([2,1,1,3,1])
        logo=get_coin_logo(row['symbol'])
        if logo:
            try:
                r=requests.get(logo,timeout=2)
                if r.status_code==200:
                    img=Image.open(BytesIO(r.content)); cols[0].image(img,width=32)
            except: pass
        cols[0].markdown(f"**{row['symbol']}**")
        cols[1].markdown(f"{row['best_timeframe']}")
        cols[2].markdown(f"<div class='score-big'>{safe_int_or_dash(row['best_score'])}</div>", unsafe_allow_html=True)
        btn=cols[3].button("Aç", key=f"open_{row['symbol']}")
        cols[4].markdown(f"**{safe_int_or_dash(row['strong_buy_count'])}**")

        if btn:
            st.session_state.open_symbol=row['symbol']

        if st.session_state.open_symbol==row['symbol']:
            with st.expander(f"Detay — {row['symbol']}", expanded=True):
                details=row['details']
                for tf,data in details.items():
                    st.markdown(f"#### {tf}: {data['label']} (Skor: {data['score']})")
                    st.write(f"Fiyat: {data['price']}")
                    st.table(pd.Series(data['per_scores']).to_frame("puan"))
st.markdown("---")
st.caption("Bu panel yatırım tavsiyesi değildir.")
