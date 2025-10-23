# app.py
# MEXC Contract scanner — 4-column professional UI with TradingView widget
import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
from io import BytesIO
from PIL import Image

# Try optional plotly for indicator bars (fallback to table)
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

st.set_page_config(page_title="MEXC Vadeli — Profesyonel Panel", layout="wide", initial_sidebar_state="expanded")

# ---------------- CONFIG ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {
    '1m': 'Min1', '5m': 'Min5', '15m': 'Min15', '30m': 'Min30',
    '1h': 'Min60', '4h': 'Hour4', '8h': 'Hour8', '1d': 'Day1'
}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','8h':'480','1d':'D'}

DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','8h','1d']

# ---------------- CSS (daha profesyonel tema)
st.markdown("""
<style>
body { background: #0b0f14; color: #e6eef6; }
.block { background: linear-gradient(180deg,#0c1116,#071018); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.04); }
.coin-item { padding:8px; border-radius:8px; }
.coin-item:hover { background: rgba(255,255,255,0.02); }
.badge { padding:6px 8px; border-radius:8px; font-weight:700; }
.badge-green { background:#054E2B; color:#BFFFE0; }
.badge-red { background:#4A0B0B; color:#FFCBCB; }
.small-muted { color:#9aa3b2; font-size:12px; }
.score-big { font-size:20px; font-weight:800; }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers & MEXC endpoints (same as before) ----------------
def mexc_symbol_from(symbol: str) -> str:
    s = symbol.strip().upper()
    if '_' in s:
        return s
    if s.endswith('USDT'):
        return s[:-4] + "_USDT"
    return s[:-4] + "_" + s[-4:]

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

def get_top_contracts_by_volume(limit=100):
    data = fetch_contract_ticker()
    def vol(x):
        # support different naming
        return float(x.get('volume24') or x.get('amount24') or 0)
    items = sorted(data, key=vol, reverse=True)
    return [it.get('symbol') for it in items[:limit]]

def fetch_contract_klines(symbol_mexc, interval_mexc):
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    params = {'interval': interval_mexc}
    try:
        j = fetch_json(url, params=params)
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

# ---------------- Indicators & Scoring helpers (kept from previous improved logic) ----------------
def nw_smooth(series, bandwidth=8):
    y = np.asarray(series)
    n = len(y)
    if n == 0:
        return np.array([])
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
        df['ema20']=df['ema50']=df['ema200']=np.nan
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
            df['bb_lower']=df['bb_mid']=df['bb_upper']=np.nan
    except Exception:
        df['bb_lower']=df['bb_mid']=df['bb_upper']=np.nan
    try:
        adx = ta.adx(df['high'], df['low'], df['close'])
        df['adx14'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx.columns else np.nan
    except Exception:
        df['adx14'] = np.nan
    try:
        df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    except Exception:
        df['atr14']=np.nan
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

def normalize_by_volatility(value, latest_close, atr):
    try:
        vol_ratio = (atr / (latest_close + 1e-9))
        factor = 1.0 / (1.0 + vol_ratio * 5.0)
        return value * factor
    except Exception:
        return value

def score_signals(latest, prev, funding, weights):
    per = {}
    reasons = []
    total = 0
    atr = float(latest.get('atr14', np.nan) if not pd.isna(latest.get('atr14', np.nan)) else 0.0)
    price = float(latest.get('close', np.nan) if not pd.isna(latest.get('close', np.nan)) else 0.0)

    # EMA
    try:
        w = weights.get('ema', 20)
        contrib = 0
        if latest['ema20'] > latest['ema50'] > latest['ema200']:
            contrib = +w; reasons.append("EMA alignment bullish")
        elif latest['ema20'] < latest['ema50'] < latest['ema200']:
            contrib = -w; reasons.append("EMA alignment bearish")
        per['ema'] = normalize_by_volatility(contrib, price, atr); total += per['ema']
    except Exception:
        per['ema'] = 0

    # MACD
    try:
        w = weights.get('macd', 15)
        p_h = float(prev.get('macd_hist', 0)); l_h = float(latest.get('macd_hist', 0))
        contrib = 0
        if p_h < 0 and l_h > 0:
            scale = min(abs(l_h) / (abs(p_h)+1e-9), 3.0); contrib = w * scale; reasons.append("MACD crossover bullish")
        elif p_h > 0 and l_h < 0:
            scale = min(abs(l_h) / (abs(p_h)+1e-9), 3.0); contrib = -w * scale; reasons.append("MACD crossover bearish")
        per['macd'] = normalize_by_volatility(contrib, price, atr); total += per['macd']
    except Exception:
        per['macd'] = 0

    # RSI
    try:
        w = weights.get('rsi', 12); rsi = float(latest.get('rsi14', np.nan))
        if rsi < 25: contrib = w * (1 + (30 - rsi)/10); reasons.append(f"RSI very low {rsi:.1f}")
        elif rsi < 35: contrib = w * 0.6; reasons.append(f"RSI low {rsi:.1f}")
        elif rsi > 75: contrib = -w * (1 + (rsi - 70)/10); reasons.append(f"RSI very high {rsi:.1f}")
        elif rsi > 65: contrib = -w * 0.6; reasons.append(f"RSI high {rsi:.1f}")
        else: contrib = 0
        per['rsi'] = normalize_by_volatility(contrib, price, atr); total += per['rsi']
    except Exception:
        per['rsi'] = 0

    # Bollinger
    try:
        w = weights.get('bb', 8)
        if latest['close'] > latest['bb_upper']: contrib = w; reasons.append("Price above BB upper")
        elif latest['close'] < latest['bb_lower']: contrib = -w; reasons.append("Price below BB lower")
        else: contrib = 0
        per['bb'] = normalize_by_volatility(contrib, price, atr); total += per['bb']
    except Exception:
        per['bb'] = 0

    # ADX
    try:
        w = weights.get('adx', 6); adx = float(latest.get('adx14', 0) if not pd.isna(latest.get('adx14', np.nan)) else 0)
        if adx > 35: contrib = w; reasons.append("ADX strong trend")
        elif adx > 25: contrib = int(w*0.6)
        else: contrib = 0
        per['adx'] = contrib; total += per['adx']
    except Exception:
        per['adx'] = 0

    # Volume
    try:
        w = weights.get('vol', 6); vol_osc = float(latest.get('vol_osc', 0))
        if vol_osc > 0.5: contrib = w; reasons.append("Volume spike")
        elif vol_osc < -0.5: contrib = -w; reasons.append("Volume drop")
        else: contrib = 0
        per['vol'] = normalize_by_volatility(contrib, price, atr); total += per['vol']
    except Exception:
        per['vol'] = 0

    # NW slope
    try:
        w = weights.get('nw', 8); nw_s = float(latest.get('nw_slope', 0))
        slope_pct = (nw_s / (price + 1e-9)) * 10000
        if slope_pct > 0.1: contrib = min(w * (slope_pct / 0.2), w*2); reasons.append("NW slope positive")
        elif slope_pct < -0.1: contrib = -min(w * (abs(slope_pct) / 0.2), w*2); reasons.append("NW slope negative")
        else: contrib = 0
        per['nw_slope'] = normalize_by_volatility(contrib, price, atr); total += per['nw_slope']
    except Exception:
        per['nw_slope'] = 0

    # Funding contrarian
    try:
        w = weights.get('funding', 20); fr = funding.get('fundingRate', 0.0)
        if fr > 0.0006: per['funding'] = -w; reasons.append(f"Funding positive {fr:.6f}")
        elif fr < -0.0006: per['funding'] = w; reasons.append(f"Funding negative {fr:.6f}")
        else: per['funding'] = 0
        total += per['funding']
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

# ---------------- Scan engine (defensive) ----------------
@st.cache_data(ttl=120)
def run_scan(symbols, timeframes, weights, thresholds, top_n=100):
    results = []
    for sym in symbols[:top_n]:
        entry = {'symbol': sym, 'details': {}}
        best_score = None; best_tf = None
        buy_count = 0; strong_buy = 0; sell_count = 0
        mexc_sym = mexc_symbol_from(sym)
        for tf in timeframes:
            interval = INTERVAL_MAP.get(tf)
            if interval is None:
                entry['details'][tf] = None; continue
            try:
                df = fetch_contract_klines(mexc_sym, interval)
                if df is None or df.empty or len(df) < 30:
                    entry['details'][tf] = None; continue
                df_ind = compute_indicators(df)
                if df_ind is None or len(df_ind) < 3:
                    entry['details'][tf] = None; continue
                latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]
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
                entry['details'][tf] = None; continue
        entry['best_timeframe'] = best_tf
        entry['best_score'] = int(best_score) if best_score is not None else None
        entry['buy_count'] = buy_count
        entry['strong_buy_count'] = strong_buy
        entry['sell_count'] = sell_count
        results.append(entry)
    return pd.DataFrame(results)

# ---------------- UI: Sidebar ----------------
st.sidebar.header("Tarama Ayarları")
mode = st.sidebar.selectbox("Sembol kaynağı", ["Top 50 by vol","Top 100 by vol","Custom list"])
if mode == "Custom list":
    custom = st.sidebar.text_area("Virgülle ayrılmış semboller (ör: BTCUSDT,ETHUSDT)", value="BTCUSDT,ETHUSDT")
    symbols = [s.strip().upper() for s in custom.split(',') if s.strip()]
else:
    try:
        top = get_top_contracts_by_volume(50 if mode=="Top 50 by vol" else 100)
        symbols = [s.replace('_','') for s in top]
    except Exception:
        symbols = ["BTCUSDT","ETHUSDT"]

if not symbols:
    st.sidebar.error("Seçili sembol listesi boş. Lütfen custom list girin veya Top 50/100 seçin.")
    st.stop()

timeframes = st.sidebar.multiselect("Zaman dilimleri", options=ALL_TFS, default=DEFAULT_TFS)
max_possible = max(1, len(symbols))
top_n = st.sidebar.slider("İlk N coin taransın", min_value=1, max_value=max_possible, value=min(50, max_possible))

with st.sidebar.expander("Ağırlıklar (gelişmiş)"):
    w_ema = st.number_input("EMA", value=25)
    w_macd = st.number_input("MACD", value=20)
    w_rsi = st.number_input("RSI", value=15)
    w_bb = st.number_input("Bollinger", value=10)
    w_adx = st.number_input("ADX", value=7)
    w_vol = st.number_input("Volume", value=10)
    w_funding = st.number_input("Funding", value=30)
    w_nw = st.number_input("NW slope", value=8)
weights = {'ema': w_ema, 'macd': w_macd, 'rsi': w_rsi, 'bb': w_bb, 'adx': w_adx, 'vol': w_vol, 'funding': w_funding, 'nw': w_nw}

with st.sidebar.expander("Sinyal eşikleri"):
    strong_buy_t = st.slider("GÜÇLÜ AL ≥", 10, 100, 60)
    buy_t = st.slider("AL ≥", 0, 80, 20)
    sell_t = st.slider("SAT ≤", -80, 0, -20)
    strong_sell_t = st.slider("GÜÇLÜ SAT ≤", -100, -10, -60)
thresholds = (strong_buy_t, buy_t, sell_t, strong_sell_t)

scan = st.sidebar.button("🔍 Tara / Yenile")

if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'selected_symbol' not in st.session_state: st.session_state.selected_symbol = None
if 'selected_tf' not in st.session_state: st.session_state.selected_tf = DEFAULT_TFS[0]

if scan:
    with st.spinner("Tarama çalışıyor... Bu işlem coin sayısına göre zaman alır."):
        st.session_state.scan_results = run_scan(symbols, timeframes, weights, thresholds, top_n=top_n)
        st.session_state.last_scan = datetime.utcnow()

# ---------------- UI: Main 4-column layout ----------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli")
st.write("Sol sütunda sinyal listesi. Tıklayınca diğer panellerde TradingView grafiği, gösterge katkıları ve özet görünecek.")

df = st.session_state.scan_results
if df is None or df.empty:
    st.info("Henüz tarama yok veya sonuç boş. Yan panelden Tarayıcıyı çalıştırın.")
else:
    # create 4 columns
    col1, col2, col3, col4 = st.columns([2.2,3.2,2.6,2.0])

    # ========== COL 1: Signal list ==========
    with col1:
        st.markdown("### 📡 Sinyal Listesi")
        # quick filters
        q = st.selectbox("Hızlı filtre", ["All","Only Strong Buys","Only Buys","Only Sells"], index=0)
        df_view = df.copy()
        if q == "Only Strong Buys":
            df_view = df_view[df_view['strong_buy_count']>0]
        elif q == "Only Buys":
            df_view = df_view[df_view['buy_count']>0]
        elif q == "Only Sells":
            df_view = df_view[df_view['sell_count']>0]
        # sorting
        sort = st.selectbox("Sırala", ["Best Score","Strong Buy Count","Buy Count","Sell Count","Symbol"], index=0)
        desc = st.checkbox("Azalan sırada", value=True)
        if sort == "Best Score": df_view = df_view.sort_values(by='best_score', ascending=not desc, na_position='last')
        elif sort == "Strong Buy Count": df_view = df_view.sort_values(by='strong_buy_count', ascending=not desc)
        elif sort == "Buy Count": df_view = df_view.sort_values(by='buy_count', ascending=not desc)
        elif sort == "Sell Count": df_view = df_view.sort_values(by='sell_count', ascending=not desc)
        else: df_view = df_view.sort_values(by='symbol', ascending=not desc)

        # show limited list with clickable rows
        for _, row in df_view.head(60).iterrows():
            score = row.get('best_score')
            label = row.get('best_timeframe') or '-'
            sb = int(row.get('strong_buy_count',0))
            badge = "badge-green" if score is not None and score >= strong_buy_t else ("badge-red" if score is not None and score <= strong_sell_t else "small-muted")
            with st.container():
                st.markdown(f"<div class='coin-item block'><div style='display:flex; justify-content:space-between; align-items:center;'>"
                            f"<div><b>{row['symbol']}</b><div class='small-muted'>{label} • SB: {sb}</div></div>"
                            f"<div style='text-align:right'><div class='{badge}' style='padding:6px 10px;border-radius:8px'>{score if score is not None else '-'}</div>"
                            f"<div style='font-size:12px' class='small-muted'> {row.get('buy_count',0)} buys / {row.get('sell_count',0)} sells</div></div>"
                            f"</div></div>", unsafe_allow_html=True)
                if st.button("Detay", key=f"det_{row['symbol']}"):
                    st.session_state.selected_symbol = row['symbol']
                    st.session_state.selected_tf = row.get('best_timeframe') or DEFAULT_TFS[0]

    # ========== COL 2: TradingView chart ==========
    with col2:
        st.markdown("### 📈 Mum Grafiği (TradingView)")
        sel = st.session_state.selected_symbol or df.iloc[0]['symbol']
        sel_tf = st.session_state.selected_tf or DEFAULT_TFS[0]
        st.markdown(f"**Seçili:** {sel}   •   **TF:**")
        tf_choice = st.selectbox("Grafik TF", options=ALL_TFS, index=ALL_TFS.index(sel_tf) if sel_tf in ALL_TFS else 2, key="tv_tf")
        st.session_state.selected_tf = tf_choice
        tv_interval = TV_INTERVAL_MAP.get(tf_choice, '60')
        # TradingView widget embed
        tv_container_id = f"tv_{sel.replace('/','_')}_{tv_interval}"
        tv_html = f"""
        <div class="tradingview-widget-container" style="height:520px; width:100%">
          <div id="{tv_container_id}"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "container_id": "{tv_container_id}",
            "symbol": "BINANCE:{sel}",
            "interval": "{tv_interval}",
            "timezone": "Europe/Istanbul",
            "theme": "dark",
            "style": "1",
            "locale": "tr",
            "allow_symbol_change": true,
            "details": true,
            "withdateranges": true,
            "hide_side_toolbar": false,
            "enable_publishing": false,
            "hideideas": true
          }});
          </script>
        </div>
        """
        st.components.v1.html(tv_html, height=540)

    # ========== COL 3: Indicator contributions & reasons ==========
    with col3:
        st.markdown("### 🧩 Gösterge Katkıları & Nedenler")
        sel = st.session_state.selected_symbol or df.iloc[0]['symbol']
        # get details for selected
        sel_row = df[df['symbol']==sel]
        details = sel_row.iloc[0]['details'] if not sel_row.empty else {}
        if not details:
            st.write("Seçili coin için gösterge/veri yok.")
        else:
            # Show per-TF cards
            for tf in timeframes:
                cell = details.get(tf)
                if not cell:
                    st.markdown(f"**{tf}** — Veri yok.")
                    continue
                st.markdown(f"#### {tf} — {cell.get('label','-')}  •  Skor: **{cell.get('score','-')}**")
                # reasons
                reasons = cell.get('reasons', [])
                if reasons:
                    st.write("**Nedenler:** " + "; ".join(reasons))
                per_scores = cell.get('per_scores', {})
                if per_scores:
                    dfp = pd.DataFrame([{'indicator':k,'points':v} for k,v in per_scores.items()])
                    # Use plotly if available for a nicer horizontal bar
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(dfp.sort_values('points'), x='points', y='indicator', orientation='h', color='points', color_continuous_scale='RdYlGn')
                        fig.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10), template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # compact table with colored sign
                        st.table(dfp.set_index('indicator'))
                else:
                    st.write("Gösterge katkı verisi yok.")
                st.markdown("---")

    # ========== COL 4: Summary, stats, scoring logic ==========
    with col4:
        st.markdown("### 📋 Özet & Puanlama Mantığı")
        # Quick stats
        total_coins = len(df)
        total_strong_buy = df['strong_buy_count'].sum()
        avg_score = int(df['best_score'].dropna().mean()) if not df['best_score'].dropna().empty else '-'
        st.metric("Tarama sonucu - coin", f"{total_coins}")
        st.metric("Toplam Strong Buys (SUM TF SB)", f"{int(total_strong_buy)}")
        st.metric("Ortalama Skor", f"{avg_score}")

        st.markdown("#### Puanlama mantığı (özet)")
        st.write("""
        - Her gösterge **katkı** (örn. EMA, MACD, RSI, BB, ADX, Volume, NW-slope, Funding) ±weight kadar puan üretir.  
        - Katkılar **ATR** (volatility) oranına göre normalize edilir: volatilite yüksekse katkı azaltılır (daha muhafazakar).  
        - MACD crossover büyüklüğü, RSI seviyesinin derecesi, Bollinger dışına çıkma, ADX trend gücü gibi faktörler ayrı ayrı puanlandırılır.  
        - Funding rate kontraryen mantıkla değerlendirilir (pozitif funding -> crowd long -> hafif negatif).  
        - Tüm katkılar toplandıktan sonra skor -100..100 aralığına sıkıştırılır.  
        - Threshold'lar sidebar'dan değiştirilebilir.  
        """)
        st.markdown("#### SB (Strong Buy) nedir?")
        st.write("Belirli TF'lerde 'GÜÇLÜ AL' etiketi alan TF sayısıdır. Örneğin SB:2 -> iki zaman diliminde güçlü al.")

        st.markdown("#### Daha fazla seçenek")
        if st.button("Sinyal açıklamalarını aç (detaylı)"):
            st.write("""
            **Detaylı puanlama örneği (göstergeler):**
            - EMA alignment: +weight eğer 20>50>200, -weight tersi için.
            - MACD: crossover yönü ve histogram büyüklüğüne göre ölçeklendirme.
            - RSI: 0-100 aralığında; 25 altı = güçlü al, 75 üstü = güçlü sat (kademeli).
            - Bollinger: fiyat üstünde => momentum al; altında => sat.
            - ADX: 25-35 arası trend var, >35 güçlü trend (pozitif/negatif).
            - Volume oscillator: hacim patlaması al-sat gücünü destekler.
            - NW slope: smoothed momentum gösterir.
            """)

st.markdown("---")
st.caption("Uyarı: Bu araç yatırım tavsiyesi değildir. Sinyalleri kullanmadan önce mutlaka risk yönetimi uygulayın.")
