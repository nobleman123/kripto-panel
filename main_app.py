# main_app.py
# Streamlit MEXC contract sinyal uygulaması - full, mobile-friendly, stable TradingView embed

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
import ai_engine
import streamlit.components.v1 as components  # <-- HATA ÇÖZÜMÜ İÇİN EKLENDİ

# optional plotly for indicator bars
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

st.set_page_config(page_title="MEXC Vadeli - Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# ---------------- CONFIG ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
# mapping for our simplified TFs
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':7,'vol':10,'funding':30,'nw':8}

# CSS
st.markdown("""
<style>
body { background: #0b0f14; color: #e6eef6; }
.block { background: linear-gradient(180deg,#0c1116,#071018); padding:10px; border-radius:10px; border:1px solid rgba(255,255,255,0.04); margin-bottom:8px;}
.coin-row { padding:8px; border-radius:8px; }
.coin-row:hover { background: rgba(255,255,255,0.02); }
.small-muted { color:#9aa3b2; font-size:12px; }
.score-card { background:#081226; padding:8px; border-radius:8px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers & MEXC endpoints ----------------
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
    return [s.replace('_','') for s in syms if s]

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
        df = pd.DataFrame({'timestamp': pd.to_datetime(d.get('time'), unit='s'),
                           'open': d.get('open'),'high': d.get('high'),'low': d.get('low'),
                           'close': d.get('close'),'volume': d.get('vol')})
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

# --------------- Indicators & scoring (robust) ----------------
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

def label_from_score(score, thresholds):
    strong_buy_t, buy_t, sell_t, strong_sell_t = thresholds
    if score is None: return "NO DATA"
    if score >= strong_buy_t: return "GÜÇLÜ AL"
    if score >= buy_t: return "AL"
    if score <= strong_sell_t: return "GÜÇLÜ SAT"
    if score <= sell_t: return "SAT"
    return "TUT"

def score_signals(latest, prev, funding, weights):
    per = {}; reasons = []; total = 0
    try:
        w = weights.get('ema', 20)
        contrib = 0
        if latest['ema20'] > latest['ema50'] > latest['ema200']:
            contrib = +w; reasons.append("EMA alignment bullish")
        elif latest['ema20'] < latest['ema50'] < latest['ema200']:
            contrib = -w; reasons.append("EMA alignment bearish")
        per['ema'] = contrib; total += contrib
    except Exception:
        per['ema']=0
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
        per['macd']=0
    try:
        w = weights.get('rsi', 12); rsi = float(latest.get('rsi14', np.nan))
        contrib = 0
        if rsi < 30: contrib = w; reasons.append("RSI oversold")
        elif rsi > 70: contrib = -w; reasons.append("RSI overbought")
        per['rsi'] = contrib; total += contrib
    except Exception:
        per['rsi']=0
    try:
        w = weights.get('bb', 8)
        if latest['close'] > latest['bb_upper']: contrib = w; reasons.append("Above BB upper")
        elif latest['close'] < latest['bb_lower']: contrib = -w; reasons.append("Below BB lower")
        else: contrib = 0
        per['bb'] = contrib; total += contrib
    except Exception:
        per['bb']=0
    try:
        w = weights.get('vol', 6); vol_osc = float(latest.get('vol_osc', 0))
        if vol_osc > 0.4: contrib = w; reasons.append("Volume spike")
        elif vol_osc < -0.4: contrib = -w; reasons.append("Volume drop")
        else: contrib = 0
        per['vol'] = contrib; total += contrib
    except Exception:
        per['vol']=0
    try:
        w = weights.get('nw', 8); nw_s = float(latest.get('nw_slope', 0))
        if nw_s > 0: contrib = w; reasons.append("NW slope +")
        elif nw_s < 0: contrib = -w; reasons.append("NW slope -")
        else: contrib = 0
        per['nw'] = contrib; total += contrib
    except Exception:
        per['nw']=0
    try:
        w = weights.get('funding', 20); fr = funding.get('fundingRate', 0.0)
        if fr > 0.0006: per['funding'] = -w; reasons.append("Funding +")
        elif fr < -0.0006: per['funding'] = w; reasons.append("Funding -")
        else: per['funding'] = 0
        total += per['funding']
    except Exception:
        per['funding']=0
    total = int(max(min(total, 100), -100))
    return total, per, reasons

# ---------------- Scan engine (cached) ----------------
@st.cache_data(ttl=120)
def run_scan(symbols, timeframes, weights, thresholds, top_n=100):
    results = []
    for sym in symbols[:top_n]:
        entry = {'symbol': sym, 'details': {}}
        best_score = None; best_tf = None; buy_count=0; strong_buy=0; sell_count=0
        mexc_sym = mexc_symbol_from(sym)
        for tf in timeframes:
            interval = INTERVAL_MAP.get(tf)
            if interval is None:
                entry['details'][tf] = None; continue
            df = fetch_contract_klines(mexc_sym, interval)
            if df is None or df.empty or len(df) < 40:
                entry['details'][tf] = None; continue
            df_ind = compute_indicators(df)
            if df_ind is None or len(df_ind) < 3:
                entry['details'][tf] = None; continue
            latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]
            funding = fetch_contract_funding_rate(mexc_sym)
            score, per_scores, reasons = score_signals(latest, prev, funding, weights)
            label = label_from_score(score, thresholds)
            entry['details'][tf] = {'score': int(score), 'label': label, 'price': float(latest['close']),
                                   'per_scores': per_scores, 'reasons': reasons,
                                   'rsi14': float(latest.get('rsi14', np.nan)),
                                   'macd_hist': float(latest.get('macd_hist', np.nan)),
                                   'vol_osc': float(latest.get('vol_osc', np.nan)),
                                   'atr14': float(latest.get('atr14', np.nan)),
                                   'nw_slope': float(latest.get('nw_slope', np.nan))}
            if best_score is None or score > best_score:
                best_score = score; best_tf = tf
            if label in ['AL','GÜÇLÜ AL']: buy_count += 1
            if label == 'GÜÇLÜ AL': strong_buy += 1
            if label in ['SAT','GÜÇLÜ SAT']: sell_count += 1
        entry['best_timeframe'] = best_tf
        entry['best_score'] = int(best_score) if best_score is not None else None
        entry['buy_count'] = buy_count
        entry['strong_buy_count'] = strong_buy
        entry['sell_count'] = sell_count
        results.append(entry)
    return pd.DataFrame(results)

# ------------- YÜKLEDİĞİNİZ KODDAKİ HATALI FONKSİYON ("removeChild" HATASININ KAYNAĞI) ------------
# def show_tradingview(symbol: str, interval_tv: str, height: int = 480):
#     """
#     Safe TradingView embed:
#     - uses a persistent placeholder in session_state
#     - constructs a unique container id per render to avoid DOM removeChild conflicts
#     """
#     ph_key = "tv_placeholder"
#     if ph_key not in st.session_state:
#         st.session_state[ph_key] = st.empty()
#     placeholder = st.session_state[ph_key]
#     uid = f"tv_{symbol.replace('/','_')}_{interval_tv}_{int(time.time()*1000)}"
#     ...
#     placeholder.html(tradingview_html, height=height)
# --------------------------------------------------------------------------------


# +++++++++++++ OLMASI GEREKEN DÜZELTİLMİŞ FONKSİYON (HATA ÇÖZÜMÜ) +++++++++++++
def show_tradingview(symbol: str, interval_tv: str, height: int = 480):
    """
    Güvenli TradingView yerleştirmesi (removeChild hatası çözümü):
    - st.empty() ve session_state KULLANILMADAN, 
    - st.components.v1.html kullanılarak izole bir iframe içinde yükler.
    """
    
    # Widget'ın içindeki div için benzersiz bir ID
    uid = f"tv_widget_{symbol.replace('/','_')}_{interval_tv}"
    
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:{height}px; width:100%;">
      <div id="{uid}" style="height:100%; width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      (function() {{
        try {{
          new TradingView.widget({{
            "container_id": "{uid}",
            "symbol": "BINANCE:{symbol}",
            "interval": "{interval_tv}",
            "autosize": true,
            "timezone": "Europe/Istanbul",
            "theme": "dark",
            "style": "1",
            "locale": "tr",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "hide_side_toolbar": false,
            "hideideas": true
          }});
        }} catch(e) {{
          var el = document.getElementById("{uid}");
          if(el) el.innerHTML = "<div style='color:#f66;padding:10px;'>Grafik yüklenemedi: "+e.toString()+"</div>";
        }}
      }})(); 
      </script>
    </div>
    """
    
    # st.components.v1.html kullanarak HTML'i güvenle render et
    # Bu, onu bir iframe'e koyar ve DOM çakışmalarını önler.
    components.html(tradingview_html, height=height, scrolling=False)
# +++++++++++++ DÜZELTİLMİŞ FONKSİYONUN SONU +++++++++++++


# ---------------- UI ----------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli (Full)")
st.sidebar.header("Tarama Ayarları")

mode = st.sidebar.selectbox("Sembol kaynağı", ["Top by volume (200)","Custom list"])
if mode == "Custom list":
    custom = st.sidebar.text_area("Virgülle ayrılmış semboller (örn: BTCUSDT,ETHUSDT)", value="BTCUSDT,ETHUSDT")
    symbols = [s.strip().upper() for s in custom.split(',') if s.strip()]
else:
    symbols = get_top_contracts_by_volume(200)

if not symbols:
    st.sidebar.error("Sembol listesi boş.")
    st.stop()

timeframes = st.sidebar.multiselect("Zaman dilimleri", options=ALL_TFS, default=DEFAULT_TFS)
top_n = st.sidebar.slider("İlk N coin taransın", min_value=5, max_value=min(200,len(symbols)), value=min(50,len(symbols)))
with st.sidebar.expander("Ağırlıklar"):
    w_ema = st.number_input("EMA", value=DEFAULT_WEIGHTS['ema'])
    w_macd = st.number_input("MACD", value=DEFAULT_WEIGHTS['macd'])
    w_rsi = st.number_input("RSI", value=DEFAULT_WEIGHTS['rsi'])
    w_bb = st.number_input("BB", value=DEFAULT_WEIGHTS['bb'])
    w_adx = st.number_input("ADX", value=DEFAULT_WEIGHTS['adx'])
    w_vol = st.number_input("VOL", value=DEFAULT_WEIGHTS['vol'])
    w_funding = st.number_input("Funding", value=DEFAULT_WEIGHTS['funding'])
    w_nw = st.number_input("NW slope", value=DEFAULT_WEIGHTS['nw'])
weights = {'ema':w_ema,'macd':w_macd,'rsi':w_rsi,'bb':w_bb,'adx':w_adx,'vol':w_vol,'funding':w_funding,'nw':w_nw}
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
    with st.spinner("Tarama çalışıyor. Bu bir kaç dakika alabilir..."):
        st.session_state.scan_results = run_scan(symbols, timeframes, weights, thresholds, top_n=top_n)
        st.session_state.last_scan = datetime.utcnow()

df = st.session_state.scan_results
if df is None or df.empty:
    st.info("Henüz tarama yok. Yan panelden Tara / Yenile ile başlat.")
else:
    # Prepare AI predictions for selected snapshot per coin (best TF)
    ai_list = []
    for _, row in df.iterrows():
        best_tf = row.get('best_timeframe')
        details = row.get('details', {}) or {}
        snapshot = details.get(best_tf) if details else None
        if not snapshot: continue
        indicators = {'score': snapshot.get('score'),
                      'rsi14': snapshot.get('rsi14'),
                      'macd_hist': snapshot.get('macd_hist'),
                      'vol_osc': snapshot.get('vol_osc'),
                      'atr14': snapshot.get('atr14'),
                      'nw_slope': snapshot.get('nw_slope'),
                      'price': snapshot.get('price')}
        ai_expl = ai_engine.predict_probability(indicators)
        entry_target = ai_engine.compute_entry_target_stop(price=indicators['price'], atr=indicators.get('atr14', None))
        ai_list.append({'symbol': row['symbol'], 'best_score': row.get('best_score'), 'best_tf': best_tf,
                        'label': snapshot.get('label'), 'price': indicators['price'], 'ai_prob': ai_expl['probability'],
                        'ai_text': ai_expl['text'], 'indicators': indicators, 'target_info': entry_target,
                        'per_scores': snapshot.get('per_scores'), 'reasons': snapshot.get('reasons', [])})
    ai_df = pd.DataFrame(ai_list)

    # Layout: left list, right detail (responsive)
    left, right = st.columns([1.6, 2.4])

    with left:
        st.markdown("### 🔎 Sinyal Listesi (filtreleyip tıklayın)")
        filter_label = st.selectbox("Kategori", ["All","GÜÇLÜ AL","AL","SAT","GÜÇLÜ SAT","TUT"], index=0)
        min_prob = st.slider("AI min olasılık", 0.0, 1.0, 0.35, step=0.05)
        filtered = ai_df.copy()
        if filter_label != "All": filtered = filtered[filtered['label'] == filter_label]
        filtered = filtered[filtered['ai_prob'] >= min_prob]
        # sort by ai_prob desc
        filtered = filtered.sort_values(by='ai_prob', ascending=False)
        for _, r in filtered.head(120).iterrows():
            emoji = "⚪"
            if r['label']=='GÜÇLÜ AL': emoji='🚀'
            elif r['label']=='AL': emoji='🟢'
            elif r['label']=='SAT' or r['label']=='GÜÇLÜ SAT': emoji='🔴'
            cols = st.columns([0.6,2,1])
            cols[0].markdown(f"<div style='font-size:20px'>{emoji}</div>", unsafe_allow_html=True)
            cols[1].markdown(f"**{r['symbol']}** •  {r['best_tf']}  \nSkor: {r['best_score']}  •  AI: {r['ai_prob']:.2f}")
            if cols[2].button("Detay", key=f"det_{r['symbol']}"):
                st.session_state.selected_symbol = r['symbol']
                st.session_state.selected_tf = r['best_tf']

    with right:
        st.markdown("### 📈 Seçili Coin Detayı")
        sel = st.session_state.selected_symbol or (ai_df.iloc[0]['symbol'] if not ai_df.empty else None)
        sel_tf = st.session_state.selected_tf or DEFAULT_TFS[0]
        if sel is None:
            st.write("Listeden bir coin seçin.")
        else:
            st.markdown(f"**{sel}** •  TF: {sel_tf}")
            interval = TV_INTERVAL_MAP.get(sel_tf, '60')
            
            # Düzeltilmiş, güvenli TradingView fonksiyonu burada çağrılıyor
            show_tradingview(sel, interval, height=420)
            
            # display AI info & target/stop
            row = next((x for x in ai_list if x['symbol']==sel), None)
            if row:
                st.markdown("#### AI Tahmini")
                st.write(row['ai_text'])
                ti = row['target_info']
                st.markdown(f"**Giriş:** {ti['entry']:.6f}   **Stop:** {ti['stop']:.6f}   **Hedef:** {ti['target']:.6f}")
                # buttons: mark correct (save), export
                b1, b2, b3 = st.columns([1,1,1])
                if b1.button("✅ Tahmini Doğru İşaretle"):
                    rec = {'symbol': sel, 'tf': row['best_tf'], 'entry': ti['entry'], 'stop': ti['stop'],
                           'target': ti['target'], 'price_at_mark': row['price'], 'ai_prob': row['ai_prob'],
                           'score': row['best_score'], 'timestamp': datetime.utcnow().isoformat()}
                    ok = ai_engine.save_record(rec)
                    if ok:
                        st.success("Tahmin doğru olarak kaydedildi.")
                    else:
                        st.error("Kayıt başarısız.")
                if b2.button("❌ Hatalı Tahmin"):
                    st.warning("Hatalı olarak işaretlendi.")
                if b3.button("📥 JSON İndir"):
                    st.download_button("İndir", data=str(row), file_name=f"{sel}_signal.json")
                st.markdown("#### Gösterge Katkıları")
                per = row.get('per_scores', {})
                if per:
                    dfp = pd.DataFrame([{'indicator':k,'points':v} for k,v in per.items()])
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(dfp.sort_values('points'), x='points', y='indicator', orientation='h', color='points', color_continuous_scale='RdYlGn')
                        fig.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10), template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.table(dfp.set_index('indicator'))
                else:
                    st.write("Gösterge verisi yok.")
    # bottom: summary & records
    st.markdown("---")
    cols = st.columns([1,1,1,1])
    cols[0].metric("Tarama coin", f"{len(df)}")
    cols[1].metric("Toplam SB", f"{int(df['strong_buy_count'].sum()) if not df.empty else 0}")
    cols[2].metric("Ortalama Skor", f"{int(df['best_score'].dropna().mean()) if not df['best_score'].dropna().empty else '-'}")
    cols[3].metric("Kayıtlı Doğru Tahmin", f"{len(ai_engine.load_records())}")

    st.markdown("### ✅ Doğru Tahminler (Arşiv)")
    recs = ai_engine.load_records()
    if recs:
        st.dataframe(pd.DataFrame(recs).sort_values(by='timestamp', ascending=False))
    else:
        st.write("Henüz kayıt yok.")

st.caption("Uyarı: Eğitim/deneme amaçlıdır. Yatırım tavsiyesi değildir.")
