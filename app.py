# app.py
# Streamlit MEXC contract sinyal uygulaması - (Hata düzeltildi, UI ve AI geliştirildi)

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
import ai_engine  # <-- Yeni motorumuz
import streamlit.components.v1 as components
import json

# optional plotly for indicator bars
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

st.set_page_config(page_title="MEXC Vadeli - Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# ---------------- CONFIG ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
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
/* st.metric için daha büyük yazı tipi */
[data-testid="stMetricValue"] {
    font-size: 24px;
}
[data-testid="stMetricLabel"] {
    font-size: 16px;
}
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
def run_scan(symbols, timeframes, weights, thresholds, gemini_api_key, top_n=100):
    results = []
    
    # top_n, taranacak sembol listesinin uzunluğunu geçemez
    symbols_to_scan = symbols[:min(top_n, len(symbols))]
    
    for sym in symbols_to_scan:
        entry = {'symbol': sym, 'details': {}}
        best_score = None; best_tf = None; buy_count=0; strong_buy=0; sell_count=0
        mexc_sym = mexc_symbol_from(sym)
        
        # Sadece 1 kez funding rate çek
        funding = fetch_contract_funding_rate(mexc_sym)
        
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
            
            # Eski skorlama sistemi (Heuristic AI için girdi olarak)
            score, per_scores, reasons = score_signals(latest, prev, funding, weights)
            label = label_from_score(score, thresholds)
            
            # --- YENİ AI ANALİZ MOTORU ÇAĞRISI ---
            indicators_snapshot = {
                'score': int(score),
                'price': float(latest['close']),
                'rsi14': float(latest.get('rsi14', np.nan)),
                'macd_hist': float(latest.get('macd_hist', np.nan)),
                'vol_osc': float(latest.get('vol_osc', np.nan)),
                'atr14': float(latest.get('atr14', np.nan)),
                'nw_slope': float(latest.get('nw_slope', np.nan)),
                'bb_upper': float(latest.get('bb_upper', np.nan)), # <-- YENİ EKLENDİ
                'bb_lower': float(latest.get('bb_lower', np.nan))  # <-- YENİ EKLENDİ
            }
            ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=gemini_api_key)
            # --- BİTTİ ---

            entry['details'][tf] = {
                'score': int(score), 'label': label, 'price': float(latest['close']),
                'per_scores': per_scores, 'reasons': reasons,
                'ai_analysis': ai_analysis  # <-- Yeni AI analiz sonucunu buraya kaydet
            }
            
            current_best_metric = ai_analysis.get('confidence', 0) if ai_analysis.get('signal') != 'NEUTRAL' else 0
            if best_score is None or current_best_metric > best_score:
                best_score = current_best_metric
                best_tf = tf
            
            if label in ['AL','GÜÇLÜ AL']: buy_count += 1
            if label == 'GÜÇLÜ AL': strong_buy += 1
            if label in ['SAT','GÜÇLÜ SAT']: sell_count += 1
            
        entry['best_timeframe'] = best_tf
        entry['best_score'] = int(best_score) if best_score is not None else None # Artık bu 'en iyi güven'
        entry['buy_count'] = buy_count
        entry['strong_buy_count'] = strong_buy
        entry['sell_count'] = sell_count
        results.append(entry)
    return pd.DataFrame(results)

# ------------- GÜVENLİ TradingView GÖMME FONKSİYONU (Değişiklik Yok) ------------
def show_tradingview(symbol: str, interval_tv: str, height: int = 480):
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
    components.html(tradingview_html, height=height, scrolling=False)

# ---------------- UI ----------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli (Hibrit AI)")
st.sidebar.header("Tarama Ayarları")

gemini_api_key = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", help="Daha gelişmiş analiz için Gemini API anahtarınızı girin.")

mode = st.sidebar.selectbox("Sembol kaynağı", ["Top by volume (200)","Custom list"])

# --- HATA DÜZELTMESİ: SLIDER LOGIĞI ---
if mode == "Custom list":
    custom = st.sidebar.text_area("Virgülle ayrılmış semboller (örn: BTCUSDT,ETHUSDT)", value="BTCUSDT,ETHUSDT")
    symbols = [s.strip().upper() for s in custom.split(',') if s.strip()]
    top_n = len(symbols) # Custom listedeki tüm sembolleri tara
else:
    symbols = get_top_contracts_by_volume(200)
    # Slider'ı sadece "Top by volume" seçiliyse göster ve max_value'yu sabitle
    top_n = st.sidebar.slider("İlk N coin taransın", min_value=5, max_value=200, value=50)

if not symbols:
    st.sidebar.error("Sembol listesi boş. Lütfen 'Custom list' kullanıyorsanız en az bir sembol girin.")
    st.stop()
# --- HATA DÜZELTMESİ SONU ---

timeframes = st.sidebar.multiselect("Zaman dilimleri", options=ALL_TFS, default=DEFAULT_TFS)

# --- YENİ: ALGORİTMA AÇIKLAMASI ---
with st.sidebar.expander("Sistem Algoritması Sinyal Mantığı"):
    st.markdown("""
    Algoritma (Gemini AI kapalıyken), sinyalleri puanlamak için bu kuralları kullanır:
    - **RSI (Göreceli Güç Endeksi):** 30'un altı (Aşırı Satım) **LONG** için güçlü bir sinyaldir. 70'in üstü (Aşırı Alım) **SHORT** için güçlü bir sinyaldir.
    - **MACD Histogram:** Pozitif (0'ın üzeri) olması **LONG** momentumunu, negatif olması **SHORT** momentumunu destekler.
    - **Nadaraya-Watson (nw_slope):** Bu, trendin yönünü belirler. Pozitif eğim **LONG**, negatif eğim **SHORT** sinyalini güçlendirir.
    - **Bollinger Bantları (BB):** Fiyatın üst bandı (`bb_upper`) kırması bir **SHORT** (geri çekilme) sinyali, alt bandı (`bb_lower`) kırması bir **LONG** (tepki) sinyali olarak değerlendirilir.
    - **Hacim (vol_osc):** Yüksek hacim, mevcut trendin (RSI, MACD veya NW) gücünü onaylar.
    """)

# --- YENİ: İSİMLENDİRME GÜNCELLENDİ ---
with st.sidebar.expander("Sistem Algoritması Ağırlıkları (Heuristic)"):
    w_ema = st.number_input("EMA", value=DEFAULT_WEIGHTS['ema'])
    w_macd = st.number_input("MACD", value=DEFAULT_WEIGHTS['macd'])
    w_rsi = st.number_input("RSI", value=DEFAULT_WEIGHTS['rsi'])
    w_bb = st.number_input("BB", value=DEFAULT_WEIGHTS['bb'])
    w_adx = st.number_input("ADX", value=DEFAULT_WEIGHTS['adx'])
    w_vol = st.number_input("VOL", value=DEFAULT_WEIGHTS['vol'])
    w_funding = st.number_input("Funding", value=DEFAULT_WEIGHTS['funding'])
    w_nw = st.number_input("NW slope", value=DEFAULT_WEIGHTS['nw'])
weights = {'ema':w_ema,'macd':w_macd,'rsi':w_rsi,'bb':w_bb,'adx':w_adx,'vol':w_vol,'funding':w_funding,'nw':w_nw}
with st.sidebar.expander("Sistem Algoritması Sinyal Eşikleri"):
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
    spinner_msg = "Tarama çalışıyor (Heuristic Mod)..."
    if gemini_api_key and ai_engine.GEMINI_AVAILABLE:
        spinner_msg = "Tarama çalışıyor (Gemini AI Modu)... Bu biraz daha uzun sürebilir..."
    
    with st.spinner(spinner_msg):
        st.session_state.scan_results = run_scan(symbols, timeframes, weights, thresholds, gemini_api_key, top_n=top_n)
        st.session_state.last_scan = datetime.utcnow()

df = st.session_state.scan_results
if df is None or df.empty:
    st.info("Henüz tarama yok. Yan panelden Tara / Yenile ile başlat.")
else:
    # --- YENİ: AI Analizlerini Hazırla ---
    ai_list = []
    for _, row in df.iterrows():
        best_tf = row.get('best_timeframe')
        details = row.get('details', {}) or {}
        snapshot = details.get(best_tf) if details else None
        if not snapshot: continue
        
        ai_analysis = snapshot.get('ai_analysis')
        if not ai_analysis: continue

        ai_list.append({
            'symbol': row['symbol'],
            'best_tf': best_tf,
            'price': snapshot.get('price'),
            'ai_signal': ai_analysis.get('signal', 'NEUTRAL'),
            'ai_confidence': ai_analysis.get('confidence', 0),
            'ai_text': ai_analysis.get('explanation', 'Açıklama yok.'),
            'target_info': ai_analysis, # entry, stop_loss, take_profit içerir
            'per_scores': snapshot.get('per_scores'), # Eski skorlar
            'reasons': snapshot.get('reasons', []) # Eski nedenler
        })
    ai_df = pd.DataFrame(ai_list)

    # Layout
    left, right = st.columns([1.6, 2.4])

    with left:
        st.markdown("### 🔎 AI Sinyal Listesi (filtreleyip tıklayın)")
        
        # --- YENİ: Filtreler ---
        filter_signal = st.selectbox("Sinyal Türü", ["All","LONG","SHORT","NEUTRAL"], index=0)
        min_confidence = st.slider("AI Minimum Güven (%)", 0, 100, 50, step=5)
        
        filtered = ai_df.copy()
        if filter_signal != "All": filtered = filtered[filtered['ai_signal'] == filter_signal]
        filtered = filtered[filtered['ai_confidence'] >= min_confidence]
        
        # Güvene göre sırala
        filtered = filtered.sort_values(by='ai_confidence', ascending=False)
        
        for _, r in filtered.head(120).iterrows():
            emoji = "⚪"
            if r['ai_signal']=='LONG': emoji='🚀'
            elif r['ai_signal']=='SHORT': emoji='🔴'
            
            cols = st.columns([0.6,2,1])
            cols[0].markdown(f"<div style='font-size:20px'>{emoji}</div>", unsafe_allow_html=True)
            cols[1].markdown(f"**{r['symbol']}** • {r['best_tf']} \nAI Sinyal: **{r['ai_signal']}** (%{r['ai_confidence']})")
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
            st.markdown(f"**{sel}** • TF: **{sel_tf}**")
            interval = TV_INTERVAL_MAP.get(sel_tf, '60')
            
            # Güvenli TradingView Gömme
            show_tradingview(sel, interval, height=420)
            
            # --- YENİ: AI Analizini Göster ---
            row = next((x for x in ai_list if x['symbol']==sel), None)
            if row:
                st.markdown("#### 🧠 AI Analizi ve Ticaret Planı")
                st.markdown(row['ai_text']) # Gemini'den veya Heuristic'ten gelen tam açıklama
                
                # --- YENİ: GÖRÜNÜR SEVİYELER (st.metric) ---
                ti = row['target_info']
                if ti.get('stop_loss') and ti.get('take_profit'):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Giriş (Entry)", f"{ti['entry']:.5f}")
                    c2.metric("Stop Loss", f"{ti['stop_loss']:.5f}")
                    c3.metric("Hedef (Target)", f"{ti['take_profit']:.5f}")
                
                # Kayıt butonları
                b1, b2, b3 = st.columns([1,1,1])
                if b1.button("✅ Tahmini Doğru İşaretle"):
                    rec = {'symbol': sel, 'tf': row['best_tf'], 
                           'entry': ti['entry'], 'stop': ti['stop_loss'], 'target': ti['take_profit'],
                           'price_at_mark': row['price'], 'ai_signal': row['ai_signal'], 'ai_confidence': row['ai_confidence'],
                           'timestamp': datetime.utcnow().isoformat()}
                    ok = ai_engine.save_record(rec)
                    if ok: st.success("Tahmin doğru olarak kaydedildi.")
                    else: st.error("Kayıt başarısız.")
                if b2.button("❌ Hatalı Tahmin"):
                    st.warning("Hatalı olarak işaretlendi.")
                if b3.button("📥 JSON İndir"):
                    st.download_button("İndir", data=json.dumps(row, indent=2, ensure_ascii=False), file_name=f"{sel}_signal.json")
                
                # Eski skorlama sisteminin katkılarını göster (opsiyonel)
                st.markdown("#### Heuristic Gösterge Katkıları (Eski Sistem)")
                per = row.get('per_scores', {})
                if per and PLOTLY_AVAILABLE:
                    dfp = pd.DataFrame([{'indicator':k,'points':v} for k,v in per.items()])
                    fig = px.bar(dfp.sort_values('points'), x='points', y='indicator', orientation='h', color='points', color_continuous_scale='RdYlGn')
                    fig.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10), template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
                elif per:
                    st.table(pd.DataFrame([{'indicator':k,'points':v} for k,v in per.items()]).set_index('indicator'))
                else:
                    st.write("Heuristic skor verisi yok.")

    # Özet metrikler
    st.markdown("---")
    cols = st.columns([1,1,1,1])
    cols[0].metric("Tarama coin", f"{len(df)}")
    cols[1].metric("Toplam LONG Sinyal", f"{len(ai_df[ai_df['ai_signal'] == 'LONG'])}" if not ai_df.empty else 0)
    cols[2].metric("Toplam SHORT Sinyal", f"{len(ai_df[ai_df['ai_signal'] == 'SHORT'])}" if not ai_df.empty else 0)
    cols[3].metric("Kayıtlı Doğru Tahmin", f"{len(ai_engine.load_records())}")

    st.markdown("### ✅ Doğru Tahminler (Arşiv)")
    recs = ai_engine.load_records()
    if recs:
        st.dataframe(pd.DataFrame(recs).sort_values(by='timestamp', ascending=False))
    else:
        st.write("Henüz kayıt yok.")

st.caption("Uyarı: Eğitim/deneme amaçlıdır. Yatırım tavsiyesi değildir.")
