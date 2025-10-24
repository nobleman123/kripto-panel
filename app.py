# app.py
# Streamlit MEXC contract sinyal uygulaması - (Gelişmiş Özellikler ve UI)

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime
import ai_engine  # <-- Gelişmiş motorumuz
import streamlit.components.v1 as components
import json
import logging # Hata ayıklama için

# Hata ayıklama loglamasını ayarla (opsiyonel)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# optional plotly for indicator bars
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly kütüphanesi bulunamadı. Grafik gösterimi sınırlı olacak.")

st.set_page_config(page_title="MEXC Vadeli - Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# ---------------- CONFIG ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':7,'vol':10,'funding':30,'nw':8}
SCALP_TFS = ['1m', '5m', '15m']
SWING_TFS = ['4h', '1d']

# CSS (Metric boyutları ayarlandı)
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
    font-size: 22px; /* Biraz küçültüldü */
    line-height: 1.2;
}
[data-testid="stMetricLabel"] {
    font-size: 14px; /* Biraz küçültüldü */
    white-space: nowrap; /* Etiketlerin alta kaymasını önle */
}
</style>
""", unsafe_allow_html=True)

# ---------------- API Helpers ----------------
@st.cache_data(ttl=3600) # Sembol listesini 1 saat cache'le
def fetch_all_contract_symbols():
    """Tüm MEXC vadeli işlem sembollerini çeker."""
    url = f"{CONTRACT_BASE}/contract/detail"
    try:
        j = fetch_json(url)
        data = j.get('data', [])
        # Sadece USDT paritelerini al ve '_USDT' formatından temizle
        symbols = [item['symbol'].replace('_USDT', 'USDT') for item in data if isinstance(item, dict) and item.get('symbol', '').endswith('_USDT')]
        logging.info(f"{len(symbols)} adet MEXC vadeli işlem sembolü çekildi.")
        return sorted(list(set(symbols))) # Tekrarları kaldır ve sırala
    except Exception as e:
        logging.error(f"Tüm MEXC sembolleri çekilemedi: {e}")
        st.error(f"MEXC sembol listesi alınamadı: {e}")
        return ["BTCUSDT", "ETHUSDT"] # Hata durumunda varsayılan

def fetch_json(url, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API isteği başarısız: {url} - Hata: {e}")
        # st.toast(f"API isteği hatası: {e}", icon="🚨") # Kullanıcıya kısa bilgi
        return None # Hata durumunda None dön

def fetch_contract_ticker():
    url = f"{CONTRACT_BASE}/contract/ticker"
    j = fetch_json(url)
    return j.get('data', []) if j else []

@st.cache_data(ttl=60) # Hacim verisini 1 dakika cache'le
def get_top_contracts_by_volume(limit=200):
    data = fetch_contract_ticker()
    if not data: return []
    def vol(x):
        try:
            return float(x.get('volume24') or x.get('amount24') or 0)
        except (ValueError, TypeError):
            return 0
    items = sorted(data, key=vol, reverse=True)
    syms = [it.get('symbol') for it in items[:limit]]
    return [s.replace('_USDT','USDT') for s in syms if s and s.endswith('_USDT')]

def mexc_symbol_from(symbol: str) -> str:
    s = symbol.strip().upper()
    if '_' in s: return s
    if s.endswith('USDT'): return s[:-4] + "_USDT"
    # Diğer pariteler için (eğer gerekirse) ek mantık eklenebilir
    logging.warning(f"Beklenmeyen sembol formatı: {symbol}. Dönüştürme başarısız olabilir.")
    return s # Varsayılan olarak orijinali döndür

@st.cache_data(ttl=30) # Kline verisini 30 saniye cache'le
def fetch_contract_klines(symbol_mexc, interval_mexc):
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    j = fetch_json(url, params={'interval': interval_mexc})
    if not j: return pd.DataFrame()
    d = j.get('data') or {}
    times = d.get('time', [])
    if not times: return pd.DataFrame()
    try:
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(d.get('time'), unit='s'),
            'open': pd.to_numeric(d.get('open'), errors='coerce'),
            'high': pd.to_numeric(d.get('high'), errors='coerce'),
            'low': pd.to_numeric(d.get('low'), errors='coerce'),
            'close': pd.to_numeric(d.get('close'), errors='coerce'),
            'volume': pd.to_numeric(d.get('vol'), errors='coerce')
        })
        df = df.dropna() # Eksik veri içeren satırları kaldır
        return df
    except Exception as e:
        logging.error(f"Kline verisi işlenirken hata ({symbol_mexc}, {interval_mexc}): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60) # Funding rate verisini 1 dakika cache'le
def fetch_contract_funding_rate(symbol_mexc):
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"
    j = fetch_json(url)
    if not j: return {'fundingRate': 0.0}
    data = j.get('data') or {}
    try:
        return {'fundingRate': float(data.get('fundingRate') or 0)}
    except (ValueError, TypeError):
        return {'fundingRate': 0.0}

# --------------- Indicators & scoring (robust) ----------------
# ... (compute_indicators, nw_smooth, label_from_score, score_signals fonksiyonları DEĞİŞMEDİ) ...
# (Bu fonksiyonlar önceki yanıtta olduğu gibi kalacak)
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
    # EMA
    try: df['ema20'] = ta.ema(df['close'], length=20)
    except Exception: df['ema20'] = np.nan
    try: df['ema50'] = ta.ema(df['close'], length=50)
    except Exception: df['ema50'] = np.nan
    try: df['ema200'] = ta.ema(df['close'], length=200)
    except Exception: df['ema200'] = np.nan
    # MACD
    try:
        macd = ta.macd(df['close'])
        df['macd_hist'] = macd.iloc[:,1] if isinstance(macd, pd.DataFrame) and macd.shape[1]>=2 else np.nan
    except Exception: df['macd_hist'] = np.nan
    # RSI
    try: df['rsi14'] = ta.rsi(df['close'], length=14)
    except Exception: df['rsi14'] = np.nan
    # Bollinger Bands
    try:
        bb = ta.bbands(df['close'])
        if isinstance(bb, pd.DataFrame) and bb.shape[1]>=3:
            df['bb_lower'] = bb.iloc[:,0]; df['bb_mid'] = bb.iloc[:,1]; df['bb_upper'] = bb.iloc[:,2]
        else: df[['bb_lower','bb_mid','bb_upper']] = np.nan
    except Exception: df[['bb_lower','bb_mid','bb_upper']] = np.nan
    # ADX
    try:
        adx = ta.adx(df['high'], df['low'], df['close'])
        df['adx14'] = adx['ADX_14'] if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx.columns else np.nan
    except Exception: df['adx14'] = np.nan
    # ATR
    try: df['atr14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    except Exception: df['atr14'] = np.nan
    # Volume Oscillator
    try:
        df['vol_ma_short'] = ta.sma(df['volume'], length=20)
        df['vol_ma_long'] = ta.sma(df['volume'], length=50)
        df['vol_osc'] = (df['vol_ma_short'] - df['vol_ma_long']) / (df['vol_ma_long'] + 1e-9)
    except Exception: df['vol_osc'] = np.nan
    # Nadaraya-Watson Slope
    try:
        sm = nw_smooth(df['close'].values, bandwidth=8)
        if len(sm) == len(df):
            df['nw_smooth'] = sm
            df['nw_slope'] = pd.Series(sm).diff().fillna(0)
        else: df['nw_smooth'] = np.nan; df['nw_slope'] = np.nan
    except Exception: df['nw_smooth'] = np.nan; df['nw_slope'] = np.nan
    
    # NaN değerleri sadece gerekli sütunlarda kontrol et ve kaldır
    required_cols = ['close', 'ema20', 'ema50', 'ema200', 'macd_hist', 'rsi14', 'bb_upper', 'bb_lower', 'atr14', 'vol_osc', 'nw_slope']
    df = df.dropna(subset=required_cols)
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
    # EMA
    try:
        w = weights.get('ema', 20)
        contrib = 0
        if latest.get('ema20', 0) > latest.get('ema50', 0) > latest.get('ema200', 0): contrib = +w; reasons.append("EMA bullish")
        elif latest.get('ema20', 0) < latest.get('ema50', 0) < latest.get('ema200', 0): contrib = -w; reasons.append("EMA bearish")
        per['ema'] = contrib; total += contrib
    except Exception: per['ema']=0
    # MACD
    try:
        w = weights.get('macd', 15)
        p_h = float(prev.get('macd_hist', 0)); l_h = float(latest.get('macd_hist', 0))
        contrib = 0
        if p_h < 0 and l_h > 0: contrib = w; reasons.append("MACD cross bullish")
        elif p_h > 0 and l_h < 0: contrib = -w; reasons.append("MACD cross bearish")
        per['macd'] = contrib; total += contrib
    except Exception: per['macd']=0
    # RSI
    try:
        w = weights.get('rsi', 12); rsi = float(latest.get('rsi14', 50))
        contrib = 0
        if rsi < 30: contrib = w; reasons.append("RSI oversold")
        elif rsi > 70: contrib = -w; reasons.append("RSI overbought")
        per['rsi'] = contrib; total += contrib
    except Exception: per['rsi']=0
    # Bollinger Bands
    try:
        w = weights.get('bb', 8)
        contrib = 0
        if latest.get('close', 0) > latest.get('bb_upper', float('inf')): contrib = -w; reasons.append("Above BB upper (Short)") # Üst bandı aşarsa Short sinyali
        elif latest.get('close', 0) < latest.get('bb_lower', 0): contrib = w; reasons.append("Below BB lower (Long)") # Alt bandı kırarsa Long sinyali
        per['bb'] = contrib; total += contrib
    except Exception: per['bb']=0
    # Volume Oscillator
    try:
        w = weights.get('vol', 6); vol_osc = float(latest.get('vol_osc', 0))
        contrib = 0
        if vol_osc > 0.4: contrib = w; reasons.append("Volume spike")
        # elif vol_osc < -0.4: contrib = -w; reasons.append("Volume drop") # Düşük hacim genelde nötr
        per['vol'] = contrib; total += contrib
    except Exception: per['vol']=0
    # NW Slope
    try:
        w = weights.get('nw', 8); nw_s = float(latest.get('nw_slope', 0))
        contrib = 0
        if nw_s > 0: contrib = w; reasons.append("NW slope +")
        elif nw_s < 0: contrib = -w; reasons.append("NW slope -")
        per['nw'] = contrib; total += contrib
    except Exception: per['nw']=0
    # Funding Rate
    try:
        w = weights.get('funding', 20); fr = funding.get('fundingRate', 0.0)
        contrib = 0
        if fr > 0.0006: contrib = -w; reasons.append("Funding High (Short)") # Yüksek funding short yönlü baskı
        elif fr < -0.0006: contrib = w; reasons.append("Funding Low (Long)") # Düşük funding long yönlü baskı
        per['funding'] = contrib; total += contrib
    except Exception: per['funding']=0
    
    total = int(max(min(total, 100), -100))
    return total, per, reasons

# ---------------- Scan engine (cached) ----------------
# @st.cache_data(ttl=120) # Cache'leme, API anahtarı değişince sorun yaratabilir, şimdilik kapalı
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key):
    """Ana tarama fonksiyonu."""
    results = []
    total_symbols = len(symbols_to_scan)
    progress_bar = st.progress(0, text="Tarama başlatılıyor...")

    for i, sym in enumerate(symbols_to_scan):
        progress_text = f"Taranıyor: {sym} ({i+1}/{total_symbols})"
        progress_bar.progress(i / total_symbols, text=progress_text)
        
        entry = {'symbol': sym, 'details': {}}
        best_ai_confidence = -1 # AI güvenine göre en iyiyi seçeceğiz
        best_tf = None
        mexc_sym = mexc_symbol_from(sym)
        if not mexc_sym.endswith("_USDT"): continue # Sadece USDT paritelerini işle

        funding = fetch_contract_funding_rate(mexc_sym)
        
        current_tf_results = {} # Geçici olarak TF sonuçlarını tut

        for tf in timeframes:
            interval = INTERVAL_MAP.get(tf)
            if interval is None: continue
            
            # Scalp/Swing modu belirle
            scan_mode = "Normal"
            if tf in SCALP_TFS: scan_mode = "Scalp"
            elif tf in SWING_TFS: scan_mode = "Swing"

            df = fetch_contract_klines(mexc_sym, interval)
            if df is None or df.empty or len(df) < 50: # Daha fazla veri isteyelim
                logging.warning(f"Yetersiz kline verisi: {sym} - {tf}")
                continue
            df_ind = compute_indicators(df)
            if df_ind is None or df_ind.empty or len(df_ind) < 3:
                logging.warning(f"İndikatör hesaplama hatası: {sym} - {tf}")
                continue
            
            latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]
            
            score, per_scores, reasons = score_signals(latest, prev, funding, weights)
            label = label_from_score(score, thresholds)
            
            indicators_snapshot = {
                'symbol': sym, # AI'a sembolü de gönderelim
                'timeframe': tf, # AI'a zaman dilimini de gönderelim
                'scan_mode': scan_mode, # AI'a modu gönderelim
                'score': int(score),
                'price': float(latest['close']),
                'rsi14': float(latest.get('rsi14', np.nan)),
                'macd_hist': float(latest.get('macd_hist', np.nan)),
                'vol_osc': float(latest.get('vol_osc', np.nan)),
                'atr14': float(latest.get('atr14', np.nan)),
                'nw_slope': float(latest.get('nw_slope', np.nan)),
                'bb_upper': float(latest.get('bb_upper', np.nan)),
                'bb_lower': float(latest.get('bb_lower', np.nan)),
                'funding_rate': funding.get('fundingRate', 0.0) # Funding rate'i de ekle
            }
            # Eksik veri kontrolü ve temizleme
            indicators_snapshot = {k: v for k, v in indicators_snapshot.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

            try:
                ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=gemini_api_key)
            except Exception as e:
                logging.error(f"AI Analiz hatası ({sym}, {tf}): {e}")
                st.toast(f"{sym}-{tf} AI analizi başarısız: {e}", icon="⚠️")
                ai_analysis = {"signal": "ERROR", "confidence": 0, "explanation": f"AI Hatası: {e}"}


            current_tf_results[tf] = {
                'score': int(score), 'label': label, 'price': float(latest['close']),
                'per_scores': per_scores, 'reasons': reasons,
                'ai_analysis': ai_analysis
            }
            
            # En iyi AI güvenine sahip zaman dilimini bul
            current_confidence = ai_analysis.get('confidence', 0) if ai_analysis.get('signal') not in ['NEUTRAL', 'ERROR'] else -1
            if current_confidence > best_ai_confidence:
                best_ai_confidence = current_confidence
                best_tf = tf
            
        entry['details'] = current_tf_results
        entry['best_timeframe'] = best_tf
        entry['best_score'] = int(best_ai_confidence) if best_ai_confidence >= 0 else 0 # Negatif güven olmaz
        
        # Eski sayımlar (isteğe bağlı olarak kalabilir ama ana metrik AI güveni)
        entry['buy_count'] = sum(1 for d in current_tf_results.values() if d and d['label'] in ['AL', 'GÜÇLÜ AL'])
        entry['strong_buy_count'] = sum(1 for d in current_tf_results.values() if d and d['label'] == 'GÜÇLÜ AL')
        entry['sell_count'] = sum(1 for d in current_tf_results.values() if d and d['label'] in ['SAT', 'GÜÇLÜ SAT'])
        
        results.append(entry)

    progress_bar.empty() # Tarama bitince barı kaldır
    return pd.DataFrame(results)

# ------------- Daily Market Forecast --------------
@st.cache_data(ttl=7200) # 2 saatte bir tahmin al
def get_daily_forecast(api_key):
    """Gemini kullanarak günlük piyasa yönü tahmini alır."""
    if not api_key or not ai_engine.GEMINI_AVAILABLE:
        return None
    try:
        logging.info("Günlük piyasa tahmini isteniyor...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = """
        Sen bir kripto para piyasa analistisin. Genel piyasa duyarlılığını (haberler, korku/açgözlülük endeksi, BTC dominansı, majör coin trendleri, genel makroekonomik durum vb.) analiz ederek bugünkü genel piyasa yönü tahminini yap.

        Olası Tahminler:
        - GÜÇLÜ YÜKSELİŞ (Genel alım baskısı, olumlu haberler)
        - YÜKSELİŞ (Hafif alım baskısı, nötr/olumlu hava)
        - NÖTR/KARARSIZ (Yön belirsiz, yatay hareket beklentisi)
        - DÜŞÜŞ (Hafif satış baskısı, nötr/olumsuz hava)
        - GÜÇLÜ DÜŞÜŞ (Genel satış baskısı, olumsuz haberler)

        Cevabını SADECE bu tahminlerden biri olarak ver ve kısa bir (1-2 cümle) gerekçe ekle.
        Örnek: YÜKSELİŞ - BTC'nin direnci kırması ve fonlama oranlarının pozitife dönmesiyle hafif alım beklentisi hakim.
        """
        response = model.generate_content(prompt)
        logging.info("Günlük piyasa tahmini alındı.")
        return response.text.strip()
    except Exception as e:
        logging.error(f"Günlük piyasa tahmini alınamadı: {e}")
        return f"Tahmin alınamadı: {e}"

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
            "symbol": "BINANCE:{symbol}", // TradingView için genellikle BINANCE kullanılır
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

# --- Günlük Piyasa Tahmini ---
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", help="Gelişmiş AI analizi ve Piyasa Tahmini için.")
daily_forecast = get_daily_forecast(gemini_api_key_ui)
if daily_forecast:
    st.markdown(f"**🗓️ Günlük Piyasa Yönü Tahmini:** {daily_forecast}")
    st.markdown("---")


st.sidebar.header("Tarama Ayarları")

# --- SEMBOL SEÇİMİ (Geliştirildi) ---
all_symbols = fetch_all_contract_symbols() # Tüm sembolleri başta çek
mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top Hacim (Max 200)","Özel Liste Seç"])

symbols_to_scan = []
if mode == "Özel Liste Seç":
    selected_symbols = st.sidebar.multiselect("Taramak İstediğiniz Coinleri Seçin", options=all_symbols, default=["BTCUSDT", "ETHUSDT"])
    symbols_to_scan = selected_symbols
    top_n_scan = len(symbols_to_scan) # Özel listedeki tümünü tara
else: # Top Hacim
    symbols_by_volume = get_top_contracts_by_volume(200)
    top_n_scan = st.sidebar.slider("İlk N Coin Taransın", min_value=5, max_value=len(symbols_by_volume), value=min(50, len(symbols_by_volume)))
    symbols_to_scan = symbols_by_volume[:top_n_scan]

if not symbols_to_scan:
    st.sidebar.error("Taranacak sembol bulunamadı.")
    st.stop()
# --- SEMBOL SEÇİMİ SONU ---


timeframes = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS)

with st.sidebar.expander("Sistem Algoritması Sinyal Mantığı"):
    st.markdown("""
    Algoritma (Gemini AI kapalıyken), sinyalleri puanlamak için bu kuralları kullanır:
    - **RSI:** Aşırı alım/satım bölgeleri (70/30) ve genel trend (50 üzeri/altı) puanlanır.
    - **MACD:** Histogramın sıfır çizgisini kesmesi (crossover) ve pozitif/negatif bölgede olması puanlanır.
    - **NW Slope:** Trendin yönünü belirler, en yüksek ağırlıklardan birine sahiptir.
    - **Bollinger Bantları:** Fiyatın bant dışına taşması ters yönlü (reversal) sinyal olarak puanlanır.
    - **Hacim:** Yüksek hacim mevcut sinyali destekler.
    - **Funding Rate:** Aşırı yüksek/düşük funding oranları ters yönlü sinyali destekler.
    """)

with st.sidebar.expander("Sistem Algoritması Ağırlıkları (Heuristic)"):
    # ... (Ağırlık inputları değişmedi) ...
    w_ema = st.number_input("EMA", value=DEFAULT_WEIGHTS['ema'])
    w_macd = st.number_input("MACD", value=DEFAULT_WEIGHTS['macd'])
    w_rsi = st.number_input("RSI", value=DEFAULT_WEIGHTS['rsi'])
    w_bb = st.number_input("BB", value=DEFAULT_WEIGHTS['bb'])
    w_adx = st.number_input("ADX", value=DEFAULT_WEIGHTS['adx'])
    w_vol = st.number_input("VOL", value=DEFAULT_WEIGHTS['vol'])
    w_funding = st.number_input("Funding", value=DEFAULT_WEIGHTS['funding'])
    w_nw = st.number_input("NW slope", value=DEFAULT_WEIGHTS['nw'])
weights_ui = {'ema':w_ema,'macd':w_macd,'rsi':w_rsi,'bb':w_bb,'adx':w_adx,'vol':w_vol,'funding':w_funding,'nw':w_nw}

with st.sidebar.expander("Sistem Algoritması Sinyal Eşikleri"):
    # ... (Eşik inputları değişmedi) ...
    strong_buy_t = st.slider("GÜÇLÜ AL ≥", 10, 100, 60)
    buy_t = st.slider("AL ≥", 0, 80, 20)
    sell_t = st.slider("SAT ≤", -80, 0, -20)
    strong_sell_t = st.slider("GÜÇLÜ SAT ≤", -100, -10, -60)
thresholds_ui = (strong_buy_t, buy_t, sell_t, strong_sell_t)

# --- Sinyal Takip Sistemi Başlangıcı ---
if 'tracked_signals' not in st.session_state:
    st.session_state.tracked_signals = {} # Sembol -> Sinyal Detayı

# --- UI Başlangıcı ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'selected_symbol' not in st.session_state: st.session_state.selected_symbol = None
if 'selected_tf' not in st.session_state: st.session_state.selected_tf = DEFAULT_TFS[0]

if scan:
    if not timeframes:
        st.sidebar.error("Lütfen en az bir zaman dilimi seçin.")
        st.stop()
    spinner_msg = "Tarama çalışıyor (Algoritma Modu)..."
    if gemini_api_key_ui and ai_engine.GEMINI_AVAILABLE:
        spinner_msg = "Tarama çalışıyor (Gemini AI Modu)... Bu biraz daha uzun sürebilir..."
    
    with st.spinner(spinner_msg):
        st.session_state.scan_results = run_scan(symbols_to_scan, timeframes, weights_ui, thresholds_ui, gemini_api_key_ui)
        st.session_state.last_scan = datetime.now() # Yerel saat

df = st.session_state.scan_results
if df is None or df.empty:
    st.info("Henüz tarama yok veya seçili kriterlere uygun coin bulunamadı.")
else:
    # --- AI Analizlerini Hazırla ---
    ai_list = []
    for _, row in df.iterrows():
        best_tf = row.get('best_timeframe')
        details = row.get('details', {}) or {}
        snapshot = details.get(best_tf) if best_tf and details else None # En iyi TF'e göre al
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
            'target_info': ai_analysis,
            'algo_score': snapshot.get('score'), # Algoritma skoru
            'algo_label': snapshot.get('label'), # Algoritma etiketi
            'per_scores': snapshot.get('per_scores'),
            'reasons': snapshot.get('reasons', [])
        })
    ai_df = pd.DataFrame(ai_list)

    # Layout
    left, right = st.columns([1.6, 2.4])

    with left:
        st.markdown("### 🔎 AI Sinyal Listesi (filtreleyip tıklayın)")
        
        filter_signal = st.selectbox("Sinyal Türü", ["All","LONG","SHORT","NEUTRAL", "ERROR"], index=0)
        min_confidence = st.slider("AI Minimum Güven (%)", 0, 100, 30, step=5) # Default 30'a düşürüldü
        
        filtered = ai_df.copy()
        if filter_signal != "All": filtered = filtered[filtered['ai_signal'] == filter_signal]
        filtered = filtered[filtered['ai_confidence'] >= min_confidence]
        filtered = filtered.sort_values(by='ai_confidence', ascending=False)
        
        st.caption(f"{len(filtered)} sinyal bulundu.")

        for _, r in filtered.head(120).iterrows():
            emoji = "⚪"
            if r['ai_signal']=='LONG': emoji='🚀'
            elif r['ai_signal']=='SHORT': emoji='🔻' # Daha belirgin Short ikonu
            elif r['ai_signal']=='ERROR': emoji='⚠️'

            cols = st.columns([0.6,2,1])
            cols[0].markdown(f"<div style='font-size:20px'>{emoji}</div>", unsafe_allow_html=True)
            cols[1].markdown(f"**{r['symbol']}** • {r['best_tf']} \nAI: **{r['ai_signal']}** (%{r['ai_confidence']}) Algo: {r['algo_label']} ({r['algo_score']})") # Karşılaştırma eklendi
            if cols[2].button("Detay", key=f"det_{r['symbol']}"):
                st.session_state.selected_symbol = r['symbol']
                st.session_state.selected_tf = r['best_tf']

    with right:
        st.markdown("### 📈 Seçili Coin Detayı")
        sel = st.session_state.selected_symbol or (ai_df.iloc[0]['symbol'] if not ai_df.empty else None)
        sel_tf = st.session_state.selected_tf or (ai_df.iloc[0]['best_tf'] if not ai_df.empty else DEFAULT_TFS[0])

        if sel is None:
            st.write("Listeden bir coin seçin.")
        else:
            st.markdown(f"**{sel}** • TF: **{sel_tf}**")
            interval_tv = TV_INTERVAL_MAP.get(sel_tf, '60')
            
            show_tradingview(sel, interval_tv, height=400) # Yüksekliği biraz azalttık
            
            row = next((x for x in ai_list if x['symbol']==sel), None)
            if row:
                st.markdown("#### 🧠 AI Analizi ve Ticaret Planı")
                st.markdown(row['ai_text'])
                
                ti = row['target_info']
                entry_val = ti.get('entry')
                stop_val = ti.get('stop_loss')
                target_val = ti.get('take_profit')

                # Seviyeleri daha görünür yap
                if entry_val is not None and stop_val is not None and target_val is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Giriş (Entry)", f"{entry_val:.5f}")
                    c2.metric("Stop Loss", f"{stop_val:.5f}", delta=f"{((stop_val-entry_val)/entry_val*100):.2f}%", delta_color="inverse")
                    c3.metric("Hedef (Target)", f"{target_val:.5f}", delta=f"{((target_val-entry_val)/entry_val*100):.2f}%")
                elif entry_val is not None:
                     st.metric("Fiyat", f"{entry_val:.5f}")


                # --- Sinyal Takip Butonu ---
                track_key = f"track_{sel}_{sel_tf}"
                is_tracked = track_key in st.session_state.tracked_signals
                track_button_label = "❌ Takipten Çıkar" if is_tracked else "📌 Sinyali Takip Et"
                if st.button(track_button_label, key=track_key):
                    if is_tracked:
                        del st.session_state.tracked_signals[track_key]
                        st.toast(f"{sel} ({sel_tf}) takipten çıkarıldı.", icon="🗑️")
                        st.experimental_rerun() # Sayfayı yenile buton durumunu güncellemek için
                    else:
                        st.session_state.tracked_signals[track_key] = {
                            'symbol': sel,
                            'tf': sel_tf,
                            'signal': row['ai_signal'],
                            'confidence': row['ai_confidence'],
                            'entry': entry_val,
                            'stop': stop_val,
                            'target': target_val,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        st.toast(f"{sel} ({sel_tf}) takibe alındı!", icon="📌")
                        st.experimental_rerun()

                # Kayıt/İndirme Butonları
                b1, b2, b3 = st.columns([1,1,1])
                if b1.button("✅ Tahmin Başarılı"): # Daha anlaşılır etiket
                    rec = {'symbol': sel, 'tf': row['best_tf'],
                           'entry': entry_val, 'stop': stop_val, 'target': target_val,
                           'price_at_mark': row['price'], 'ai_signal': row['ai_signal'], 'ai_confidence': row['ai_confidence'],
                           'outcome': 'Success', # Sonucu kaydet
                           'timestamp': datetime.utcnow().isoformat()}
                    ok = ai_engine.save_record(rec)
                    if ok: st.success("Başarılı tahmin kaydedildi.")
                    else: st.error("Kayıt başarısız.")
                if b2.button("❌ Tahmin Başarısız"):
                    rec = {'symbol': sel, 'tf': row['best_tf'],
                           'entry': entry_val, 'stop': stop_val, 'target': target_val,
                           'price_at_mark': row['price'], 'ai_signal': row['ai_signal'], 'ai_confidence': row['ai_confidence'],
                           'outcome': 'Failure', # Sonucu kaydet
                           'timestamp': datetime.utcnow().isoformat()}
                    ok = ai_engine.save_record(rec)
                    if ok: st.warning("Başarısız tahmin kaydedildi.")
                    else: st.error("Kayıt başarısız.")

                if b3.button("📥 Analizi İndir"):
                    st.download_button("JSON İndir", data=json.dumps(row, indent=2, ensure_ascii=False), file_name=f"{sel}_{sel_tf}_signal.json")

                # Eski skorlama sisteminin katkılarını göster (opsiyonel)
                with st.expander("Algoritma Puan Katkıları (Eski Sistem)"):
                    per = row.get('per_scores', {})
                    if per and PLOTLY_AVAILABLE:
                        # ... (Plotly kodu değişmedi) ...
                        dfp = pd.DataFrame([{'indicator':k,'points':v} for k,v in per.items()])
                        fig = px.bar(dfp.sort_values('points'), x='points', y='indicator', orientation='h', color='points', color_continuous_scale='RdYlGn')
                        fig.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10), template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    elif per:
                        st.table(pd.DataFrame([{'indicator':k,'points':v} for k,v in per.items()]).set_index('indicator'))
                    else:
                        st.write("Algoritma skor verisi yok.")

    # --- Takip Edilen Sinyaller ---
    st.markdown("---")
    st.markdown("### 📌 Takip Edilen Sinyaller")
    if st.session_state.tracked_signals:
        tracked_list = list(st.session_state.tracked_signals.values())
        tracked_df = pd.DataFrame(tracked_list)
        st.dataframe(tracked_df[['timestamp', 'symbol', 'tf', 'signal', 'confidence', 'entry', 'stop', 'target']].sort_values(by='timestamp', ascending=False), use_container_width=True)
    else:
        st.info("Henüz takip edilen sinyal yok. Detay ekranından 'Sinyali Takip Et' butonuna tıklayarak ekleyebilirsiniz.")


    # Özet metrikler ve Kayıtlı Tahminler
    st.markdown("---")
    cols = st.columns(4)
    cols[0].metric("Taranan Coin", f"{len(df)}")
    cols[1].metric("Toplam LONG Sinyal (>%30)", f"{len(ai_df[(ai_df['ai_signal'] == 'LONG') & (ai_df['ai_confidence'] >= 30)])}")
    cols[2].metric("Toplam SHORT Sinyal (>%30)", f"{len(ai_df[(ai_df['ai_signal'] == 'SHORT') & (ai_df['ai_confidence'] >= 30)])}")
    cols[3].metric("Kayıtlı Tahmin Sayısı", f"{len(ai_engine.load_records())}")

    with st.expander("✅ Kayıtlı Tahminler (Arşiv)"):
        recs = ai_engine.load_records()
        if recs:
            st.dataframe(pd.DataFrame(recs).sort_values(by='timestamp', ascending=False), use_container_width=True)
        else:
            st.write("Henüz kayıtlı tahmin yok.")

st.caption("Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
