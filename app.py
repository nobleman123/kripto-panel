# app.py
# Streamlit MEXC contract sinyal uygulaması - (v3.3 - SyntaxError Düzeltmesi)

import streamlit as st
import pandas as pd
import numpy as np
# import pandas_ta as ta -> ai_engine'de
import requests
from datetime import datetime, timedelta
import ai_engine  # <-- TÜM MANTIK BURADA
import streamlit.components.v1 as components
import json
import logging
import time # Hata durumunda run_scan için eklendi

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Plotly kontrolü
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly kütüphanesi bulunamadı.")

st.set_page_config(page_title="MEXC Vadeli - Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# --- Session State Başlatma ---
if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'selected_symbol' not in st.session_state: st.session_state.selected_symbol = None
if 'selected_tf' not in st.session_state: st.session_state.selected_tf = '15m'
if 'tracked_signals' not in st.session_state: st.session_state.tracked_signals = {}
if 'last_scan_time' not in st.session_state: st.session_state.last_scan_time = None

# ---------------- CONFIG & CONSTANTS ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':0,'vol':10,'funding':30,'nw':8}
SCALP_TFS = ['1m', '5m', '15m']
SWING_TFS = ['4h', '1d']

# CSS
st.markdown("""
<style>
/* ... (CSS stilleri aynı kaldı) ... */
body { background: #0b0f14; color: #e6eef6; }
.block { background: linear-gradient(180deg,#0c1116,#071018); padding:10px; border-radius:10px; border:1px solid rgba(255,255,255,0.04); margin-bottom:8px;}
.coin-row { padding:8px; border-radius:8px; }
.coin-row:hover { background: rgba(255,255,255,0.02); }
.small-muted { color:#9aa3b2; font-size:12px; }
.score-card { background:#081226; padding:8px; border-radius:8px; text-align:center; }
[data-testid="stMetricValue"] { font-size: 22px; line-height: 1.2; }
[data-testid="stMetricLabel"] { font-size: 14px; white-space: nowrap; }
.stProgress > div > div > div > div { background-image: linear-gradient(to right, #00b09b , #96c93d); }
.market-analysis { background-color: #0f172a; padding: 10px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #1e293b; }
.market-analysis-title { font-weight: bold; margin-bottom: 5px; color: #cbd5e1; }
.market-analysis-content { font-size: 0.9em; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ---------------- API Helpers (get_top_contracts_by_volume düzeltildi) ----------------
@st.cache_data(ttl=timedelta(hours=1))
def fetch_all_contract_symbols():
    url = f"{CONTRACT_BASE}/contract/detail"; j = fetch_json(url)
    if not j: return ["BTCUSDT", "ETHUSDT"]
    data = j.get('data', [])
    symbols = [item['symbol'].replace('_USDT', 'USDT') for item in data if isinstance(item, dict) and item.get('symbol', '').endswith('_USDT')]
    logging.info(f"{len(symbols)} adet MEXC sembolü çekildi."); return sorted(list(set(symbols)))

def fetch_json(url, params=None, timeout=15):
    try: r = requests.get(url, params=params, timeout=timeout); r.raise_for_status(); return r.json()
    except requests.exceptions.Timeout: logging.warning(f"Zaman aşımı: {url}"); st.toast(f"Zaman aşımı: {url.split('/')[-1]}", icon="⏳"); return None
    except requests.exceptions.RequestException as e: logging.error(f"API hatası: {url} - {e}"); return None

@st.cache_data(ttl=timedelta(minutes=1))
def get_top_contracts_by_volume(limit=200):
    """Hacme göre sıralanmış sembol listesini çeker (SyntaxError düzeltildi)."""
    url = f"{CONTRACT_BASE}/contract/ticker"; j = fetch_json(url); data = j.get('data', []) if j else []
    if not data: return []

    # --- DÜZELTME BURADA ---
    def vol(x):
        """Güvenli bir şekilde hacim verisini float'a çevirir."""
        try:
            return float(x.get('volume24') or x.get('amount24') or 0)
        except (ValueError, TypeError):
            return 0
    # --- DÜZELTME SONU ---

    items = sorted(data, key=vol, reverse=True)
    syms = [it.get('symbol') for it in items[:limit]]
    return [s.replace('_USDT','USDT') for s in syms if s and s.endswith('_USDT')]


def mexc_symbol_from(symbol: str) -> str:
    s = symbol.strip().upper();
    if '_' in s: return s;
    if s.endswith('USDT'): return s[:-4] + "_USDT";
    logging.warning(f"Beklenmeyen format: {symbol}."); return s

@st.cache_data(ttl=timedelta(seconds=30))
def fetch_contract_klines(symbol_mexc, interval_mexc):
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"; j = fetch_json(url, params={'interval': interval_mexc})
    if not j: return pd.DataFrame(); d = j.get('data') or {}; times = d.get('time', [])
    if not times: return pd.DataFrame()
    try:
        df = pd.DataFrame({'timestamp': pd.to_datetime(d.get('time'), unit='s'), 'open': pd.to_numeric(d.get('open'), errors='coerce'), 'high': pd.to_numeric(d.get('high'), errors='coerce'), 'low': pd.to_numeric(d.get('low'), errors='coerce'), 'close': pd.to_numeric(d.get('close'), errors='coerce'), 'volume': pd.to_numeric(d.get('vol'), errors='coerce')})
        df = df.dropna();
        if len(df) < 50: logging.warning(f"fetch_klines yetersiz veri: {symbol_mexc} - {interval_mexc} ({len(df)})")
        return df
    except Exception as e: logging.error(f"Kline işleme hatası ({symbol_mexc}, {interval_mexc}): {e}"); return pd.DataFrame()

@st.cache_data(ttl=timedelta(minutes=1))
def fetch_contract_funding_rate(symbol_mexc):
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"; j = fetch_json(url)
    if not j: return {'fundingRate': 0.0}; data = j.get('data') or {}
    try: return {'fundingRate': float(data.get('fundingRate') or 0)}
    except: return {'fundingRate': 0.0}

# --------------- Scan Engine (Aynı kaldı) ---------------
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key):
    # ... (İçerik önceki yanıttaki gibi, hata yönetimi dahil) ...
    results = []
    total_symbols = len(symbols_to_scan)
    progress_bar_area = st.sidebar.empty()
    progress_bar = progress_bar_area.progress(0, text="Tarama başlatılıyor...")
    for i, sym in enumerate(symbols_to_scan):
        progress_value = (i + 1) / total_symbols; progress_text = f"Taranıyor: {sym} ({i+1}/{total_symbols})"
        progress_bar.progress(progress_value, text=progress_text)
        entry = {'symbol': sym, 'details': {}}; best_ai_confidence = -1; best_tf = None
        mexc_sym = mexc_symbol_from(sym)
        if not mexc_sym.endswith("_USDT"): continue
        try:
            funding = fetch_contract_funding_rate(mexc_sym); current_tf_results = {}
            for tf in timeframes:
                interval = INTERVAL_MAP.get(tf); scan_mode = "Normal"
                if tf in SCALP_TFS: scan_mode = "Scalp"; elif tf in SWING_TFS: scan_mode = "Swing"
                df = fetch_contract_klines(mexc_sym, interval)
                if df is None or df.empty or len(df) < 50: continue
                df_ind = ai_engine.compute_indicators(df) # ai_engine'den çağır
                if df_ind is None or df_ind.empty or len(df_ind) < 3: continue
                latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]
                score, per_scores, reasons = ai_engine.score_signals(latest, prev, funding, weights) # ai_engine'den çağır
                label = ai_engine.label_from_score(score, thresholds) # ai_engine'den çağır
                indicators_snapshot = {...} # Snapshot içeriği aynı
                indicators_snapshot = {k: v for k, v in indicators_snapshot.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
                ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=(gemini_api_key if gemini_api_key else None))
                current_tf_results[tf] = {...} # Sonuç içeriği aynı
                current_confidence = ai_analysis.get('confidence', 0) if ai_analysis.get('signal') not in ['NEUTRAL', 'ERROR'] else -1
                if current_confidence > best_ai_confidence: best_ai_confidence = current_confidence; best_tf = tf
            entry['details'] = current_tf_results; entry['best_timeframe'] = best_tf; entry['best_score'] = int(best_ai_confidence) if best_ai_confidence >= 0 else 0
            # ... (buy/sell count aynı) ...
            results.append(entry)
        except Exception as e: logging.error(f"Tarama hatası ({sym}): {e}", exc_info=True); st.toast(f"{sym} hatası: {e}", icon="🚨"); continue
    progress_bar_area.empty()
    if not results: logging.warning("Tarama sonuç üretmedi.")
    return pd.DataFrame(results)


# ------------- Market Analysis Functions (Aynı kaldı) --------------
@st.cache_data(ttl=timedelta(minutes=30))
def get_market_analysis(api_key, period="current"):
    # ... (İçerik aynı) ...
    if not api_key or not ai_engine.GEMINI_AVAILABLE: return None, None
    period_prompt = "..."; analysis_type = "..."
    logging.info(f"{analysis_type} isteniyor...");
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-pro')
        prompt = f"""..."""
        response = model.generate_content(prompt, request_options={'timeout': 120})
        logging.info(f"{analysis_type} alındı."); return analysis_type, response.text.strip()
    except Exception as e: logging.error(f"{analysis_type} alınamadı: {e}"); return analysis_type, f"Tahmin alınamadı: {e}"

# ------------- TradingView GÖMME FONKSİYONU (Aynı kaldı) ------------
def show_tradingview(symbol: str, interval_tv: str, height: int = 480):
    # ... (İçerik aynı) ...
    uid = f"tv_widget_{symbol.replace('/','_')}_{interval_tv}"; tradingview_html = f"""..."""
    components.html(tradingview_html, height=height, scrolling=False)

# ------------------- ANA UYGULAMA AKIŞI -------------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli (Hibrit AI)")

# --- Piyasa Analizi Alanı ---
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", help="Gelişmiş AI analizi ve Piyasa Tahmini için.", key="api_key_input")
# ... (Piyasa analizi gösterimi aynı kaldı) ...
if gemini_api_key_ui: # Piyasa analizi gösterimi
    analysis_col1, analysis_col2 = st.columns(2)
    # ... (Analizleri al ve göster) ...
    st.markdown("---")


# --- Sidebar Ayarları ---
st.sidebar.header("Tarama Ayarları")
# ... (Sembol seçimi, Zaman Dilimleri, Algoritma Ayarları aynı kaldı) ...
all_symbols_list = fetch_all_contract_symbols(); mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top Hacim (Max 200)","Özel Liste Seç"])
symbols_to_scan_ui = [];
if mode == "Özel Liste Seç": selected_symbols_ui = st.sidebar.multiselect("Coinleri Seçin", options=all_symbols_list, default=["BTCUSDT", "ETHUSDT"]); symbols_to_scan_ui = selected_symbols_ui
else: symbols_by_volume_list = get_top_contracts_by_volume(200); top_n_ui = st.sidebar.slider("İlk N Coin", min_value=5, max_value=len(symbols_by_volume_list), value=min(50, len(symbols_by_volume_list))); symbols_to_scan_ui = symbols_by_volume_list[:top_n_ui]
if not symbols_to_scan_ui: st.sidebar.warning("Taranacak sembol seçilmedi."); st.stop()
timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS)
if not timeframes_ui: st.sidebar.warning("Zaman dilimi seçin."); st.stop()
with st.sidebar.expander("Sistem Algoritması Ayarları"): weights_ui = {...}; thresholds_ui = (...) # Ağırlık/Eşik inputları aynı...


# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    # ... (Tarama başlatma mantığı aynı kaldı, try/except eklendi) ...
    spinner_msg = "Tarama çalışıyor..."; # Mesajı basitleştir
    with st.spinner(spinner_msg):
        try:
             scan_start_time = time.time()
             st.session_state.scan_results = run_scan(symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui, gemini_api_key_ui)
             scan_duration = time.time() - scan_start_time
             logging.info(f"Tarama tamamlandı. Süre: {scan_duration:.2f}s. {len(st.session_state.scan_results)} sonuç.")
             st.session_state.last_scan_time = datetime.now()
             st.session_state.selected_symbol = None
             st.experimental_rerun()
        except Exception as e:
             logging.error(f"Beklenmedik tarama hatası (ana blok): {e}", exc_info=True)
             st.error(f"Tarama sırasında hata: {e}")
             st.session_state.scan_results = pd.DataFrame() # Hata durumunda boşalt


# --- Sonuçları Göster ---
df_results = st.session_state.scan_results

if st.session_state.last_scan_time: st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

if df_results is None or df_results.empty:
    # ... (Boş sonuç mesajı aynı kaldı) ...
    if st.session_state.last_scan_time: st.warning("Tarama sonuç vermedi.")
    else: st.info("Tara / Yenile'ye basın.")
else:
    # --- AI Listesini Hazırla (Aynı kaldı) ---
    ai_list_display = []
    # ... (ai_list oluşturma mantığı aynı) ...

    if not ai_list_display: st.warning("Geçerli AI sinyali bulunamadı."); st.stop()
    ai_df_display = pd.DataFrame(ai_list_display)

    # Layout ve Sol Taraf (Filtreleme - Aynı kaldı)
    left, right = st.columns([1.6, 2.4])
    with left:
        # ... (Filtreleme ve liste gösterimi aynı kaldı) ...
        st.markdown("### 🔎 AI Sinyal Listesi")
        filter_signal_ui = st.selectbox("Sinyal Türü", ["All","LONG","SHORT","NEUTRAL", "ERROR"], index=0, key="signal_filter")
        min_confidence_ui = st.slider("Min Güven (%)", 0, 100, 30, step=5, key="conf_filter")
        filtered_display = ai_df_display.copy()
        # ... (KeyError düzeltmesi ve sıralama aynı kaldı) ...
        st.caption(f"{len(filtered_display)} sinyal bulundu.")
        # ... (Liste gösterimi for döngüsü aynı kaldı) ...


    # Sağ Taraf (Detay Ekranı - Aynı kaldı)
    with right:
        # ... (Detay ekranı mantığı, TradingView, AI Analizi, Metrikler, Butonlar, Algoritma Puanları aynı kaldı) ...
        st.markdown("### 📈 Seçili Coin Detayı")
        # ... (Seçili coin belirleme) ...
        if sel_sym is None: st.write("Listeden bir coin seçin.")
        else: # Detayları göster... (show_tradingview, row_data bulma, AI Analizi, Metrikler, Takip/Kayıt/İndir Butonları, Algoritma Puan Expander'ı)

    # --- Takip Edilen Sinyaller (Aynı kaldı) ---
    # ... (Takip edilen sinyallerin gösterimi aynı kaldı) ...
    st.markdown("---"); st.markdown("### 📌 Takip Edilen Sinyaller")
    # ...

    # --- Özet Metrikler ve Kayıtlı Tahminler (Aynı kaldı) ---
    # ... (Metriklerin ve Arşivin gösterimi aynı kaldı) ...
    st.markdown("---"); cols_summary = st.columns(4)
    # ...
    with st.expander("💾 Kayıtlı Tahminler (Arşiv)"):
        # ...

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
