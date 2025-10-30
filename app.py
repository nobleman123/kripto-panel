# app.py
# Streamlit MEXC contract sinyal uygulaması - (v5.3 - Coin Listesi Fallback Düzeltmesi)

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import ai_engine
import streamlit.components.v1 as components
import json
import logging
import time
import math

# --- Temel Ayarlar ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="MEXC Vadeli - Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# --- Plotly Kontrolü ---
try: import plotly.express as px; PLOTLY_AVAILABLE = True
except ImportError: PLOTLY_AVAILABLE = False; logging.warning("Plotly yok.")

# --- Session State Başlatma (Güvenli) ---
default_values = {
    'scan_results': pd.DataFrame(), 'selected_symbol': None, 'selected_tf': '15m',
    'tracked_signals': {}, 'last_scan_time': None, 'active_tab': "📊 Genel AI"
}
for key, default_value in default_values.items():
    if key not in st.session_state: st.session_state[key] = default_value

# ---------------- CONFIG & CONSTANTS ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']
DEFAULT_TFS_REQUESTED = ['15m','1h','4h']
DEFAULT_TFS = [tf for tf in DEFAULT_TFS_REQUESTED if tf in ALL_TFS]
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':0,'vol':10,'funding':30,'nw':8}
SCALP_TFS = ['1m', '5m', '15m']; SWING_TFS = ['4h', '1d']
EMA_TREND_LENGTH = ai_engine.EMA_TREND_LENGTH
SPECTER_ATR_LENGTH = ai_engine.SPECTER_ATR_LENGTH
MA_TYPES = ['EMA', 'SMA', 'SMMA', 'WMA', 'VWMA']
MAX_SIGNALS_TO_SHOW = 150

# --- CSS ---
st.markdown("""<style>...</style>""", unsafe_allow_html=True) # Önceki CSS aynı

# ---------------- API Yardımcı Fonksiyonları (Sağlamlaştırıldı) ----------------
@st.cache_data(ttl=timedelta(hours=1))
def fetch_all_contract_symbols():
    url = f"{CONTRACT_BASE}/contract/detail"
    logging.info("Tüm semboller çekiliyor...")
    data = fetch_json(url)
    if data and 'data' in data and isinstance(data['data'], list):
        symbols = [item['symbol'].replace('_USDT', 'USDT') for item in data['data']
                   if isinstance(item, dict) and item.get('symbol', '').endswith('_USDT')]
        logging.info(f"{len(symbols)} sembol bulundu.")
        return sorted(list(set(symbols)))
    logging.error("fetch_all_contract_symbols: Geçersiz veri formatı veya API hatası.")
    return ["BTCUSDT", "ETHUSDT"] # Fallback

def fetch_json(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        content_type = r.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            return r.json()
        else:
            logging.error(f"JSON beklenirken farklı içerik tipi alındı: {content_type} - URL: {url}")
            return None
    except requests.exceptions.Timeout:
        logging.warning(f"Zaman aşımı: {url}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API hatası: {url} - {e}")
        return None
    except json.JSONDecodeError as e:
         logging.error(f"JSON Decode Hatası: {url} - {e}")
         return None

@st.cache_data(ttl=timedelta(minutes=1))
def get_top_contracts_by_volume(limit=200):
    url = f"{CONTRACT_BASE}/contract/ticker"
    logging.info(f"Top {limit} hacimli sembol çekiliyor...")
    data = fetch_json(url)
    if not data or 'data' not in data or not isinstance(data['data'], list):
        logging.error("get_top_contracts_by_volume: Geçersiz veri formatı veya API hatası.")
        return [] # Hata durumunda boş liste döndür

    def vol(x):
        try:
            volume_str = x.get('volume24') or x.get('amount24') or '0'
            return float(volume_str)
        except (ValueError, TypeError, AttributeError):
            return 0

    valid_items = [item for item in data['data'] if isinstance(item, dict)]
    try:
        items = sorted(valid_items, key=vol, reverse=True)
    except Exception as e:
        logging.error(f"Hacme göre sıralama hatası: {e}")
        items = valid_items

    syms = [it.get('symbol') for it in items[:limit] if it.get('symbol')]
    result = [s.replace('_USDT','USDT') for s in syms if s.endswith('_USDT')]
    logging.info(f"{len(result)} hacimli sembol işlendi.")
    return result

def mexc_symbol_from(symbol: str) -> str: # USDT ekler
    s = symbol.strip().upper();
    if not s: return ""
    if '_' in s: return s;
    if s.endswith('USDT'): return s[:-4] + "_USDT";
    return s + "_USDT"

@st.cache_data(ttl=timedelta(seconds=30))
def fetch_contract_klines(symbol_mexc, interval_mexc): # Daha sağlam
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    data = fetch_json(url, params={'interval': interval_mexc})
    if not data or 'data' not in data or not isinstance(data['data'], dict):
         logging.warning(f"Geçersiz kline verisi: {symbol_mexc} - {interval_mexc}")
         return pd.DataFrame()
    d = data['data']; times = d.get('time')
    if not isinstance(times, list) or not times:
         logging.warning(f"Kline 'time' verisi eksik: {symbol_mexc} - {interval_mexc}")
         return pd.DataFrame()
    try:
        df = pd.DataFrame({'timestamp': pd.to_datetime(d.get('time'), unit='s', errors='coerce'),'open': pd.to_numeric(d.get('open'), errors='coerce'),'high': pd.to_numeric(d.get('high'), errors='coerce'),'low': pd.to_numeric(d.get('low'), errors='coerce'),'close': pd.to_numeric(d.get('close'), errors='coerce'),'volume': pd.to_numeric(d.get('vol'), errors='coerce')})
        df = df.dropna(subset=['timestamp', 'close']).reset_index(drop=True)
        if len(df) < 50: logging.warning(f"fetch_klines az veri: {symbol_mexc} ({len(df)})")
        return df
    except Exception as e:
        logging.error(f"Kline işleme hatası ({symbol_mexc}): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=timedelta(minutes=1))
def fetch_contract_funding_rate(symbol_mexc): # Daha sağlam
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"
    data = fetch_json(url)
    if not data or 'data' not in data or not isinstance(data['data'], dict): return {'fundingRate': 0.0}
    try: return {'fundingRate': float(data['data'].get('fundingRate') or 0)}
    except (ValueError, TypeError): return {'fundingRate': 0.0}


# ---------------- Scan Engine (app.py içinde tanımlı - Önceki gibi) ----------------
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key,
             vr_lookback, vr_confirm, vr_vol_multi, combo_adx_thresh,
             specter_ma_type, specter_ma_length):
    """Ana tarama fonksiyonu - ai_engine'deki analizleri çağırır."""
    results = []
    total_symbols = len(symbols_to_scan)
    progress_bar_area = st.sidebar.empty()
    progress_bar = progress_bar_area.progress(0, text="Tarama başlatılıyor...")

    for i, sym in enumerate(symbols_to_scan):
        progress_value = (i + 1) / total_symbols
        progress_text = f"Taranıyor: {sym} ({i+1}/{total_symbols})"
        try: progress_bar.progress(progress_value, text=progress_text)
        except Exception: pass

        entry = {'symbol': sym, 'details': {}}
        best_ai_confidence = -1; best_tf = None
        mexc_sym = mexc_symbol_from(sym)
        if not mexc_sym.endswith("_USDT"): continue

        try:
            funding = fetch_contract_funding_rate(mexc_sym)
            if funding is None: funding = {'fundingRate': 0.0}
            current_tf_results = {}

            for tf in timeframes:
                interval = INTERVAL_MAP.get(tf); scan_mode = "Normal"
                if tf in SCALP_TFS: scan_mode = "Scalp"
                elif tf in SWING_TFS: scan_mode = "Swing"

                df = fetch_contract_klines(mexc_sym, interval)
                min_bars_needed = max(50, vr_lookback + vr_confirm + 2, SPECTER_ATR_LENGTH + 5, ai_engine.EMA_TREND_LENGTH + 5)
                if df is None or df.empty or len(df) < min_bars_needed:
                    logging.debug(f"Yetersiz kline ({sym}-{tf}): {len(df) if df is not None else 0}/{min_bars_needed}")
                    continue

                df_ind = ai_engine.compute_indicators(df, ma_type=specter_ma_type, ma_length=specter_ma_length)
                if df_ind is None or df_ind.empty or len(df_ind) < 3:
                    logging.warning(f"İndikatör hesaplanamadı: {sym}-{tf}")
                    continue

                latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]
                score, per_scores, reasons = ai_engine.score_signals(latest, prev, funding, weights)
                label = ai_engine.label_from_score(score, thresholds)
                volume_reversal_analysis = ai_engine.analyze_volume_reversal(df_ind, look_back=vr_lookback, confirm_in=vr_confirm, vol_multiplier=vr_vol_multi)
                strategy_combo_analysis = ai_engine.analyze_strategy_combo(latest, adx_threshold=combo_adx_thresh)
                specter_trend_analysis = ai_engine.analyze_specter_trend(df_ind)

                indicators_snapshot = {
                    'symbol': sym, 'timeframe': tf, 'scan_mode': scan_mode, 'score': int(score), 'price': float(latest['close']),
                    'rsi14': latest.get('rsi14'), 'macd_hist': latest.get('macd_hist'), 'vol_osc': latest.get('vol_osc'),
                    'atr14': latest.get('atr14'), 'nw_slope': latest.get('nw_slope'), 'bb_upper': latest.get('bb_upper'),
                    'bb_lower': latest.get('bb_lower'), 'funding_rate': funding.get('fundingRate')
                }
                indicators_snapshot = {k: v for k, v in indicators_snapshot.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
                general_ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=(gemini_api_key if gemini_api_key else None))

                current_tf_results[tf] = {
                    'score': int(score), 'label': label, 'price': float(latest['close']), 'per_scores': per_scores, 'reasons': reasons,
                    'ai_analysis': general_ai_analysis, 'volume_reversal': volume_reversal_analysis,
                    'strategy_combo': strategy_combo_analysis, 'specter_trend': specter_trend_analysis
                }

                current_confidence = general_ai_analysis.get('confidence', 0) if general_ai_analysis.get('signal') not in ['NEUTRAL', 'ERROR'] else -1
                if current_confidence > best_ai_confidence:
                    best_ai_confidence = current_confidence; best_tf = tf

            entry['details'] = current_tf_results; entry['best_timeframe'] = best_tf; entry['best_score'] = int(best_ai_confidence) if best_ai_confidence >= 0 else 0
            entry['buy_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') in ['AL', 'GÜÇLÜ AL'])
            entry['sell_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') in ['SAT', 'GÜÇLÜ SAT'])
            results.append(entry)

        except Exception as e:
            logging.error(f"Tarama sırasında {sym} için hata: {e}", exc_info=True)
            st.toast(f"{sym} taranırken hata: {e}", icon="🚨")
            continue

    try: progress_bar_area.empty()
    except Exception: pass
    if not results: logging.warning("Tarama hiç sonuç üretmedi.")
    return pd.DataFrame(results)


# ------------- Market Analysis Functions (Aynı kaldı) --------------
@st.cache_data(ttl=timedelta(minutes=30))
def get_market_analysis(api_key, period="current"):
    # ... (İçerik aynı) ...
    pass

# ------------- TradingView GÖMME FONKSİYONU (Aynı kaldı) ------------
def show_tradingview(symbol: str, interval_tv: str, height: int = 480):
    # ... (İçerik aynı) ...
    pass

# ------------------- ANA UYGULAMA AKIŞI -------------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli (Hibrit AI)")

# --- Piyasa Analizi Alanı ---
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", key="api_key_input")
if gemini_api_key_ui:
    analysis_col1, analysis_col2 = st.columns(2)
    # ... (Analiz gösterimi aynı) ...
    st.markdown("---")

# --- Sidebar Ayarları ---
st.sidebar.header("Tarama Ayarları")
all_symbols_list = fetch_all_contract_symbols() # Başlangıçta 1 kez çekilir
mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top Hacim","Özel Liste"])

symbols_to_scan_ui = []
if mode == "Özel Liste":
    selected_symbols_ui = st.sidebar.multiselect("Coinleri Seçin (Arayabilirsiniz)", options=all_symbols_list, default=["BTCUSDT", "ETHUSDT"])
    if not selected_symbols_ui:
        st.sidebar.warning("Lütfen taranacak en az bir coin seçin.")
        st.stop()
    symbols_to_scan_ui = selected_symbols_ui
else: # Top Hacim
    symbols_by_volume_list = get_top_contracts_by_volume(200)
    
    # --- DÜZELTME: Coin Listesi Fallback ---
    if not symbols_by_volume_list:
        st.sidebar.warning("Hacim verisi alınamadı. Tüm coin listesine (detay) geri dönülüyor...")
        logging.warning("get_top_contracts_by_volume() boş döndü. fetch_all_contract_symbols() deneniyor.")
        symbols_by_volume_list = fetch_all_contract_symbols() # Fallback
        if not symbols_by_volume_list:
             st.sidebar.error("Kritik: MEXC'den hiçbir sembol listesi alınamadı.")
             st.stop()
    # --- Düzeltme Sonu ---

    max_symbols = len(symbols_by_volume_list); min_val_slider = 5
    max_val_slider = max(min_val_slider, max_symbols)
    default_val_slider = max(min_val_slider, min(50, max_symbols))
    top_n_ui = st.sidebar.slider("İlk N Coin", min_value=min_val_slider, max_value=max_val_slider, value=default_val_slider)
    symbols_to_scan_ui = symbols_by_volume_list[:top_n_ui]

if not symbols_to_scan_ui: st.sidebar.error("Taranacak sembol listesi boş!"); st.stop()

timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS, key="timeframes_multiselect")
if not timeframes_ui: st.sidebar.warning("Zaman dilimi seçin."); st.stop()

# --- Diğer Ayarlar (Expander'lar - Güvenli) ---
with st.sidebar.expander("☁️ Specter Trend Ayarları"): specter_ma_type_ui=st.selectbox(...); specter_ma_length_ui=st.slider(...)
with st.sidebar.expander("📈 Hacim Dönüş Ayarları"): vr_lookback_ui=st.slider(...); vr_confirm_ui=st.slider(...); vr_vol_multi_ui=st.slider(...)
with st.sidebar.expander("💡 Strateji Komb. Ayarları"): combo_adx_thresh_ui=st.slider(...)
with st.sidebar.expander("⚙️ Algoritma Ayarları (Eski)"): weights_ui={...}; thresholds_ui=(...)

# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    spinner_msg = "Tarama çalışıyor...";
    with st.spinner(spinner_msg):
        st.session_state.scan_results = run_scan( # Doğrudan run_scan (bu dosyadaki)
            symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui,
            gemini_api_key_ui, vr_lookback_ui, vr_confirm_ui, vr_vol_multi_ui,
            combo_adx_thresh_ui, specter_ma_type_ui, specter_ma_length_ui
        )
        st.session_state.last_scan_time = datetime.now()
        st.session_state.selected_symbol = None

# --- Sonuçları Göster ---
df_results = st.session_state.scan_results
if st.session_state.last_scan_time: st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

if df_results is None or df_results.empty:
    if st.session_state.last_scan_time: st.warning("Tarama sonuç vermedi veya hata oluştu.")
    else: st.info("Başlamak için 'Tara / Yenile' butonuna basın.")
else:
    # --- Veri Hazırlama (Aynı kaldı) ---
    all_signals_list = []
    # ... (Veri listelerini doldurma mantığı aynı) ...
    all_signals_df = pd.DataFrame(all_signals_list)
    if all_signals_df.empty: st.warning("Tarama sonuçları işlenemedi."); st.stop()

    # --- Sekmeleri Oluştur ---
    tab_titles = ["📊 Genel AI", "📈 Hacim Dönüş", "💡 Strateji Komb.", "☁️ Specter Trend"]
    # ... (Sekme oluşturma aynı) ...
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    # --- Sekme İçerikleri (Aynı kaldı) ---
    with tab1: # Genel AI ...
        # ... (İçerik aynı - SyntaxError düzeltmesi dahil) ...
        pass
    with tab2: # Hacim Dönüş ...
        # ... (İçerik aynı) ...
        pass
    with tab3: # Strateji Komb. ...
        # ... (İçerik aynı) ...
        pass
    with tab4: # Specter Trend ...
        # ... (İçerik aynı) ...
        pass

    # --- Takip Edilen Sinyaller (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

    # --- Özet Metrikler ve Kayıtlı Tahminler (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
