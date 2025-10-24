# app.py
# Streamlit MEXC contract sinyal uygulaması - (v3.9 - TypeError Fix & Robustness)

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

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Plotly kontrolü
try: import plotly.express as px; PLOTLY_AVAILABLE = True
except ImportError: PLOTLY_AVAILABLE = False; logging.warning("Plotly yok.")

st.set_page_config(page_title="MEXC Vadeli - Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# --- Session State Başlatma ---
# ... (Aynı kaldı) ...
if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'selected_symbol' not in st.session_state: st.session_state.selected_symbol = None
if 'selected_tf' not in st.session_state: st.session_state.selected_tf = '15m'
if 'tracked_signals' not in st.session_state: st.session_state.tracked_signals = {}
if 'last_scan_time' not in st.session_state: st.session_state.last_scan_time = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = "📊 Genel AI"

# ---------------- CONFIG & CONSTANTS ----------------
# ... (Aynı kaldı) ...
CONTRACT_BASE = "https://contract.mexc.com/api/v1"; INTERVAL_MAP = {...}; TV_INTERVAL_MAP = {...}; ALL_TFS = [...]; DEFAULT_TFS_REQUESTED = ['15m','1h','4h']; DEFAULT_TFS = [tf for tf in DEFAULT_TFS_REQUESTED if tf in ALL_TFS]; DEFAULT_WEIGHTS = {...}; SCALP_TFS = [...]; SWING_TFS = [...]; EMA_TREND_LENGTH = ai_engine.EMA_TREND_LENGTH; SPECTER_ATR_LENGTH = ai_engine.SPECTER_ATR_LENGTH; MA_TYPES = [...]

# CSS
# ... (Aynı kaldı) ...
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

# ---------------- API Helpers (Aynı kaldı) ----------------
# ... (fetch_all_contract_symbols, fetch_json, get_top_contracts_by_volume, mexc_symbol_from, fetch_contract_klines, fetch_contract_funding_rate - Aynı kaldı) ...
@st.cache_data(ttl=timedelta(hours=1))
def fetch_all_contract_symbols(): #...
    pass
def fetch_json(url, params=None, timeout=15): #...
    pass
@st.cache_data(ttl=timedelta(minutes=1))
def get_top_contracts_by_volume(limit=200): #... (içindeki vol fonksiyonu düzeltilmişti)
    pass
def mexc_symbol_from(symbol: str) -> str: #...
    pass
@st.cache_data(ttl=timedelta(seconds=30))
def fetch_contract_klines(symbol_mexc, interval_mexc): #...
    pass
@st.cache_data(ttl=timedelta(minutes=1))
def fetch_contract_funding_rate(symbol_mexc): #...
    pass

# ---------------- Scan Engine (Aynı kaldı) ----------------
# run_scan fonksiyonu önceki yanıttaki gibi kalacak
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key,
             vr_lookback, vr_confirm, vr_vol_multi, combo_adx_thresh,
             specter_ma_type, specter_ma_length):
    # ... (Fonksiyonun tüm içeriği önceki yanıttaki gibi) ...
    results = []
    # ... (Progress bar, döngüler, API çağrıları, analiz fonksiyon çağrıları) ...
    return pd.DataFrame(results)

# ------------- Market Analysis Functions (Aynı kaldı) --------------
@st.cache_data(ttl=timedelta(minutes=30))
def get_market_analysis(api_key, period="current"): # ... (İçerik aynı) ...
    pass

# ------------- TradingView GÖMME FONKSİYONU (Aynı kaldı) ------------
def show_tradingview(symbol: str, interval_tv: str, height: int = 480): # ... (İçerik aynı) ...
    pass

# ------------------- ANA UYGULAMA AKIŞI -------------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli (Hibrit AI)")

# --- Piyasa Analizi Alanı ---
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", key="api_key_input")
# ... (Piyasa analizi gösterimi aynı kaldı) ...

# --- Sidebar Ayarları ---
st.sidebar.header("Tarama Ayarları")
all_symbols_list = fetch_all_contract_symbols()
mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top Hacim","Özel Liste"])

symbols_to_scan_ui = []
if mode == "Özel Liste":
    selected_symbols_ui = st.sidebar.multiselect("Coinleri Seçin", options=all_symbols_list, default=["BTCUSDT", "ETHUSDT"])
    symbols_to_scan_ui = selected_symbols_ui
else: # Top Hacim
    symbols_by_volume_list = get_top_contracts_by_volume(200)
    # --- TypeError Düzeltmesi ---
    if not symbols_by_volume_list:
        st.sidebar.error("MEXC'den hacim verisi alınamadı veya hiç vadeli işlem yok.")
        logging.error("get_top_contracts_by_volume boş liste döndürdü.")
        st.stop() # Hata durumunda durdur
    else:
        # Slider değerlerini güvenli hale getir
        max_symbols = len(symbols_by_volume_list)
        min_val_slider = 5
        max_val_slider = max(min_val_slider, max_symbols) # max_value en az min_value olmalı
        default_val_slider = max(min_val_slider, min(50, max_symbols)) # value, [min_value, max_value] arasında kalmalı

        top_n_ui = st.sidebar.slider(
            "İlk N Coin",
            min_value=min_val_slider,
            max_value=max_val_slider, # Güvenli max değer
            value=default_val_slider # Güvenli varsayılan değer
        )
        symbols_to_scan_ui = symbols_by_volume_list[:top_n_ui]
    # --- Düzeltme Sonu ---


if not symbols_to_scan_ui: st.sidebar.warning("Taranacak sembol seçilmedi."); st.stop() # Tekrar kontrol

timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS, key="timeframes_multiselect")
if not timeframes_ui: st.sidebar.warning("Zaman dilimi seçin."); st.stop()

# ... (Specter, Hacim, Strateji, Algoritma ayarları expander'ları aynı kaldı) ...
with st.sidebar.expander("☁️ Specter Trend Ayarları"): specter_ma_type_ui=st.selectbox(...); specter_ma_length_ui=st.slider(...)
with st.sidebar.expander("📈 Hacim Teyitli Dönüş Ayarları"): vr_lookback_ui=st.slider(...); vr_confirm_ui=st.slider(...); vr_vol_multi_ui=st.slider(...)
with st.sidebar.expander("💡 Strateji Kombinasyon Ayarları"): combo_adx_thresh_ui=st.slider(...)
with st.sidebar.expander("⚙️ Sistem Algoritması Ayarları (Eski)"): weights_ui={...}; thresholds_ui=(...)


# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    # ... (Tarama başlatma mantığı ve try/except aynı kaldı) ...
    spinner_msg = "Tarama çalışıyor...";
    with st.spinner(spinner_msg):
        try:
             scan_start_time = time.time()
             # Önceki tarama sonuçlarını temizle (isteğe bağlı)
             # st.session_state.scan_results = pd.DataFrame()
             st.session_state.scan_results = run_scan(
                 symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui,
                 gemini_api_key_ui, vr_lookback_ui, vr_confirm_ui, vr_vol_multi_ui,
                 combo_adx_thresh_ui, specter_ma_type_ui, specter_ma_length_ui
             )
             scan_duration = time.time() - scan_start_time
             logging.info(f"Tarama tamamlandı. Süre: {scan_duration:.2f}s. {len(st.session_state.scan_results)} sonuç.")
             st.session_state.last_scan_time = datetime.now()
             st.session_state.selected_symbol = None
             # st.experimental_rerun() # Sayfayı yenilemek bazen sorun çıkarabilir, kaldırdık. State zaten güncellendi.
        except Exception as e:
             logging.error(f"Beklenmedik tarama hatası (ana blok): {e}", exc_info=True)
             st.error(f"Tarama sırasında bir hata oluştu. Detaylar için logları kontrol edin.")
             # st.exception(e) # Kullanıcıya tam hatayı gösterme
             st.session_state.scan_results = pd.DataFrame() # Hata durumunda state'i boşalt


# --- Sonuçları Göster ---
df_results = st.session_state.scan_results # State'den al

if st.session_state.last_scan_time: st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

if df_results is None or df_results.empty:
    if st.session_state.last_scan_time: # Eğer tarama yapıldıysa ama sonuç yoksa
        st.warning("Tarama tamamlandı ancak seçili kriterlere uygun coin bulunamadı veya verilerde sorun oluştu.")
    else: # Henüz tarama yapılmadıysa
        st.info("Henüz tarama yapılmadı. Lütfen yan panelden ayarları yapılandırıp 'Tara / Yenile' butonuna basın.")
else:
    # --- Veri Hazırlama (Aynı kaldı) ---
    general_ai_list = []; volume_reversal_list = []; strategy_combo_list = []; specter_trend_list = []
    # ... (Veri listelerini doldurma mantığı aynı) ...
    general_ai_df = pd.DataFrame(general_ai_list); #... (diğer df'ler)

    # --- Sekmeleri Oluştur (Aynı kaldı) ---
    tab_titles = ["📊 Genel AI", "📈 Hacim Dönüş", "💡 Strateji Komb.", "☁️ Specter Trend"]
    # ... (Sekme oluşturma mantığı aynı) ...
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    # --- Sekme İçerikleri (Aynı kaldı) ---
    with tab1: # Genel AI ...
        # ... (Filtreleme, Liste, Detay) ...
        pass
    with tab2: # Hacim Dönüş ...
        # ... (Filtreleme, Liste, Detay) ...
        pass
    with tab3: # Strateji Komb. ...
        # ... (Filtreleme, Liste, Detay) ...
        pass
    with tab4: # Specter Trend ...
        # ... (Filtreleme, Liste, Detay) ...
        pass

    # --- Takip Edilen Sinyaller (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

    # --- Özet Metrikler ve Kayıtlı Tahminler (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
