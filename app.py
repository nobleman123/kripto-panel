# app.py
# Profesyonel Sinyal Paneli (v6.1 - NameError Düzeltmesi)

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import ai_engine  # Gelişmiş analiz motorumuzu import ediyoruz
import streamlit.components.v1 as components
import json
import logging
import time
from typing import Dict, List, Tuple, Any # <-- HATA DÜZELTMESİ: Eklendi

# --- Temel Ayarlar ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
st.set_page_config(page_title="MEXC Vadeli - Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# --- Plotly Kontrolü ---
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly kütüphanesi bulunamadı.")

# --- Session State Başlatma (Güvenli) ---
default_values = {
    'scan_results': pd.DataFrame(),    # Tarama sonuçları için DataFrame
    'selected_signal_data': None,  # Tıklanan sinyalin tüm verisi
    'last_scan_time': None,        # Son taramanın zamanı
    'tracked_signals': []          # Takip edilen/kaydedilen sinyaller
}
for key, default_value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ---------------- CONFIG & CONSTANTS ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d'] # 1W yok
DEFAULT_TFS_REQUESTED = ['15m','1h','4h']
DEFAULT_TFS = [tf for tf in DEFAULT_TFS_REQUESTED if tf in ALL_TFS] # Doğrulama
SCALP_TFS = ['1m', '5m', '15m']
SWING_TFS = ['4h', '1d']

# ---------------- CSS ----------------
st.markdown("""
<style>
    /* ... (CSS stilleri aynı kaldı) ... */
</style>
""", unsafe_allow_html=True)

# ---------------- API Yardımcı Fonksiyonları (Sağlamlaştırıldı) ----------------
@st.cache_data(ttl=timedelta(hours=1))
def fetch_all_contract_symbols():
    # ... (İçerik aynı kaldı) ...
    pass
def fetch_json(url, params=None, timeout=15):
    # ... (İçerik aynı kaldı) ...
    pass
@st.cache_data(ttl=timedelta(minutes=1))
def get_top_contracts_by_volume(limit=200):
    # ... (İçerik aynı kaldı) ...
    pass
def mexc_symbol_from(symbol: str) -> str:
    # ... (İçerik aynı kaldı) ...
    pass
@st.cache_data(ttl=timedelta(seconds=30))
def fetch_contract_klines(symbol_mexc, interval_mexc):
    # ... (İçerik aynı kaldı) ...
    pass

# ---------------- Scan Engine (app.py içinde) ----------------
def run_scan(symbols_to_scan, timeframes, weights, gemini_api_key):
    """
    Ana tarama fonksiyonu. ai_engine'deki analizleri çağırır.
    """
    results = []
    total_symbols = len(symbols_to_scan)
    progress_bar_area = st.sidebar.empty()
    progress_bar = progress_bar_area.progress(0, text="Tarama başlatılıyor...")

    for i, sym in enumerate(symbols_to_scan):
        progress_value = (i + 1) / total_symbols
        progress_text = f"Taranıyor: {sym} ({i+1}/{total_symbols})"
        try: progress_bar.progress(progress_value, text=progress_text)
        except: pass

        mexc_sym = mexc_symbol_from(sym)
        if not mexc_sym.endswith("_USDT"): continue

        try:
            for tf in timeframes:
                interval = INTERVAL_MAP.get(tf)
                if interval is None: continue

                scan_mode = "Normal"
                if tf in SCALP_TFS: scan_mode = "Scalp"
                elif tf in SWING_TFS: scan_mode = "Swing"

                df = fetch_contract_klines(mexc_sym, interval)
                min_bars_needed = 100
                if df is None or df.empty or len(df) < min_bars_needed:
                    logging.debug(f"Yetersiz kline ({sym}-{tf}): {len(df) if df is not None else 0}")
                    continue

                # --- 1. İndikatörleri Hesapla ---
                df_ind = ai_engine.compute_indicators(df)
                if df_ind is None or df_ind.empty or len(df_ind) < 2:
                    logging.warning(f"İndikatör hesaplanamadı: {sym}-{tf}")
                    continue

                latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]

                # --- 2. Algoritmik Puanlama ---
                algo_score, algo_label, algo_contributions = ai_engine.score_signals(latest, prev, weights, tf, SCALP_TFS)

                # --- 3. Gemini AI Analizi ---
                gemini_analysis = None
                if gemini_api_key:
                    try:
                        indicators_snapshot = {
                            'symbol': sym, 'timeframe': tf, 'scan_mode': scan_mode,
                            'price': float(latest['close']),
                            'rsi': latest.get('rsi'), 'macd_hist': latest.get('macd_hist'),
                            'ema_cross_signal': 1 if latest.get('ema_short',0) > latest.get('ema_long',0) else -1,
                            'bb_percent': latest.get('bbp'),
                            'stoch_k': latest.get('stoch_k'), 'stoch_d': latest.get('stoch_d'),
                            'adx': latest.get('adx'), 'dmi_plus': latest.get('dmi_plus'), 'dmi_minus': latest.get('dmi_minus'),
                            'volume_spike': latest.get('volume_spike'),
                            'atr_percent': latest.get('atr_percent'),
                            'algo_score': algo_score
                        }
                        indicators_snapshot = {k: (v if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in indicators_snapshot.items()}
                        gemini_analysis = ai_engine.get_gemini_analysis(indicators_snapshot, api_key=gemini_api_key)
                    except Exception as e:
                        logging.error(f"Gemini analizi hatası ({sym}-{tf}): {e}")
                        gemini_analysis = {"error": str(e)}

                # --- Sonuçları Birleştir ---
                results.append({
                    'symbol': sym, 'tf': tf,
                    'price': float(latest['close']),
                    'algo_score': algo_score,
                    'algo_label': algo_label,
                    'algo_contributions': algo_contributions,
                    'gemini_analysis': gemini_analysis,
                    'timestamp': datetime.now()
                })

        except Exception as e:
            logging.error(f"Tarama sırasında {sym} için hata: {e}", exc_info=True)
            st.toast(f"{sym} taranırken hata: {e}", icon="🚨")
            continue

    try: progress_bar_area.empty()
    except: pass
    
    if not results: logging.warning("Tarama hiç sonuç üretmedi.")
    return pd.DataFrame(results)


# ------------- Market Analysis Functions (Aynı kaldı) --------------
@st.cache_data(ttl=timedelta(minutes=30))
def get_market_analysis(api_key):
    # ... (İçerik aynı kaldı) ...
    pass

# ------------- UI Yardımcı Fonksiyonları ------------
def show_tradingview(symbol: str, interval_tv: str):
    """TradingView widget'ını güvenli bir şekilde HTML bileşeni olarak basar."""
    # ... (İçerik aynı kaldı) ...
    pass

def plot_indicator_contributions(contributions: Dict[str, float]): # <-- Hata buradaydı
    """İndikatör katkılarını Plotly bar chart olarak çizer."""
    if not PLOTLY_AVAILABLE or not contributions:
        st.caption("Puan katkı detayı yok veya Plotly yüklü değil.")
        return
    # ... (Grafik çizdirme kodu aynı kaldı) ...
    pass

# ------------------- ANA UYGULAMA AKIŞI -------------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli")

# --- 1. SOL: Ayarlar (Sidebar) ---
st.sidebar.header("Tarama Ayarları")
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", key="api_key_input")
timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS, key="timeframes_multiselect")
if not timeframes_ui: st.sidebar.warning("Lütfen en az bir zaman dilimi seçin."); st.stop()
# ... (Sembol seçimi ve hata yönetimi aynı kaldı) ...
all_symbols_list = fetch_all_contract_symbols(); mode = st.sidebar.radio("Sembol Kaynağı", ["Top Hacim","Özel Liste"], key="mode_radio")
symbols_to_scan_ui = [];
# ... (Sembol listesi oluşturma - TypeError fix dahil - aynı kaldı) ...
if not symbols_to_scan_ui: st.sidebar.error("Taranacak sembol seçilmedi!"); st.stop()
# ... (Ağırlık ayarları expander'ı aynı kaldı) ...
with st.sidebar.expander("Gelişmiş Ağırlık Ayarları"):
    weights_ui = {}
    weights_ui['rsi_weight'] = st.slider("RSI Ağırlığı", 0, 50, 25)
    # ... (diğer slider'lar) ...
scan = st.sidebar.button("🔍 Tara / Yenile")
if st.session_state.last_scan_time: st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

# --- 2. ÜST: Piyasa Analizi ---
# ... (Aynı kaldı) ...

# --- 3. ORTA ve SAĞ: Detay Alanı (Layout) ---
col_mid, col_right = st.columns([1.8, 1.2])
# ... (Yer tutucular aynı kaldı) ...

# --- 4. ALT: Tarama Sonuçları ve Geçmiş ---
tab_results, tab_history = st.tabs(["Tarama Sonuçları", "Geçmiş / Takip Edilenler"])
with tab_results:
    if scan:
        with st.spinner(f"{len(symbols_to_scan_ui)} coin taranıyor..."):
            st.session_state.scan_results = run_scan(symbols_to_scan_ui, timeframes_ui, weights_ui, gemini_api_key_ui)
            st.session_state.last_scan_time = datetime.now()
            st.session_state.selected_signal_data = None # Taramadan sonra seçimi sıfırla

    df_results = st.session_state.scan_results
    if df_results is None or df_results.empty:
        st.caption("Henüz tarama sonucu yok.")
    else:
        # ... (Filtreleme, Sıralama ve Sonuçları Listeleme mantığı aynı kaldı) ...
        # ... (Detay butonuna tıklandığında selected_signal_data'yı ayarlama) ...
        pass
with tab_history:
    # ... (Aynı kaldı) ...
    st.info("Sinyal takip sistemi yakında eklenecektir.")

# --- 5. Detay Panellerini Doldur (Aynı kaldı) ---
selected_data = st.session_state.get('selected_signal_data')
if selected_data is not None:
    with col_mid:
        # ... (Grafik gösterme) ...
        pass
    with col_right:
        # ... (Puan Metrikleri, Gemini Karşılaştırması, İndikatör Katkı Grafiği gösterme) ...
        pass
