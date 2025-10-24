# app.py
# Streamlit MEXC contract sinyal uygulaması - (v4.0 - Base App + Tabs + Fixes)

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import ai_engine # Gelişmiş motor
import streamlit.components.v1 as components
import json
import logging
import time
import math # Güvenli min/max için

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Plotly kontrolü
try: import plotly.express as px; PLOTLY_AVAILABLE = True
except ImportError: PLOTLY_AVAILABLE = False; logging.warning("Plotly yok.")

st.set_page_config(page_title="MEXC Vadeli - Profesyonel Signal Paneli", layout="wide", initial_sidebar_state="collapsed")

# --- Session State Başlatma (Güvenli) ---
default_values = {
    'scan_results': pd.DataFrame(),
    'selected_symbol': None,
    'selected_tf': '15m',
    'tracked_signals': {},
    'last_scan_time': None,
    'active_tab': "📊 Genel AI"
}
for key, default_value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# ---------------- CONFIG & CONSTANTS ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']
DEFAULT_TFS_REQUESTED = ['15m','1h','4h']
DEFAULT_TFS = [tf for tf in DEFAULT_TFS_REQUESTED if tf in ALL_TFS] # Doğrulama
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':0,'vol':10,'funding':30,'nw':8}
SCALP_TFS = ['1m', '5m', '15m']; SWING_TFS = ['4h', '1d']
EMA_TREND_LENGTH = ai_engine.EMA_TREND_LENGTH
SPECTER_ATR_LENGTH = ai_engine.SPECTER_ATR_LENGTH
MA_TYPES = ['EMA', 'SMA', 'SMMA', 'WMA', 'VWMA']
MAX_SIGNALS_TO_SHOW = 150 # Liste başına max sinyal

# CSS
# ... (Aynı kaldı) ...
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

# ---------------- API Helpers (Aynı kaldı) ----------------
# ... (fetch_all_contract_symbols, fetch_json, get_top_contracts_by_volume, mexc_symbol_from, fetch_contract_klines, fetch_contract_funding_rate) ...
@st.cache_data(ttl=timedelta(hours=1))
def fetch_all_contract_symbols(): #...
    pass
def fetch_json(url, params=None, timeout=15): #...
    pass
@st.cache_data(ttl=timedelta(minutes=1))
def get_top_contracts_by_volume(limit=200): #...
    pass
def mexc_symbol_from(symbol: str) -> str: #...
    pass
@st.cache_data(ttl=timedelta(seconds=30))
def fetch_contract_klines(symbol_mexc, interval_mexc): #...
    pass
@st.cache_data(ttl=timedelta(minutes=1))
def fetch_contract_funding_rate(symbol_mexc): #...
    pass

# ---------------- Scan Engine (Aynı kaldı - ai_engine'deki tüm analizleri çağırır) ----------------
# run_scan fonksiyonu önceki yanıttaki gibi kalacak
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key,
             vr_lookback, vr_confirm, vr_vol_multi, combo_adx_thresh,
             specter_ma_type, specter_ma_length):
    # ... (Fonksiyonun tüm içeriği önceki yanıttaki gibi) ...
    results = []
    # ... (Progress bar, döngüler, API çağrıları, ai_engine'deki tüm analiz fonksiyon çağrıları) ...
    # İçinde df_ind = ai_engine.compute_indicators(...) çağrısı var
    # İçinde score, per_scores, reasons = ai_engine.score_signals(...) çağrısı var
    # İçinde volume_reversal_analysis = ai_engine.analyze_volume_reversal(...) çağrısı var
    # İçinde strategy_combo_analysis = ai_engine.analyze_strategy_combo(...) çağrısı var
    # İçinde specter_trend_analysis = ai_engine.analyze_specter_trend(...) çağrısı var
    # İçinde general_ai_analysis = ai_engine.get_ai_prediction(...) çağrısı var
    # Sonuçları 'details' altında birleştirir: 'ai_analysis', 'volume_reversal', 'strategy_combo', 'specter_trend'
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
if gemini_api_key_ui:
    analysis_col1, analysis_col2 = st.columns(2)
    # ... (Analizleri al ve göster) ...
    st.markdown("---")


# --- Sidebar Ayarları ---
st.sidebar.header("Tarama Ayarları")
all_symbols_list = fetch_all_contract_symbols(); mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top Hacim","Özel Liste"])
symbols_to_scan_ui = [];
if mode == "Özel Liste": selected_symbols_ui = st.sidebar.multiselect("Coinleri Seçin", options=all_symbols_list, default=["BTCUSDT", "ETHUSDT"]); symbols_to_scan_ui = selected_symbols_ui
else: # Top Hacim
    symbols_by_volume_list = get_top_contracts_by_volume(200)
    if not symbols_by_volume_list: # Hata kontrolü
        st.sidebar.error("MEXC hacim verisi alınamadı."); st.stop()
    else:
        # Güvenli slider değerleri
        max_symbols = len(symbols_by_volume_list); min_val_slider = 5; max_val_slider = max(min_val_slider, max_symbols)
        # default_val_slider = max(min_val_slider, min(50, max_symbols)) # Orijinal
        # Önceki hatanın kök nedeni min() kullanımı olabilir, math.min ile deneyelim veya daha basit yapalım
        default_val_slider = 50 if max_symbols >= 50 else max(min_val_slider, max_symbols) # 50 veya max (hangisi küçükse, ama en az 5)

        top_n_ui = st.sidebar.slider("İlk N Coin", min_value=min_val_slider, max_value=max_val_slider, value=default_val_slider)
        symbols_to_scan_ui = symbols_by_volume_list[:top_n_ui]

if not symbols_to_scan_ui: st.sidebar.warning("Taranacak sembol seçilmedi."); st.stop()

# Zaman dilimleri (Hata düzeltmesi zaten uygulanmıştı)
timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS, key="timeframes_multiselect")
if not timeframes_ui: st.sidebar.warning("Zaman dilimi seçin."); st.stop()

# --- Diğer Ayarlar (Expander'lar) ---
with st.sidebar.expander("☁️ Specter Trend Ayarları"): specter_ma_type_ui=st.selectbox("MA Tipi", MA_TYPES, index=0); specter_ma_length_ui=st.slider("Kısa MA Periyodu", 5, 100, 21)
with st.sidebar.expander("📈 Hacim Dönüş Ayarları"): vr_lookback_ui=st.slider("Anchor Mum P.", 5, 50, 20); vr_confirm_ui=st.slider("Onay P.", 1, 10, 5); vr_vol_multi_ui=st.slider("Hacim Çarpanı", 1.1, 3.0, 1.5, 0.1)
with st.sidebar.expander("💡 Strateji Komb. Ayarları"): combo_adx_thresh_ui=st.slider("Minimum ADX Gücü", 10, 40, 20)
with st.sidebar.expander("⚙️ Algoritma Ayarları (Eski)"): weights_ui={...}; thresholds_ui=(...) # Inputlar aynı

# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    spinner_msg = "Tarama çalışıyor...";
    with st.spinner(spinner_msg):
        try:
             scan_start_time = time.time()
             st.session_state.scan_results = run_scan(
                 symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui,
                 gemini_api_key_ui, vr_lookback_ui, vr_confirm_ui, vr_vol_multi_ui,
                 combo_adx_thresh_ui, specter_ma_type_ui, specter_ma_length_ui
             )
             scan_duration = time.time() - scan_start_time
             logging.info(f"Tarama tamamlandı. Süre: {scan_duration:.2f}s. {len(st.session_state.scan_results)} sonuç.")
             st.session_state.last_scan_time = datetime.now()
             st.session_state.selected_symbol = None # Seçimi sıfırla
             # rerun() KULLANMA, state zaten güncellendi ve widget'lar yeniden çizilecek
        except Exception as e:
             logging.error(f"Tarama hatası (ana blok): {e}", exc_info=True)
             st.error(f"Tarama sırasında hata oluştu. Detaylar loglarda.")
             st.session_state.scan_results = pd.DataFrame()


# --- Sonuçları Göster ---
df_results = st.session_state.scan_results # State'den al (Boş olabilir)

if st.session_state.last_scan_time: st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

if df_results is None or df_results.empty:
    if st.session_state.last_scan_time: st.warning("Tarama sonuç vermedi veya hata oluştu.")
    else: st.info("Tara / Yenile'ye basın.")
else:
    # --- Veri Hazırlama (Tüm Analiz Türleri İçin) ---
    all_signals_list = [] # Tek liste kullanıp sonra filtreleyelim
    for _, row in df_results.iterrows():
        symbol = row['symbol']; details = row.get('details', {})
        for tf, tf_data in details.items():
            if not tf_data: continue
            # Tüm analizleri tek bir kayda ekle
            record = {
                'symbol': symbol, 'tf': tf, 'price': tf_data.get('price'),
                'ai_analysis': tf_data.get('ai_analysis'),
                'volume_reversal': tf_data.get('volume_reversal'),
                'strategy_combo': tf_data.get('strategy_combo'),
                'specter_trend': tf_data.get('specter_trend'),
                'algo_score': tf_data.get('score'), 'algo_label': tf_data.get('label'),
                'per_scores': tf_data.get('per_scores')
            }
            all_signals_list.append(record)

    if not all_signals_list:
        st.warning("Tarama sonuçları işlenemedi.")
        st.stop()

    all_signals_df = pd.DataFrame(all_signals_list)

    # --- Sekmeleri Oluştur ---
    tab_titles = ["📊 Genel AI", "📈 Hacim Dönüş", "💡 Strateji Komb.", "☁️ Specter Trend"]
    active_tab_key = st.session_state.get('active_tab', tab_titles[0])
    try: default_tab_index = tab_titles.index(active_tab_key)
    except ValueError: default_tab_index = 0

    # st.tabs'ı st.session_state.active_tab'a göre kontrol etmiyoruz, seçimi state'e yazıyoruz
    selected_tab = st.tabs(tab_titles) # Bu fonksiyon sekmeleri oluşturur ve seçilenin index'ini döndürmez ama sekmeleri render eder

    # Hangi sekmenin aktif olduğunu UI'dan almak yerine state'i kullan
    # Kullanıcı bir sekmeye tıkladığında state'i güncellemek için bir yol bulmamız gerekebilir,
    # ancak şimdilik detay butonları state'i güncelleyecek.

    # --- Sekme 1: Genel AI Sinyalleri ---
    with selected_tab[0]: # index ile erişim
        st.session_state.active_tab = tab_titles[0] # State'i güncelle
        left1, right1 = st.columns([1.6, 2.4])
        with left1:
            st.markdown("### 🔎 Genel AI Sinyal Listesi")
            filter_signal_gen = st.selectbox("Sinyal Türü", ["All","LONG","SHORT","NEUTRAL", "ERROR"], index=0, key="gen_signal_filter")
            min_confidence_gen = st.slider("Min Güven (%)", 0, 100, 30, step=5, key="gen_conf_filter")

            # Filtreleme (ai_analysis sütunu var mı diye kontrol et)
            filtered_gen = all_signals_df[all_signals_df['ai_analysis'].notna()].copy()
            if not filtered_gen.empty:
                 # ai_analysis içinden değerleri çıkar
                 filtered_gen['ai_signal'] = filtered_gen['ai_analysis'].apply(lambda x: x.get('signal', 'N/A') if isinstance(x, dict) else 'N/A')
                 filtered_gen['ai_confidence'] = filtered_gen['ai_analysis'].apply(lambda x: x.get('confidence', 0) if isinstance(x, dict) else 0)

                 if filter_signal_gen != "All": filtered_gen = filtered_gen[filtered_gen['ai_signal'] == filter_signal_gen]
                 filtered_gen = filtered_gen[filtered_gen['ai_confidence'] >= min_confidence_gen]
                 filtered_gen = filtered_gen.sort_values(by='ai_confidence', ascending=False)

            st.caption(f"{len(filtered_gen)} sinyal bulundu.")
            # Liste gösterimi
            for _, r in filtered_gen.head(MAX_SIGNALS_TO_SHOW).iterrows():
                 emoji="⚪"; if r['ai_signal']=='LONG': emoji='🚀'; elif r['ai_signal']=='SHORT': emoji='🔻'; elif r['ai_signal']=='ERROR': emoji='⚠️'
                 cols=st.columns([0.6,2,1]); cols[0].markdown(f"<div>{emoji}</div>", unsafe_allow_html=True)
                 algo_info = f"Algo: {r.get('algo_label','N/A')} ({r.get('algo_score','N/A')})"
                 cols[1].markdown(f"**{r['symbol']}** • {r['tf']} \nAI: **{r['ai_signal']}** (%{r['ai_confidence']}) <span ...>{algo_info}</span>", unsafe_allow_html=True)
                 if cols[2].button("Detay", key=f"det_gen_{r['symbol']}_{r['tf']}"):
                      st.session_state.selected_symbol = r['symbol']; st.session_state.selected_tf = r['tf']
                      st.session_state.active_tab = tab_titles[0] # Doğru sekmeyi ayarla
                      st.experimental_rerun()

        with right1: # Detay Paneli (Genel AI)
            # ... (Detay paneli mantığı önceki gibi, row_data'yı all_signals_df'ten bul) ...
            pass # İçerik aynı

    # --- Sekme 2: Hacim Dönüş ---
    with selected_tab[1]:
        st.session_state.active_tab = tab_titles[1]
        left2, right2 = st.columns([1.6, 2.4])
        with left2:
             st.markdown("### 📈 Hacim Teyitli Dönüş Sinyalleri")
             min_score_vr = st.slider("Min Skor (1-4)", 1, 4, 2, key="vr_score_filter")

             # Filtreleme (volume_reversal ve signal kontrolü)
             filtered_vr = all_signals_df[all_signals_df['volume_reversal'].notna()].copy()
             filtered_vr = filtered_vr[filtered_vr['volume_reversal'].apply(lambda x: isinstance(x, dict) and x.get('signal') != 'NONE')]
             if not filtered_vr.empty:
                  filtered_vr['vr_signal'] = filtered_vr['volume_reversal'].apply(lambda x: x.get('signal'))
                  filtered_vr['vr_score'] = filtered_vr['volume_reversal'].apply(lambda x: x.get('score', 0))
                  filtered_vr = filtered_vr[filtered_vr['vr_score'] >= min_score_vr]
                  filtered_vr = filtered_vr.sort_values(by='vr_score', ascending=False) # Skora göre sırala

             st.caption(f"{len(filtered_vr)} sinyal bulundu.")
             # Liste gösterimi
             for _, r in filtered_vr.head(MAX_SIGNALS_TO_SHOW).iterrows():
                  emoji = '🔼' if r['vr_signal'] == 'BUY' else '🔽'
                  cols=st.columns([0.6,2,1]); cols[0].markdown(f"<div>{emoji}</div>", unsafe_allow_html=True)
                  cols[1].markdown(f"**{r['symbol']}** • {r['tf']} \nSinyal: **{r['vr_signal']}** (Skor: {r['vr_score']}/4)")
                  if cols[2].button("Detay", key=f"det_vr_{r['symbol']}_{r['tf']}"):
                       st.session_state.selected_symbol = r['symbol']; st.session_state.selected_tf = r['tf']
                       st.session_state.active_tab = tab_titles[1]
                       st.experimental_rerun()

        with right2: # Detay Paneli (Hacim Dönüş)
             # ... (Detay paneli mantığı önceki gibi, row_data'yı all_signals_df'ten bul, hacim detaylarını ve AI yorumunu göster) ...
            pass # İçerik aynı

    # --- Sekme 3: Strateji Komb. ---
    with selected_tab[2]:
        st.session_state.active_tab = tab_titles[2]
        left3, right3 = st.columns([1.6, 2.4])
        with left3:
             st.markdown("### 💡 Strateji Kombinasyon Sinyalleri")
             # Filtreleme (strategy_combo ve signal kontrolü)
             filtered_combo = all_signals_df[all_signals_df['strategy_combo'].notna()].copy()
             filtered_combo = filtered_combo[filtered_combo['strategy_combo'].apply(lambda x: isinstance(x, dict) and x.get('signal') != 'NONE')]
             if not filtered_combo.empty:
                 filtered_combo['combo_signal'] = filtered_combo['strategy_combo'].apply(lambda x: x.get('signal'))
                 # Güven %100 olduğu için filtre yok, sıralama eklenebilir (örn: zamana göre)

             st.caption(f"{len(filtered_combo)} sinyal bulundu.")
             # Liste gösterimi
             for _, r in filtered_combo.head(MAX_SIGNALS_TO_SHOW).iterrows():
                  emoji = '🟩' if r['combo_signal'] == 'BUY' else '🟥'
                  confirmations = r['strategy_combo'].get('confirming_indicators', [])
                  cols=st.columns([0.6,2,1]); cols[0].markdown(f"<div>{emoji}</div>", unsafe_allow_html=True)
                  cols[1].markdown(f"**{r['symbol']}** • {r['tf']} \nSinyal: **{r['combo_signal']}** ({len(confirmations)} Onay)")
                  if cols[2].button("Detay", key=f"det_combo_{r['symbol']}_{r['tf']}"):
                       st.session_state.selected_symbol = r['symbol']; st.session_state.selected_tf = r['tf']
                       st.session_state.active_tab = tab_titles[2]
                       st.experimental_rerun()

        with right3: # Detay Paneli (Strateji Komb.)
             # ... (Detay paneli mantığı önceki gibi, row_data'yı all_signals_df'ten bul, onayları ve AI yorumunu göster) ...
             pass # İçerik aynı

    # --- Sekme 4: Specter Trend ---
    with selected_tab[3]:
        st.session_state.active_tab = tab_titles[3]
        left4, right4 = st.columns([1.6, 2.4])
        with left4:
             st.markdown("### ☁️ Specter Trend & Retest")
             filter_trend_specter = st.selectbox("Trend Yönü", ["Tümü", "BULLISH", "BEARISH"], index=0, key="specter_trend_filter")
             filter_retest_specter = st.checkbox("Sadece Retest Sinyallerini Göster", key="specter_retest_filter")

             # Filtreleme (specter_trend kontrolü)
             filtered_specter = all_signals_df[all_signals_df['specter_trend'].notna()].copy()
             if not filtered_specter.empty:
                  filtered_specter['specter_trend_val'] = filtered_specter['specter_trend'].apply(lambda x: x.get('trend', 'N/A') if isinstance(x, dict) else 'N/A')
                  filtered_specter['specter_retest'] = filtered_specter['specter_trend'].apply(lambda x: x.get('retest_signal', 'NONE') if isinstance(x, dict) else 'NONE')

                  if filter_trend_specter != "Tümü": filtered_specter = filtered_specter[filtered_specter['specter_trend_val'] == filter_trend_specter]
                  if filter_retest_specter: filtered_specter = filtered_specter[filtered_specter['specter_retest'] != 'NONE']
                  # Sıralama (önce retest olanlar, sonra trende göre?)
                  filtered_specter = filtered_specter.sort_values(by=['specter_retest', 'symbol'], ascending=[False, True])


             st.caption(f"{len(filtered_specter)} durum bulundu.")
             # Liste gösterimi
             for _, r in filtered_specter.head(MAX_SIGNALS_TO_SHOW).iterrows():
                  trend_color = "🟢" if r['specter_trend_val'] == 'BULLISH' else ("🟠" if r['specter_trend_val'] == 'BEARISH' else "⚪")
                  retest_icon = "💎" if r['specter_retest'] != 'NONE' else ""
                  cols=st.columns([0.6,2,1]); cols[0].markdown(f"<div>{trend_color}{retest_icon}</div>", unsafe_allow_html=True)
                  retest_info = f" **{r['specter_retest']} Retest!**" if retest_icon else ""
                  cols[1].markdown(f"**{r['symbol']}** • {r['tf']} \nTrend: **{r['specter_trend_val']}**{retest_info}")
                  if cols[2].button("Detay", key=f"det_specter_{r['symbol']}_{r['tf']}"):
                       st.session_state.selected_symbol = r['symbol']; st.session_state.selected_tf = r['tf']
                       st.session_state.active_tab = tab_titles[3]
                       st.experimental_rerun()

        with right4: # Detay Paneli (Specter Trend)
            # ... (Detay paneli mantığı önceki gibi, row_data'yı all_signals_df'ten bul, specter detaylarını ve AI yorumunu göster) ...
            pass # İçerik aynı


    # --- Takip Edilen Sinyaller (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

    # --- Özet Metrikler ve Kayıtlı Tahminler (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
