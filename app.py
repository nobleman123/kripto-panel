# app.py
# Streamlit MEXC contract sinyal uygulaması - (v5.2 - Final SyntaxError Fix)

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

# ---------------- API Yardımcı Fonksiyonları (get_top_contracts_by_volume DÜZELTİLDİ) ----------------
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
        # Yanıtın gerçekten JSON olup olmadığını kontrol et
        content_type = r.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            return r.json()
        else:
            logging.error(f"JSON beklenirken farklı içerik tipi alındı: {content_type} - URL: {url}")
            return None # JSON değilse None döndür
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
    """Hacme göre sıralanmış sembol listesini çeker (SyntaxError düzeltildi)."""
    url = f"{CONTRACT_BASE}/contract/ticker"
    logging.info(f"Top {limit} hacimli sembol çekiliyor...")
    data = fetch_json(url) # fetch_json artık None döndürebilir
    if not data or 'data' not in data or not isinstance(data['data'], list):
        logging.error("get_top_contracts_by_volume: Geçersiz veri formatı veya API hatası.")
        return []

    # --- SyntaxError Düzeltmesi ---
    def vol(x):
        """Güvenli bir şekilde hacim verisini float'a çevirir."""
        # try/except bloğu ayrı satırlarda
        try:
            # Önce volume24'ü, sonra amount24'ü dene, yoksa 0 kullan
            volume_str = x.get('volume24') or x.get('amount24') or '0'
            return float(volume_str)
        except (ValueError, TypeError, AttributeError):
            # Hata durumunda veya veri yoksa 0 döndür
            return 0
    # --- Düzeltme Sonu ---

    valid_items = [item for item in data['data'] if isinstance(item, dict)] # Sadece sözlükleri işle
    # Hata ayıklama: Sıralama öncesi bazı hacim değerlerini logla
    # if valid_items: logging.debug(f"Örnek hacimler: {[vol(item) for item in valid_items[:5]]}")

    # Sıralama
    try:
        items = sorted(valid_items, key=vol, reverse=True)
    except Exception as e:
        logging.error(f"Hacme göre sıralama hatası: {e}")
        items = valid_items # Sıralama başarısız olursa orijinal sırayı kullan

    syms = [it.get('symbol') for it in items[:limit] if it.get('symbol')]
    result = [s.replace('_USDT','USDT') for s in syms if s.endswith('_USDT')]
    logging.info(f"{len(result)} hacimli sembol işlendi.")
    return result

def mexc_symbol_from(symbol: str) -> str: # USDT ekler
    s = symbol.strip().upper();
    if not s: return ""
    if '_' in s: return s;
    if s.endswith('USDT'): return s[:-4] + "_USDT";
    # logging.warning(f"Beklenmeyen format (mexc_symbol_from): {symbol}. USDT varsayılıyor."); # Çok fazla log üretebilir
    return s + "_USDT" # Tahminen USDT ekle

@st.cache_data(ttl=timedelta(seconds=30))
def fetch_contract_klines(symbol_mexc, interval_mexc): # Daha sağlam
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    data = fetch_json(url, params={'interval': interval_mexc})
    if not data or 'data' not in data or not isinstance(data['data'], dict):
         logging.warning(f"Geçersiz kline verisi: {symbol_mexc} - {interval_mexc}")
         return pd.DataFrame()
    d = data['data']
    # 'time' listesinin varlığını ve boş olmadığını kontrol et
    times = d.get('time')
    if not isinstance(times, list) or not times:
         logging.warning(f"Kline 'time' verisi eksik/geçersiz: {symbol_mexc} - {interval_mexc}")
         return pd.DataFrame()
    try:
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(d.get('time'), unit='s', errors='coerce'),
            'open': pd.to_numeric(d.get('open'), errors='coerce'),
            'high': pd.to_numeric(d.get('high'), errors='coerce'),
            'low': pd.to_numeric(d.get('low'), errors='coerce'),
            'close': pd.to_numeric(d.get('close'), errors='coerce'),
            'volume': pd.to_numeric(d.get('vol'), errors='coerce')
        })
        # Zaman damgası olmayan veya close olmayan satırları kaldır
        df = df.dropna(subset=['timestamp', 'close']).reset_index(drop=True)
        if len(df) < 50: logging.warning(f"fetch_klines az veri: {symbol_mexc} - {interval_mexc} ({len(df)})")
        return df
    except Exception as e:
        logging.error(f"Kline işleme hatası ({symbol_mexc}, {interval_mexc}): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=timedelta(minutes=1))
def fetch_contract_funding_rate(symbol_mexc): # Daha sağlam
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"
    data = fetch_json(url)
    if not data or 'data' not in data or not isinstance(data['data'], dict): return {'fundingRate': 0.0}
    try: return {'fundingRate': float(data['data'].get('fundingRate') or 0)}
    except (ValueError, TypeError): return {'fundingRate': 0.0}


# ---------------- Scan Engine Wrapper (Hata Yönetimi ile - Aynı kaldı) ----------------
def run_scan_safe(*args, **kwargs):
    """run_scan fonksiyonunu çağırır ve genel hataları yakalar."""
    try:
        scan_start_time = time.time()
        # ai_engine.run_scan DEĞİL, bu dosyadaki run_scan çağrılacak
        results_df = run_scan(*args, **kwargs) # AttributeError fix: Call local run_scan
        scan_duration = time.time() - scan_start_time
        logging.info(f"Tarama tamamlandı. Süre: {scan_duration:.2f}s. {len(results_df)} sonuç.")
        return results_df
    except Exception as e:
        logging.error(f"Beklenmedik tarama hatası (run_scan_safe): {e}", exc_info=True)
        st.error(f"Tarama sırasında kritik bir hata oluştu: {e}")
        return pd.DataFrame() # Hata durumunda boş DataFrame döndür

# ---------------- Scan Engine (app.py içinde tanımlı - Önceki gibi) ----------------
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key,
             vr_lookback, vr_confirm, vr_vol_multi, combo_adx_thresh,
             specter_ma_type, specter_ma_length):
    # ... (Fonksiyonun tüm içeriği önceki yanıttaki gibi, SyntaxError düzeltmesi dahil) ...
    results = []
    total_symbols = len(symbols_to_scan)
    progress_bar_area = st.sidebar.empty()
    progress_bar = progress_bar_area.progress(0, text="Tarama başlatılıyor...")
    # ... (Döngüler, API çağrıları, ai_engine fonksiyon çağrıları...)
    progress_bar_area.empty()
    if not results: logging.warning("Tarama hiç sonuç üretmedi.")
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
symbols_to_scan_ui = [];
# ... (Sembol listesi oluşturma - TypeError fix dahil - aynı kaldı) ...
if not symbols_to_scan_ui: st.sidebar.error("Taranacak sembol seçilmedi veya alınamadı!"); st.stop()
timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS, key="timeframes_multiselect")
if not timeframes_ui: st.sidebar.warning("Zaman dilimi seçin."); st.stop()
# ... (Specter, Hacim, Strateji, Algoritma ayarları expander'ları aynı kaldı) ...
with st.sidebar.expander("☁️ Specter Trend Ayarları"): specter_ma_type_ui=st.selectbox(...); specter_ma_length_ui=st.slider(...)
# ... (diğer expanderlar) ...

# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    spinner_msg = "Tarama çalışıyor...";
    with st.spinner(spinner_msg):
        # run_scan_safe yerine doğrudan run_scan çağırıyoruz
        st.session_state.scan_results = run_scan( # run_scan_safe kaldırıldı
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
    # ... (Boş sonuç mesajı aynı kaldı) ...
    pass
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
    # Her sekme içeriği önceki yanıttaki gibi, ilgili DataFrame'i filtreleyip gösterir.
    # Detay butonları st.session_state.selected_symbol/tf/active_tab'ı günceller.
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
