# app.py
# Streamlit MEXC contract sinyal uygulaması - (v3.1 - Logic Moved to ai_engine)

import streamlit as st
import pandas as pd
import numpy as np
# import pandas_ta as ta # -> ai_engine'e taşındı
import requests
from datetime import datetime, timedelta
import ai_engine  # <-- TÜM MANTIK BURADA
import streamlit.components.v1 as components
import json
import logging

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

# ---------------- CONFIG & CONSTANTS ----------------
CONTRACT_BASE = "https://contract.mexc.com/api/v1"
INTERVAL_MAP = {'1m':'Min1','5m':'Min5','15m':'Min15','30m':'Min30','1h':'Min60','4h':'Hour4','1d':'Day1'}
TV_INTERVAL_MAP = {'1m':'1','5m':'5','15m':'15','30m':'30','1h':'60','4h':'240','1d':'D'}
DEFAULT_TFS = ['15m','1h','4h']
ALL_TFS = ['1m','5m','15m','30m','1h','4h','1d']
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':0,'vol':10,'funding':30,'nw':8} # ADX ağırlığı 0 yapıldı
SCALP_TFS = ['1m', '5m', '15m']
SWING_TFS = ['4h', '1d']

# CSS
st.markdown("""
<style>
/* ... (Önceki CSS stilleri aynı kalacak) ... */
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

# ---------------- API Helpers (Aynı kaldı) ----------------
# ... (fetch_all_contract_symbols, fetch_json, fetch_contract_ticker, get_top_contracts_by_volume, mexc_symbol_from, fetch_contract_klines, fetch_contract_funding_rate) ...
# Bu fonksiyonlar önceki yanıttaki gibi kalacak.

# ---------------- Scan Engine (ai_engine çağrıları güncellendi) ----------------
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key):
    """Ana tarama fonksiyonu - ai_engine'i kullanır."""
    results = []
    total_symbols = len(symbols_to_scan)
    progress_bar_area = st.sidebar.empty()
    progress_bar = progress_bar_area.progress(0, text="Tarama başlatılıyor...")

    for i, sym in enumerate(symbols_to_scan):
        progress_value = (i + 1) / total_symbols
        progress_text = f"Taranıyor: {sym} ({i+1}/{total_symbols})"
        progress_bar.progress(progress_value, text=progress_text)

        entry = {'symbol': sym, 'details': {}}
        best_ai_confidence = -1
        best_tf = None
        mexc_sym = mexc_symbol_from(sym)
        if not mexc_sym.endswith("_USDT"): continue

        funding = fetch_contract_funding_rate(mexc_sym)
        current_tf_results = {}

        for tf in timeframes:
            interval = INTERVAL_MAP.get(tf)
            if interval is None: continue

            scan_mode = "Normal"
            if tf in SCALP_TFS: scan_mode = "Scalp"
            elif tf in SWING_TFS: scan_mode = "Swing"

            df = fetch_contract_klines(mexc_sym, interval)
            # Yetersiz veri kontrolü önemli
            if df is None or df.empty or len(df) < 50:
                 logging.warning(f"run_scan: Yetersiz kline verisi ({sym} - {tf})")
                 continue
            
            # --- İNDİKATÖR HESAPLAMA ai_engine'den çağrılıyor ---
            df_ind = ai_engine.compute_indicators(df)
            if df_ind is None or df_ind.empty or len(df_ind) < 3:
                 logging.warning(f"run_scan: İndikatör hesaplama başarısız ({sym} - {tf})")
                 continue
            # --- Bitti ---

            latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]

            # --- SKORLAMA ai_engine'den çağrılıyor ---
            score, per_scores, reasons = ai_engine.score_signals(latest, prev, funding, weights)
            label = ai_engine.label_from_score(score, thresholds)
            # --- Bitti ---

            indicators_snapshot = {
                'symbol': sym, 'timeframe': tf, 'scan_mode': scan_mode,
                'score': int(score), 'price': float(latest['close']),
                # Gerekli indikatörleri güvenli şekilde al
                'rsi14': latest.get('rsi14'), 'macd_hist': latest.get('macd_hist'),
                'vol_osc': latest.get('vol_osc'), 'atr14': latest.get('atr14'),
                'nw_slope': latest.get('nw_slope'), 'bb_upper': latest.get('bb_upper'),
                'bb_lower': latest.get('bb_lower'), 'funding_rate': funding.get('fundingRate')
            }
            indicators_snapshot = {k: v for k, v in indicators_snapshot.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

            try:
                ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=(gemini_api_key if gemini_api_key else None))
            except Exception as e:
                logging.error(f"AI Analiz hatası (run_scan, {sym}, {tf}): {e}")
                st.toast(f"{sym}-{tf} AI analizi başarısız.", icon="⚠️")
                ai_analysis = {"signal": "ERROR", "confidence": 0, "explanation": f"AI Hatası: {e}"}


            current_tf_results[tf] = {
                'score': int(score), 'label': label, 'price': float(latest['close']),
                'per_scores': per_scores, 'reasons': reasons,
                'ai_analysis': ai_analysis
            }

            current_confidence = ai_analysis.get('confidence', 0) if ai_analysis.get('signal') not in ['NEUTRAL', 'ERROR'] else -1
            if current_confidence > best_ai_confidence:
                best_ai_confidence = current_confidence
                best_tf = tf

        entry['details'] = current_tf_results
        entry['best_timeframe'] = best_tf
        entry['best_score'] = int(best_ai_confidence) if best_ai_confidence >= 0 else 0
        entry['buy_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') in ['AL', 'GÜÇLÜ AL'])
        entry['strong_buy_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') == 'GÜÇLÜ AL')
        entry['sell_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') in ['SAT', 'GÜÇLÜ SAT'])
        results.append(entry)

    progress_bar_area.empty()
    return pd.DataFrame(results)

# ------------- Market Analysis Functions (Aynı kaldı) --------------
# ... (get_market_analysis fonksiyonu önceki yanıttaki gibi kalacak) ...

# ------------- TradingView GÖMME FONKSİYONU (Aynı kaldı) ------------
# ... (show_tradingview fonksiyonu önceki yanıttaki gibi kalacak) ...

# ------------------- ANA UYGULAMA AKIŞI (Aynı kaldı) -------------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli (Hibrit AI)")

# --- Piyasa Analizi Alanı ---
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", help="Gelişmiş AI analizi ve Piyasa Tahmini için.", key="api_key_input")
# ... (Piyasa analizi gösterimi aynı kaldı) ...

# --- Sidebar Ayarları ---
st.sidebar.header("Tarama Ayarları")
# ... (Sembol seçimi, Zaman Dilimleri, Algoritma Ayarları aynı kaldı) ...

# --- Session State Başlatma ---
# ... (Aynı kaldı) ...

# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    # ... (Tarama başlatma mantığı aynı kaldı) ...
    st.session_state.scan_results = run_scan(symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui, gemini_api_key_ui)
    # ...

# --- Sonuçları Göster ---
df_results = st.session_state.scan_results
# ... (Sonuçların gösterilmesi, filtreleme, detay ekranı, takip sistemi, özet metrikler, arşiv gösterimi önceki yanıttaki gibi aynı kaldı) ...

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
