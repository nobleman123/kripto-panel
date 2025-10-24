# app.py
# Streamlit MEXC contract sinyal uygulaması - (v3.5 - Strateji Sekmesi)

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
import time

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Plotly kontrolü
# ... (Aynı kaldı) ...
try: import plotly.express as px; PLOTLY_AVAILABLE = True
except ImportError: PLOTLY_AVAILABLE = False; logging.warning("Plotly yok.")

st.set_page_config(page_title="MEXC Vadeli - Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# --- Session State Başlatma ---
# ... (Aynı kaldı, active_tab eklendi) ...
if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'selected_symbol' not in st.session_state: st.session_state.selected_symbol = None
if 'selected_tf' not in st.session_state: st.session_state.selected_tf = '15m'
if 'tracked_signals' not in st.session_state: st.session_state.tracked_signals = {}
if 'last_scan_time' not in st.session_state: st.session_state.last_scan_time = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = "Genel AI Sinyalleri"


# ---------------- CONFIG & CONSTANTS ----------------
# ... (Aynı kaldı) ...
CONTRACT_BASE = "https://contract.mexc.com/api/v1"; INTERVAL_MAP = {...}; TV_INTERVAL_MAP = {...}; DEFAULT_TFS = ['15m','1h','4h']; ALL_TFS = [...]; DEFAULT_WEIGHTS = {...}; SCALP_TFS = [...]; SWING_TFS = [...]
EMA_TREND_LENGTH = ai_engine.EMA_TREND_LENGTH

# CSS
# ... (Aynı kaldı) ...
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

# ---------------- API Helpers (Aynı kaldı) ----------------
# ... (fetch_all_contract_symbols, fetch_json, get_top_contracts_by_volume, mexc_symbol_from, fetch_contract_klines, fetch_contract_funding_rate - Aynı kaldı) ...

# ---------------- Scan Engine (Strateji Analizi çağrısı eklendi) ----------------
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key, vr_lookback, vr_confirm, vr_vol_multi, combo_adx_thresh):
    """Ana tarama fonksiyonu - Genel AI, Hacim Dönüşü ve Strateji Kombinasyonunu analiz eder."""
    results = []
    total_symbols = len(symbols_to_scan)
    progress_bar_area = st.sidebar.empty()
    progress_bar = progress_bar_area.progress(0, text="Tarama başlatılıyor...")

    for i, sym in enumerate(symbols_to_scan):
        # ... (Progress bar, entry tanımı, funding çekme aynı kaldı) ...
        progress_value = (i + 1) / total_symbols; progress_text = f"Taranıyor: {sym} ({i+1}/{total_symbols})"
        progress_bar.progress(progress_value, text=progress_text)
        entry = {'symbol': sym, 'details': {}}; best_ai_confidence = -1; best_tf = None
        mexc_sym = mexc_symbol_from(sym);
        if not mexc_sym.endswith("_USDT"): continue
        try:
            funding = fetch_contract_funding_rate(mexc_sym); current_tf_results = {}
            for tf in timeframes:
                # ... (Interval, scan_mode, kline çekme aynı kaldı) ...
                interval = INTERVAL_MAP.get(tf); scan_mode = "Normal"
                if tf in SCALP_TFS: scan_mode = "Scalp"; elif tf in SWING_TFS: scan_mode = "Swing"
                df = fetch_contract_klines(mexc_sym, interval)
                if df is None or df.empty or len(df) < max(50, vr_lookback + vr_confirm + 2): continue

                # --- İNDİKATÖR HESAPLAMA ---
                df_ind = ai_engine.compute_indicators(df)
                if df_ind is None or df_ind.empty or len(df_ind) < 3: continue

                latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]

                # --- 1. Genel Algoritma Skoru ---
                score, per_scores, reasons = ai_engine.score_signals(latest, prev, funding, weights)
                label = ai_engine.label_from_score(score, thresholds)

                # --- 2. Hacim Teyitli Dönüş Analizi ---
                volume_reversal_analysis = ai_engine.analyze_volume_reversal(
                    df_ind, look_back=vr_lookback, confirm_in=vr_confirm,
                    vol_multiplier=vr_vol_multi, use_ema_filter=True
                )

                # --- 3. Strateji Kombinasyon Analizi ---
                strategy_combo_analysis = ai_engine.analyze_strategy_combo(latest, adx_threshold=combo_adx_thresh)

                # --- 4. Genel AI Tahmini ---
                indicators_snapshot = { ... } # Snapshot içeriği aynı
                indicators_snapshot = {k: v for k, v in indicators_snapshot.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
                general_ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=(gemini_api_key if gemini_api_key else None))

                # --- Sonuçları Birleştir ---
                current_tf_results[tf] = {
                    'score': int(score), 'label': label, 'price': float(latest['close']),
                    'per_scores': per_scores, 'reasons': reasons,
                    'ai_analysis': general_ai_analysis,         # Genel AI sonucu
                    'volume_reversal': volume_reversal_analysis, # Hacim sonucu
                    'strategy_combo': strategy_combo_analysis   # Strateji sonucu
                }

                # En iyi TF'i belirle (Genel AI Güvenine göre)
                current_confidence = general_ai_analysis.get('confidence', 0) if general_ai_analysis.get('signal') not in ['NEUTRAL', 'ERROR'] else -1
                if current_confidence > best_ai_confidence:
                    best_ai_confidence = current_confidence; best_tf = tf

            entry['details'] = current_tf_results
            entry['best_timeframe'] = best_tf # Genel AI'a göre en iyi TF
            entry['best_score'] = int(best_ai_confidence) if best_ai_confidence >= 0 else 0 # Genel AI'a göre en iyi güven
            # ... (buy/sell count aynı) ...
            results.append(entry)
        except Exception as e: logging.error(f"Tarama hatası ({sym}): {e}", exc_info=True); st.toast(f"{sym} hatası: {e}", icon="🚨"); continue
    progress_bar_area.empty()
    if not results: logging.warning("Tarama sonuç üretmedi.")
    return pd.DataFrame(results)


# ------------- Market Analysis Functions (Aynı kaldı) --------------
# ... (get_market_analysis fonksiyonu aynı kaldı) ...

# ------------- TradingView GÖMME FONKSİYONU (Aynı kaldı) ------------
# ... (show_tradingview fonksiyonu aynı kaldı) ...

# ------------------- ANA UYGULAMA AKIŞI -------------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli (Hibrit AI)")

# --- Piyasa Analizi Alanı ---
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", help="Gelişmiş AI analizi ve Piyasa Tahmini için.", key="api_key_input")
# ... (Piyasa analizi gösterimi aynı kaldı) ...

# --- Sidebar Ayarları ---
st.sidebar.header("Tarama Ayarları")
# ... (Sembol seçimi, Zaman Dilimleri aynı kaldı) ...
all_symbols_list = fetch_all_contract_symbols(); mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top Hacim (Max 200)","Özel Liste Seç"])
symbols_to_scan_ui = [];
# ... (Sembol listesi oluşturma mantığı aynı) ...
if not symbols_to_scan_ui: st.sidebar.warning("Taranacak sembol seçilmedi."); st.stop()
timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS)
if not timeframes_ui: st.sidebar.warning("Zaman dilimi seçin."); st.stop()

# --- Hacim Teyitli Dönüş Ayarları (Aynı kaldı) ---
with st.sidebar.expander("Hacim Teyitli Dönüş Ayarları"):
    vr_lookback_ui = st.slider("Anchor Mum Periyodu", 5, 50, 20, key="vr_lookback")
    vr_confirm_ui = st.slider("Onay Periyodu", 1, 10, 5, key="vr_confirm")
    vr_vol_multi_ui = st.slider("Hacim Çarpanı", 1.1, 3.0, 1.5, step=0.1, key="vr_vol")

# --- Yeni: Strateji Kombinasyon Ayarları ---
with st.sidebar.expander("Strateji Kombinasyon Ayarları"):
     st.caption("Aşağıdaki tüm koşullar sağlandığında sinyal üretir:")
     st.markdown("- EMA Cross (20/50)\n- SuperTrend (7,3)\n- SSL Channel (14)\n- MACD Histogram\n- ADX")
     combo_adx_thresh_ui = st.slider("Minimum ADX Gücü", 10, 40, 20, key="combo_adx")

with st.sidebar.expander("Sistem Algoritması Ayarları (Eski)"):
    # ... (Ağırlıklar, Eşikler aynı kaldı) ...
    weights_ui = {...}; thresholds_ui = (...)

# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    # ... (Tarama başlatma mantığı aynı kaldı, yeni parametreler eklendi) ...
    with st.spinner("Tarama çalışıyor..."):
        try:
             st.session_state.scan_results = run_scan(
                 symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui,
                 gemini_api_key_ui, vr_lookback_ui, vr_confirm_ui, vr_vol_multi_ui,
                 combo_adx_thresh_ui # <-- Yeni parametre
             )
             # ... (Sonrası aynı) ...
        except Exception as e: # ... (Hata yönetimi aynı) ...

# --- Sonuçları Göster ---
df_results = st.session_state.scan_results
if st.session_state.last_scan_time: st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

if df_results is None or df_results.empty:
    # ... (Boş sonuç mesajı aynı kaldı) ...
    pass
else:
    # --- Veri Hazırlama (Üç Analiz Türü İçin) ---
    general_ai_list = []
    volume_reversal_list = []
    strategy_combo_list = [] # <-- Yeni liste

    for _, row in df_results.iterrows():
        symbol = row['symbol']; details = row.get('details', {})
        for tf, tf_data in details.items():
            if not tf_data: continue
            general_ai_analysis = tf_data.get('ai_analysis')
            volume_reversal_analysis = tf_data.get('volume_reversal')
            strategy_combo_analysis = tf_data.get('strategy_combo') # <-- Yeni analiz sonucu

            # Genel AI Listesi
            if general_ai_analysis: general_ai_list.append({...}) # İçerik aynı

            # Hacim Teyitli Dönüş Listesi
            if volume_reversal_analysis and volume_reversal_analysis.get('signal') != 'NONE': volume_reversal_list.append({...}) # İçerik aynı

            # Strateji Kombinasyon Listesi <-- YENİ
            if strategy_combo_analysis and strategy_combo_analysis.get('signal') != 'NONE':
                 strategy_combo_list.append({
                    'symbol': symbol, 'tf': tf, 'price': tf_data.get('price'),
                    'combo_signal': strategy_combo_analysis.get('signal'),
                    'combo_confidence': strategy_combo_analysis.get('confidence', 0),
                    'combo_confirmations': strategy_combo_analysis.get('confirming_indicators', []),
                    'combo_explanation': strategy_combo_analysis.get('explanation', ''),
                    'ai_analysis': general_ai_analysis # İlişkili genel AI
                 })


    general_ai_df = pd.DataFrame(general_ai_list)
    volume_reversal_df = pd.DataFrame(volume_reversal_list)
    strategy_combo_df = pd.DataFrame(strategy_combo_list) # <-- Yeni DataFrame

    # --- Sekmeleri Oluştur ---
    tab1, tab2, tab3 = st.tabs(["📊 Genel AI", "📈 Hacim Dönüş", "💡 Strateji Kombinasyon"])

    # --- Sekme 1: Genel AI Sinyalleri (Aynı kaldı) ---
    with tab1:
        # ... (Filtreleme, Liste, Detay Ekranı - Önceki gibi) ...
        pass

    # --- Sekme 2: Hacim Teyitli Dönüşler (Aynı kaldı) ---
    with tab2:
        # ... (Filtreleme, Liste, Detay Ekranı - Önceki gibi) ...
        pass

    # --- Sekme 3: Strateji Kombinasyon Sinyalleri ---
    with tab3:
        left3, right3 = st.columns([1.6, 2.4])
        with left3:
            st.markdown("### 💡 Strateji Kombinasyon Sinyalleri")
            # Güven %100 olduğu için filtreye gerek yok ama istenirse eklenebilir
            # min_confidence_combo = st.slider("Min Güven (%)", 0, 100, 100, step=10, key="combo_conf_filter") # Şimdilik sadece %100

            filtered_combo = strategy_combo_df.copy()
            # filtered_combo = filtered_combo[filtered_combo['combo_confidence'] >= min_confidence_combo]
            # Sıralama (henüz zaman yok ama eklenebilir)
            # filtered_combo = filtered_combo.sort_values(by='timestamp', ascending=False)

            st.caption(f"{len(filtered_combo)} strateji sinyali bulundu.")

            # Strateji sinyallerini listele
            for _, r in filtered_combo.head(MAX_SIGNALS_TO_SHOW).iterrows():
                emoji = "❓"
                if r['combo_signal'] == 'BUY': emoji = '🟩' # Farklı ikon
                elif r['combo_signal'] == 'SELL': emoji = '🟥'

                cols = st.columns([0.6, 2, 1])
                cols[0].markdown(f"<div style='font-size:20px'>{emoji}</div>", unsafe_allow_html=True)
                cols[1].markdown(f"**{r['symbol']}** • {r['tf']} \nSinyal: **{r['combo_signal']}** ({len(r['combo_confirmations'])}/{ai_engine.total_conditions})") # Kaç koşul sağlandı
                if cols[2].button("Detay", key=f"det_combo_{r['symbol']}_{r['tf']}"):
                    st.session_state.selected_symbol = r['symbol']
                    st.session_state.selected_tf = r['tf']
                    st.session_state.active_tab = "Strateji Kombinasyon Sinyalleri"
                    st.experimental_rerun()

        with right3:
            st.markdown("### 📈 Seçili Coin Detayı (Strateji Odaklı)")
            sel_sym_combo = st.session_state.selected_symbol
            sel_tf_combo = st.session_state.selected_tf

            # Başlangıçta veya tarama sonrası ilk Combo coin'i seç
            if sel_sym_combo is None and not filtered_combo.empty:
                sel_sym_combo = filtered_combo.iloc[0]['symbol']; sel_tf_combo = filtered_combo.iloc[0]['tf']
                st.session_state.selected_symbol = sel_sym_combo; st.session_state.selected_tf = sel_tf_combo

            if sel_sym_combo is None:
                st.write("Listeden bir strateji sinyali seçin.")
            else:
                 st.markdown(f"**{sel_sym_combo}** • TF: **{sel_tf_combo}**")
                 interval_tv_combo = TV_INTERVAL_MAP.get(sel_tf_combo, '60')
                 show_tradingview(sel_sym_combo, interval_tv_combo, height=400)

                 # Doğru veriyi bul (strategy_combo_list'ten)
                 row_data_combo = next((x for x in strategy_combo_list if x['symbol']==sel_sym_combo and x['tf'] == sel_tf_combo), None)

                 if row_data_combo:
                      st.markdown(f"#### 💡 Strateji Sinyali: {row_data_combo['combo_signal']}")
                      st.markdown("**Onaylanan İndikatörler:**")
                      confirmations = row_data_combo.get('combo_confirmations', [])
                      if confirmations:
                           for conf in confirmations:
                                st.markdown(f"- {conf}")
                      else:
                           st.markdown("- *Onaylanan indikatör bulunamadı.*")

                      st.markdown("---")
                      st.markdown("#### 🧠 Genel AI Yorumu (O An İçin)")
                      ai_analysis_combo = row_data_combo.get('ai_analysis')
                      if ai_analysis_combo:
                           st.markdown(ai_analysis_combo.get('explanation', 'Genel AI yorumu bulunamadı.'))
                           ti_combo = ai_analysis_combo; entry_combo = ti_combo.get('entry'); stop_combo = ti_combo.get('stop_loss'); target_combo = ti_combo.get('take_profit')
                           # Metrikleri göster (Genel AI'dan alınan seviyeler)
                           if entry_combo is not None and stop_combo is not None and target_combo is not None:
                                c1c, c2c, c3c = st.columns(3); entry_str_c=...; stop_str_c=...; target_str_c=... # Formatlama
                                c1c.metric("AI Giriş", entry_str_c); c2c.metric("AI Stop", stop_str_c); c3c.metric("AI Hedef", target_str_c)
                      else:
                           st.warning("Bu strateji sinyali anı için genel AI analizi bulunamadı.")

                      # --- Takip/Kayıt/İndir Butonları (Strateji Sinyali için) ---
                      track_key_combo = f"track_{sel_sym_combo}_{sel_tf_combo}"
                      # ... (Buton mantığı diğer sekmelerle aynı) ...

                 else:
                      st.warning(f"{sel_sym_combo} ({sel_tf_combo}) için strateji detay verisi bulunamadı.")


    # --- Takip Edilen Sinyaller (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

    # --- Özet Metrikler ve Kayıtlı Tahminler (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
