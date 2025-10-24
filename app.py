# app.py
# Streamlit MEXC contract sinyal uygulaması - (v3.6 - SyntaxError Fix + Specter Tab)

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
try: import plotly.express as px; PLOTLY_AVAILABLE = True
except ImportError: PLOTLY_AVAILABLE = False; logging.warning("Plotly yok.")

st.set_page_config(page_title="MEXC Vadeli - Profesyonel Sinyal Paneli", layout="wide", initial_sidebar_state="collapsed")

# --- Session State Başlatma ---
# ... (Aynı kaldı) ...
if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame() #... (diğer state'ler)

# ---------------- CONFIG & CONSTANTS ----------------
# ... (MA Tipleri Eklendi) ...
CONTRACT_BASE = "https://contract.mexc.com/api/v1"; INTERVAL_MAP = {...}; TV_INTERVAL_MAP = {...}; DEFAULT_TFS = ['15m','1h','4h']; ALL_TFS = [...]; DEFAULT_WEIGHTS = {...}; SCALP_TFS = [...]; SWING_TFS = [...]
EMA_TREND_LENGTH = ai_engine.EMA_TREND_LENGTH
SPECTER_ATR_LENGTH = ai_engine.SPECTER_ATR_LENGTH
MA_TYPES = ['EMA', 'SMA', 'SMMA', 'WMA', 'VWMA'] # Seçilebilir MA Tipleri

# CSS
# ... (Aynı kaldı) ...
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

# ---------------- API Helpers (Aynı kaldı) ----------------
# ... (fetch_all_contract_symbols, fetch_json, get_top_contracts_by_volume, mexc_symbol_from, fetch_contract_klines, fetch_contract_funding_rate - Aynı kaldı) ...

# ---------------- Scan Engine (Specter çağrısı eklendi, SyntaxError düzeltildi) ----------------
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key,
             vr_lookback, vr_confirm, vr_vol_multi, combo_adx_thresh,
             specter_ma_type, specter_ma_length): # <-- Specter parametreleri eklendi
    """Ana tarama fonksiyonu - Tüm analiz motorlarını çağırır."""
    results = []
    total_symbols = len(symbols_to_scan)
    progress_bar_area = st.sidebar.empty()
    progress_bar = progress_bar_area.progress(0, text="Tarama başlatılıyor...")

    for i, sym in enumerate(symbols_to_scan):
        progress_value = (i + 1) / total_symbols; progress_text = f"Taranıyor: {sym} ({i+1}/{total_symbols})"
        progress_bar.progress(progress_value, text=progress_text)
        entry = {'symbol': sym, 'details': {}}; best_ai_confidence = -1; best_tf = None
        mexc_sym = mexc_symbol_from(sym);
        if not mexc_sym.endswith("_USDT"): continue
        try:
            funding = fetch_contract_funding_rate(mexc_sym); current_tf_results = {}
            for tf in timeframes:
                interval = INTERVAL_MAP.get(tf)
                # --- SyntaxError Düzeltmesi ---
                scan_mode = "Normal"
                if tf in SCALP_TFS:
                    scan_mode = "Scalp"
                elif tf in SWING_TFS:
                    scan_mode = "Swing"
                # --- Düzeltme Sonu ---

                df = fetch_contract_klines(mexc_sym, interval)
                # Yeterli veri kontrolü (Specter ATR'yi de hesaba kat)
                min_bars_needed = max(50, vr_lookback + vr_confirm + 2, SPECTER_ATR_LENGTH + 5)
                if df is None or df.empty or len(df) < min_bars_needed: continue

                # --- İNDİKATÖR HESAPLAMA (Specter parametreleri ile) ---
                df_ind = ai_engine.compute_indicators(df, ma_type=specter_ma_type, ma_length=specter_ma_length)
                if df_ind is None or df_ind.empty or len(df_ind) < 3: continue

                latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2] # Genel AI/Skorlama için
                # Not: Specter ve VR kendi içlerinde df_ind'in son mumlarını kullanır

                # --- 1. Genel Algoritma Skoru ---
                score, per_scores, reasons = ai_engine.score_signals(latest, prev, funding, weights)
                label = ai_engine.label_from_score(score, thresholds)

                # --- 2. Hacim Teyitli Dönüş Analizi ---
                volume_reversal_analysis = ai_engine.analyze_volume_reversal(df_ind, look_back=vr_lookback, confirm_in=vr_confirm, vol_multiplier=vr_vol_multi)

                # --- 3. Strateji Kombinasyon Analizi ---
                strategy_combo_analysis = ai_engine.analyze_strategy_combo(latest, adx_threshold=combo_adx_thresh)

                # --- 4. Specter Trend Analizi ---
                specter_trend_analysis = ai_engine.analyze_specter_trend(df_ind) # Varsayılan cooldown=5

                # --- 5. Genel AI Tahmini ---
                indicators_snapshot = { ... } # Snapshot içeriği aynı
                indicators_snapshot = {k: v for k, v in indicators_snapshot.items() if ...} # NaN temizleme
                general_ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=(gemini_api_key if gemini_api_key else None))

                # --- Sonuçları Birleştir ---
                current_tf_results[tf] = {
                    'score': int(score), 'label': label, 'price': float(latest['close']),
                    'per_scores': per_scores, 'reasons': reasons,
                    'ai_analysis': general_ai_analysis,
                    'volume_reversal': volume_reversal_analysis,
                    'strategy_combo': strategy_combo_analysis,
                    'specter_trend': specter_trend_analysis # <-- Specter eklendi
                }

                # En iyi TF'i belirle (Genel AI Güvenine göre - Aynı kaldı)
                current_confidence = general_ai_analysis.get('confidence', 0) #...
                if current_confidence > best_ai_confidence: best_ai_confidence = current_confidence; best_tf = tf

            entry['details'] = current_tf_results
            entry['best_timeframe'] = best_tf; entry['best_score'] = int(best_ai_confidence) if best_ai_confidence >= 0 else 0
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
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", key="api_key_input")
# ... (Piyasa analizi gösterimi aynı kaldı) ...

# --- Sidebar Ayarları ---
st.sidebar.header("Tarama Ayarları")
# ... (Sembol seçimi, Zaman Dilimleri aynı kaldı) ...
all_symbols_list = fetch_all_contract_symbols(); mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top Hacim","Özel Liste"]) # İsim kısaltıldı
# ... (symbols_to_scan_ui oluşturma aynı) ...
timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS)
# ... (Eksik seçim kontrolü aynı) ...

# --- Yeni: Specter Trend Ayarları ---
with st.sidebar.expander("☁️ Specter Trend Ayarları"):
    specter_ma_type_ui = st.selectbox("MA Tipi", options=MA_TYPES, index=0, key="specter_ma_type") # index=0 -> EMA default
    specter_ma_length_ui = st.slider("Kısa MA Periyodu (Uzun=2x)", 5, 100, 21, key="specter_ma_len") # Default 21

# --- Diğer Ayarlar (Hacim Dönüş, Strateji, Algoritma - Aynı kaldı) ---
with st.sidebar.expander("📈 Hacim Teyitli Dönüş Ayarları"): vr_lookback_ui=...; vr_confirm_ui=...; vr_vol_multi_ui=...
with st.sidebar.expander("💡 Strateji Kombinasyon Ayarları"): combo_adx_thresh_ui=...
with st.sidebar.expander("⚙️ Sistem Algoritması Ayarları (Eski)"): weights_ui={...}; thresholds_ui=(...)

# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    # ... (Tarama başlatma mantığı aynı kaldı, yeni parametreler eklendi) ...
    with st.spinner("Tarama çalışıyor..."):
        try:
             st.session_state.scan_results = run_scan(
                 symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui,
                 gemini_api_key_ui, vr_lookback_ui, vr_confirm_ui, vr_vol_multi_ui,
                 combo_adx_thresh_ui, specter_ma_type_ui, specter_ma_length_ui # <-- Yeni parametreler
             )
             # ... (Sonrası aynı) ...
        except Exception as e: # ... (Hata yönetimi aynı) ...

# --- Sonuçları Göster ---
df_results = st.session_state.scan_results
if st.session_state.last_scan_time: st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

if df_results is None or df_results.empty: # ... (Boş sonuç mesajı aynı) ...
    pass
else:
    # --- Veri Hazırlama (Dört Analiz Türü İçin) ---
    general_ai_list = []; volume_reversal_list = []; strategy_combo_list = []; specter_trend_list = [] # <-- Specter listesi eklendi

    for _, row in df_results.iterrows():
        symbol = row['symbol']; details = row.get('details', {})
        for tf, tf_data in details.items():
            if not tf_data: continue
            general_ai_analysis = tf_data.get('ai_analysis')
            volume_reversal_analysis = tf_data.get('volume_reversal')
            strategy_combo_analysis = tf_data.get('strategy_combo')
            specter_trend_analysis = tf_data.get('specter_trend') # <-- Specter verisi

            # Genel AI Listesi
            if general_ai_analysis: general_ai_list.append({...}) # İçerik aynı (VR ve Specter bilgisi eklenebilir)

            # Hacim Teyitli Dönüş Listesi
            if volume_reversal_analysis and volume_reversal_analysis.get('signal') != 'NONE': volume_reversal_list.append({...}) # İçerik aynı

            # Strateji Kombinasyon Listesi
            if strategy_combo_analysis and strategy_combo_analysis.get('signal') != 'NONE': strategy_combo_list.append({...}) # İçerik aynı

            # Specter Trend Listesi <-- YENİ
            if specter_trend_analysis:
                 specter_trend_list.append({
                    'symbol': symbol, 'tf': tf, 'price': tf_data.get('price'),
                    'specter_trend': specter_trend_analysis.get('trend', 'NEUTRAL'),
                    'specter_retest_signal': specter_trend_analysis.get('retest_signal', 'NONE'),
                    'specter_retest_price': specter_trend_analysis.get('retest_price'),
                    'specter_status': specter_trend_analysis.get('status', ''),
                    'ai_analysis': general_ai_analysis # İlişkili genel AI
                 })

    general_ai_df = pd.DataFrame(general_ai_list)
    volume_reversal_df = pd.DataFrame(volume_reversal_list)
    strategy_combo_df = pd.DataFrame(strategy_combo_list)
    specter_trend_df = pd.DataFrame(specter_trend_list) # <-- Yeni DataFrame

    # --- Sekmeleri Oluştur ---
    tab_titles = ["📊 Genel AI", "📈 Hacim Dönüş", "💡 Strateji Komb.", "☁️ Specter Trend"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    # --- Sekme 1: Genel AI Sinyalleri (Aynı kaldı) ---
    with tab1: #...
        pass

    # --- Sekme 2: Hacim Teyitli Dönüşler (Aynı kaldı) ---
    with tab2: #...
        pass

    # --- Sekme 3: Strateji Kombinasyon Sinyalleri (Aynı kaldı) ---
    with tab3: #...
        pass

    # --- Sekme 4: Specter Trend Sinyalleri ---
    with tab4:
        left4, right4 = st.columns([1.6, 2.4])
        with left4:
            st.markdown("### ☁️ Specter Trend & Retest Sinyalleri")
            filter_trend_specter = st.selectbox("Trend Yönü", ["Tümü", "BULLISH", "BEARISH"], index=0, key="specter_trend_filter")
            filter_retest_specter = st.checkbox("Sadece Retest Sinyallerini Göster", key="specter_retest_filter")

            filtered_specter = specter_trend_df.copy()
            if not filtered_specter.empty:
                 if filter_trend_specter != "Tümü":
                      filtered_specter = filtered_specter[filtered_specter['specter_trend'] == filter_trend_specter]
                 if filter_retest_specter:
                      filtered_specter = filtered_specter[filtered_specter['specter_retest_signal'] != 'NONE']
                 # Sıralama (henüz zaman yok)
                 # filtered_specter = filtered_specter.sort_values(by='timestamp', ascending=False)

            st.caption(f"{len(filtered_specter)} durum bulundu.")

            # Specter durumlarını/sinyallerini listele
            for _, r in filtered_specter.head(MAX_SIGNALS_TO_SHOW).iterrows():
                trend_color = "🟢" if r['specter_trend'] == 'BULLISH' else ("🟠" if r['specter_trend'] == 'BEARISH' else "⚪")
                retest_icon = "💎" if r['specter_retest_signal'] != 'NONE' else ""

                cols = st.columns([0.6, 2, 1])
                cols[0].markdown(f"<div style='font-size:20px'>{trend_color}{retest_icon}</div>", unsafe_allow_html=True)
                retest_info = f" **{r['specter_retest_signal']} Retest!**" if retest_icon else ""
                cols[1].markdown(f"**{r['symbol']}** • {r['tf']} \nTrend: **{r['specter_trend']}**{retest_info}")
                if cols[2].button("Detay", key=f"det_specter_{r['symbol']}_{r['tf']}"):
                    st.session_state.selected_symbol = r['symbol']
                    st.session_state.selected_tf = r['tf']
                    st.session_state.active_tab = "☁️ Specter Trend Sinyalleri"
                    st.experimental_rerun()

        with right4:
            st.markdown("### 📈 Seçili Coin Detayı (Specter Odaklı)")
            sel_sym_spec = st.session_state.selected_symbol
            sel_tf_spec = st.session_state.selected_tf

            # Başlangıçta veya tarama sonrası ilk Specter coin'i seç
            if sel_sym_spec is None and not filtered_specter.empty:
                sel_sym_spec = filtered_specter.iloc[0]['symbol']; sel_tf_spec = filtered_specter.iloc[0]['tf']
                st.session_state.selected_symbol = sel_sym_spec; st.session_state.selected_tf = sel_tf_spec

            if sel_sym_spec is None:
                st.write("Listeden bir durum/sinyal seçin.")
            else:
                 st.markdown(f"**{sel_sym_spec}** • TF: **{sel_tf_spec}**")
                 interval_tv_spec = TV_INTERVAL_MAP.get(sel_tf_spec, '60')
                 show_tradingview(sel_sym_spec, interval_tv_spec, height=400)

                 # Doğru veriyi bul (specter_trend_list'ten)
                 row_data_spec = next((x for x in specter_trend_list if x['symbol']==sel_sym_spec and x['tf'] == sel_tf_spec), None)

                 if row_data_spec:
                      st.markdown(f"#### ☁️ Specter Analizi")
                      trend_color = "green" if row_data_spec['specter_trend'] == 'BULLISH' else ("orange" if row_data_spec['specter_trend'] == 'BEARISH' else "gray")
                      st.markdown(f"- **Trend Yönü:** <span style='color:{trend_color}; font-weight:bold;'>{row_data_spec['specter_trend']}</span>", unsafe_allow_html=True)
                      if row_data_spec['specter_retest_signal'] != 'NONE':
                           st.success(f"**Retest Sinyali: {row_data_spec['specter_retest_signal']}**")
                           st.markdown(f"- **Retest Fiyat Seviyesi:** `{row_data_spec.get('specter_retest_price', 'N/A')}`")
                      st.caption(f"Durum: {row_data_spec.get('specter_status', 'N/A')}")


                      st.markdown("---")
                      st.markdown("#### 🧠 Genel AI Yorumu (O An İçin)")
                      ai_analysis_spec = row_data_spec.get('ai_analysis')
                      if ai_analysis_spec:
                           st.markdown(ai_analysis_spec.get('explanation', 'Genel AI yorumu bulunamadı.'))
                           ti_spec = ai_analysis_spec; entry_spec = ti_spec.get('entry'); stop_spec = ti_spec.get('stop_loss'); target_spec = ti_spec.get('take_profit')
                           # Metrikleri göster (Genel AI'dan alınan seviyeler)
                           if entry_spec is not None and stop_spec is not None and target_spec is not None:
                                c1s, c2s, c3s = st.columns(3); entry_str_s=...; stop_str_s=...; target_str_s=... # Formatlama
                                c1s.metric("AI Giriş", entry_str_s); c2s.metric("AI Stop", stop_str_s); c3s.metric("AI Hedef", target_str_s)
                      else:
                           st.warning("Bu Specter durumu anı için genel AI analizi bulunamadı.")

                      # --- Takip/Kayıt/İndir Butonları (Specter Sinyali için) ---
                      # Not: Aynı session state'i (tracked_signals) kullanabiliriz.
                      track_key_spec = f"track_{sel_sym_spec}_{sel_tf_spec}"
                      # ... (Buton mantığı diğer sekmelerle aynı) ...

                 else:
                      st.warning(f"{sel_sym_spec} ({sel_tf_spec}) için Specter detay verisi bulunamadı.")


    # --- Takip Edilen Sinyaller (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

    # --- Özet Metrikler ve Kayıtlı Tahminler (Aynı kaldı) ---
    # ... (Gösterim aynı) ...

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
