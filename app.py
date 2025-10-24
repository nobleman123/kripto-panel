# app.py
# Streamlit MEXC contract sinyal uygulaması - (v3.4 - Volume Reversal Sekmesi)

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
if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'selected_symbol' not in st.session_state: st.session_state.selected_symbol = None
if 'selected_tf' not in st.session_state: st.session_state.selected_tf = '15m'
if 'tracked_signals' not in st.session_state: st.session_state.tracked_signals = {}
if 'last_scan_time' not in st.session_state: st.session_state.last_scan_time = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = "Genel AI Sinyalleri" # Sekme state'i

# ---------------- CONFIG & CONSTANTS ----------------
# ... (Aynı kaldı) ...
CONTRACT_BASE = "https://contract.mexc.com/api/v1"; INTERVAL_MAP = {...}; TV_INTERVAL_MAP = {...}; DEFAULT_TFS = ['15m','1h','4h']; ALL_TFS = [...]; DEFAULT_WEIGHTS = {...}; SCALP_TFS = [...]; SWING_TFS = [...]
EMA_TREND_LENGTH = ai_engine.EMA_TREND_LENGTH # ai_engine'den al

# CSS
# ... (Aynı kaldı) ...
st.markdown("""<style>...</style>""", unsafe_allow_html=True)

# ---------------- API Helpers (Aynı kaldı) ----------------
# ... (fetch_all_contract_symbols, fetch_json, get_top_contracts_by_volume, mexc_symbol_from, fetch_contract_klines, fetch_contract_funding_rate - Aynı kaldı) ...

# ---------------- Scan Engine (Volume Reversal çağrısı eklendi) ----------------
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key, vr_lookback, vr_confirm, vr_vol_multi):
    """Ana tarama fonksiyonu - Genel AI ve Hacim Dönüşünü analiz eder."""
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
                # ... (Interval, scan_mode belirleme, kline çekme, indikatör hesaplama aynı kaldı) ...
                interval = INTERVAL_MAP.get(tf); scan_mode = "Normal"
                if tf in SCALP_TFS: scan_mode = "Scalp"; elif tf in SWING_TFS: scan_mode = "Swing"
                df = fetch_contract_klines(mexc_sym, interval)
                if df is None or df.empty or len(df) < max(50, vr_lookback + vr_confirm + 2): continue # Hacim analizi için yeterli veri
                df_ind = ai_engine.compute_indicators(df)
                if df_ind is None or df_ind.empty or len(df_ind) < 3: continue

                latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2] # Genel AI için son mum, VR için -2 kullanılır

                # --- 1. Genel Algoritma Skoru ---
                score, per_scores, reasons = ai_engine.score_signals(latest, prev, funding, weights)
                label = ai_engine.label_from_score(score, thresholds)

                # --- 2. Hacim Teyitli Dönüş Analizi ---
                volume_reversal_analysis = ai_engine.analyze_volume_reversal(
                    df_ind, look_back=vr_lookback, confirm_in=vr_confirm,
                    vol_multiplier=vr_vol_multi, use_ema_filter=True
                )

                # --- 3. Genel AI Tahmini ---
                indicators_snapshot = { # Snapshot içeriği aynı
                    'symbol': sym, 'timeframe': tf, 'scan_mode': scan_mode, 'score': int(score), 'price': float(latest['close']),
                    'rsi14': latest.get('rsi14'), 'macd_hist': latest.get('macd_hist'), 'vol_osc': latest.get('vol_osc'),
                    'atr14': latest.get('atr14'), 'nw_slope': latest.get('nw_slope'), 'bb_upper': latest.get('bb_upper'),
                    'bb_lower': latest.get('bb_lower'), 'funding_rate': funding.get('fundingRate')
                }
                indicators_snapshot = {k: v for k, v in indicators_snapshot.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
                general_ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=(gemini_api_key if gemini_api_key else None))

                # --- Sonuçları Birleştir ---
                current_tf_results[tf] = {
                    'score': int(score), 'label': label, 'price': float(latest['close']),
                    'per_scores': per_scores, 'reasons': reasons,
                    'ai_analysis': general_ai_analysis, # Genel AI sonucu
                    'volume_reversal': volume_reversal_analysis # Hacim sonucu
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
if gemini_api_key_ui: # Piyasa analizi gösterimi...
     pass

# --- Sidebar Ayarları ---
st.sidebar.header("Tarama Ayarları")
# ... (Sembol seçimi, Zaman Dilimleri aynı kaldı) ...
all_symbols_list = fetch_all_contract_symbols(); mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top Hacim (Max 200)","Özel Liste Seç"])
symbols_to_scan_ui = [];
if mode == "Özel Liste Seç": selected_symbols_ui = st.sidebar.multiselect("Coinleri Seçin", options=all_symbols_list, default=["BTCUSDT", "ETHUSDT"]); symbols_to_scan_ui = selected_symbols_ui
else: symbols_by_volume_list = get_top_contracts_by_volume(200); top_n_ui = st.sidebar.slider("İlk N Coin", min_value=5, max_value=len(symbols_by_volume_list), value=min(50, len(symbols_by_volume_list))); symbols_to_scan_ui = symbols_by_volume_list[:top_n_ui]
if not symbols_to_scan_ui: st.sidebar.warning("Taranacak sembol seçilmedi."); st.stop()
timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS)
if not timeframes_ui: st.sidebar.warning("Zaman dilimi seçin."); st.stop()

# --- Yeni: Hacim Teyitli Dönüş Ayarları ---
with st.sidebar.expander("Hacim Teyitli Dönüş Ayarları"):
    vr_lookback_ui = st.slider("Anchor Mum Arama Periyodu", 5, 50, 20, key="vr_lookback")
    vr_confirm_ui = st.slider("Onay Bekleme Periyodu", 1, 10, 5, key="vr_confirm")
    vr_vol_multi_ui = st.slider("Hacim Çarpanı (Ortalamanın Kaç Katı)", 1.1, 3.0, 1.5, step=0.1, key="vr_vol")

with st.sidebar.expander("Sistem Algoritması Ayarları"):
    # ... (Ağırlıklar, Eşikler aynı kaldı) ...
    weights_ui = {...}; thresholds_ui = (...)

# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    # ... (Tarama başlatma mantığı aynı kaldı) ...
    with st.spinner("Tarama çalışıyor..."):
        try:
             st.session_state.scan_results = run_scan(
                 symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui,
                 gemini_api_key_ui, vr_lookback_ui, vr_confirm_ui, vr_vol_multi_ui # Yeni parametreler eklendi
             )
             st.session_state.last_scan_time = datetime.now()
             st.session_state.selected_symbol = None
             st.experimental_rerun()
        except Exception as e: # Genel hata yakalama
             logging.error(f"Beklenmedik tarama hatası (ana): {e}", exc_info=True)
             st.error(f"Tarama sırasında hata: {e}")
             st.session_state.scan_results = pd.DataFrame()


# --- Sonuçları Göster ---
df_results = st.session_state.scan_results
if st.session_state.last_scan_time: st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

if df_results is None or df_results.empty:
    # ... (Boş sonuç mesajı aynı kaldı) ...
    pass
else:
    # --- Veri Hazırlama (İki Analiz Türü İçin) ---
    general_ai_list = []
    volume_reversal_list = []

    for _, row in df_results.iterrows():
        symbol = row['symbol']
        details = row.get('details', {})
        for tf, tf_data in details.items():
            if not tf_data: continue

            # Genel AI Verisi
            general_ai_analysis = tf_data.get('ai_analysis')
            if general_ai_analysis:
                general_ai_list.append({
                    'symbol': symbol, 'tf': tf, 'price': tf_data.get('price'),
                    'ai_signal': general_ai_analysis.get('signal', 'NEUTRAL'),
                    'ai_confidence': general_ai_analysis.get('confidence', 0),
                    'ai_text': general_ai_analysis.get('explanation', '...'),
                    'target_info': general_ai_analysis,
                    'algo_score': tf_data.get('score'), 'algo_label': tf_data.get('label'),
                    'per_scores': tf_data.get('per_scores'), 'volume_reversal': tf_data.get('volume_reversal') # VR bilgisini buraya da ekle
                })

            # Hacim Teyitli Dönüş Verisi
            volume_reversal_analysis = tf_data.get('volume_reversal')
            if volume_reversal_analysis and volume_reversal_analysis.get('signal') != 'NONE':
                 volume_reversal_list.append({
                    'symbol': symbol, 'tf': tf, 'price': tf_data.get('price'), # Onay anındaki fiyatı alabiliriz
                    'vr_signal': volume_reversal_analysis.get('signal'),
                    'vr_score': volume_reversal_analysis.get('score', 0),
                    'vr_status': volume_reversal_analysis.get('status', ''),
                    'vr_details': volume_reversal_analysis, # Tüm detaylar
                    'ai_analysis': general_ai_analysis # İlişkili genel AI analizini de sakla
                 })

    general_ai_df = pd.DataFrame(general_ai_list)
    volume_reversal_df = pd.DataFrame(volume_reversal_list)

    # --- Sekmeleri Oluştur ---
    tab1, tab2 = st.tabs(["📊 Genel AI Sinyalleri", "📈 Hacim Teyitli Dönüşler"])

    # --- Sekme 1: Genel AI Sinyalleri ---
    with tab1:
        left1, right1 = st.columns([1.6, 2.4])
        with left1:
            st.markdown("### 🔎 Genel AI Sinyal Listesi")
            filter_signal_gen = st.selectbox("Sinyal Türü", ["All","LONG","SHORT","NEUTRAL", "ERROR"], index=0, key="gen_signal_filter")
            min_confidence_gen = st.slider("Min Güven (%)", 0, 100, 30, step=5, key="gen_conf_filter")

            filtered_gen = general_ai_df.copy()
            if not filtered_gen.empty:
                # ... (Genel AI filtreleme mantığı aynı kaldı - KeyError kontrolü dahil) ...
                if filter_signal_gen != "All": filtered_gen = filtered_gen[filtered_gen['ai_signal'] == filter_signal_gen]
                filtered_gen = filtered_gen[filtered_gen['ai_confidence'] >= min_confidence_gen]
                filtered_gen = filtered_gen.sort_values(by='ai_confidence', ascending=False)


            st.caption(f"{len(filtered_gen)} sinyal bulundu.")
            # ... (Genel AI liste gösterimi aynı kaldı - emoji, algo karşılaştırması vs.) ...
            MAX_SIGNALS_TO_SHOW = 150
            for _, r in filtered_gen.head(MAX_SIGNALS_TO_SHOW).iterrows():
                 emoji="⚪"; if r['ai_signal']=='LONG': emoji='🚀'; elif r['ai_signal']=='SHORT': emoji='🔻'; elif r['ai_signal']=='ERROR': emoji='⚠️'
                 cols=st.columns([0.6,2,1]); cols[0].markdown(f"<div ...>{emoji}</div>", unsafe_allow_html=True)
                 algo_info = f"Algo: {r.get('algo_label','N/A')} ({r.get('algo_score','N/A')})"
                 cols[1].markdown(f"**{r['symbol']}** • {r['tf']} \nAI: **{r['ai_signal']}** (%{r['ai_confidence']}) <span ...>{algo_info}</span>", unsafe_allow_html=True)
                 if cols[2].button("Detay", key=f"det_gen_{r['symbol']}_{r['tf']}"):
                      st.session_state.selected_symbol = r['symbol']; st.session_state.selected_tf = r['tf']
                      st.session_state.active_tab = "Genel AI Sinyalleri" # Bu sekmede kal
                      st.experimental_rerun()


        with right1:
            st.markdown("### 📈 Seçili Coin Detayı")
            sel_sym = st.session_state.selected_symbol
            sel_tf_val = st.session_state.selected_tf

            # Başlangıçta veya tarama sonrası ilk coin'i seç (Filtrelenmiş Genel AI'dan)
            if sel_sym is None and not filtered_gen.empty:
                sel_sym = filtered_gen.iloc[0]['symbol']; sel_tf_val = filtered_gen.iloc[0]['tf']
                st.session_state.selected_symbol = sel_sym; st.session_state.selected_tf = sel_tf_val

            if sel_sym is None:
                st.write("Listeden bir coin seçin.")
            else:
                # --- Detay Gösterimi (Genel AI Odaklı) ---
                st.markdown(f"**{sel_sym}** • TF: **{sel_tf_val}**")
                interval_tv_val = TV_INTERVAL_MAP.get(sel_tf_val, '60')
                show_tradingview(sel_sym, interval_tv_val, height=400)

                # Doğru veriyi bul (general_ai_list'ten)
                row_data = next((x for x in general_ai_list if x['symbol']==sel_sym and x['tf'] == sel_tf_val), None)

                if row_data:
                    st.markdown("#### 🧠 Genel AI Analizi ve Ticaret Planı")
                    st.markdown(row_data['ai_text'])
                    ti_data = row_data['target_info']; entry_val = ti_data.get('entry'); stop_val = ti_data.get('stop_loss'); target_val = ti_data.get('take_profit')
                    # ... (Metrik gösterimi aynı kaldı) ...
                    if entry_val is not None and stop_val is not None and target_val is not None: c1,c2,c3=st.columns(3); entry_str=...; stop_str=...; target_str=...; c1.metric(...); c2.metric(...); c3.metric(...)

                    # Hacim Teyitli Dönüş bilgisini de göster (varsa)
                    vr_info = row_data.get('volume_reversal')
                    if vr_info and vr_info.get('signal') != 'NONE':
                         st.info(f"**Hacim Teyitli Dönüş Sinyali:** {vr_info['signal']} ({vr_info['score']}/4) - Durum: {vr_info['status']}")

                    # --- Takip/Kayıt/İndir Butonları (Genel AI için) ---
                    track_key = f"track_{sel_sym}_{sel_tf_val}" # Key aynı kalabilir
                    is_tracked = track_key in st.session_state.tracked_signals
                    # ... (Buton mantığı aynı kaldı) ...
                    if st.button("❌ Takipten Çıkar" if is_tracked else "📌 Sinyali Takip Et", key=f"track_btn_{track_key}"): #...
                         pass
                    b1, b2, b3 = st.columns([1,1,1])
                    if b1.button("✅ Başarılı", key=f"success_{track_key}"): #... Kaydet...
                         pass
                    if b2.button("❌ Başarısız", key=f"fail_{track_key}"): #... Kaydet...
                         pass
                    if b3.button("📥 İndir", key=f"dl_{track_key}"): #... İndir...
                         pass

                    # Algoritma Puanları (Aynı kaldı)
                    with st.expander("Algoritma Puan Katkıları (Eski Sistem)"):
                         # ... (Grafik/Tablo gösterimi aynı) ...
                         pass
                else:
                    st.warning(f"{sel_sym} ({sel_tf_val}) için detay verisi bulunamadı.")

    # --- Sekme 2: Hacim Teyitli Dönüşler ---
    with tab2:
        left2, right2 = st.columns([1.6, 2.4])
        with left2:
            st.markdown("### 📈 Hacim Teyitli Dönüş Sinyalleri")
            min_score_vr = st.slider("Minimum Sinyal Skoru (1-4)", 1, 4, 2, key="vr_score_filter")

            filtered_vr = volume_reversal_df.copy()
            if not filtered_vr.empty:
                 filtered_vr = filtered_vr[filtered_vr['vr_score'] >= min_score_vr]
                 # Skora ve sonra zamana göre sırala (henüz zaman eklemedik ama eklenebilir)
                 filtered_vr = filtered_vr.sort_values(by='vr_score', ascending=False)

            st.caption(f"{len(filtered_vr)} hacim teyitli sinyal bulundu.")

            # Hacim sinyallerini listele
            for _, r in filtered_vr.head(MAX_SIGNALS_TO_SHOW).iterrows():
                emoji = "❓"
                if r['vr_signal'] == 'BUY': emoji = '🔼' # Hacim için farklı ikonlar
                elif r['vr_signal'] == 'SELL': emoji = '🔽'

                cols = st.columns([0.6, 2, 1])
                cols[0].markdown(f"<div style='font-size:20px'>{emoji}</div>", unsafe_allow_html=True)
                cols[1].markdown(f"**{r['symbol']}** • {r['tf']} \nSinyal: **{r['vr_signal']}** (Skor: {r['vr_score']}/4)")
                if cols[2].button("Detay", key=f"det_vr_{r['symbol']}_{r['tf']}"):
                    st.session_state.selected_symbol = r['symbol']
                    st.session_state.selected_tf = r['tf']
                    st.session_state.active_tab = "Hacim Teyitli Dönüşler" # Bu sekmede kal
                    st.experimental_rerun()

        with right2:
            st.markdown("### 📈 Seçili Coin Detayı (Hacim Odaklı)")
            sel_sym_vr = st.session_state.selected_symbol
            sel_tf_vr = st.session_state.selected_tf

             # Başlangıçta veya tarama sonrası ilk VR coin'i seç (Filtrelenmiş VR'dan)
            if sel_sym_vr is None and not filtered_vr.empty:
                sel_sym_vr = filtered_vr.iloc[0]['symbol']; sel_tf_vr = filtered_vr.iloc[0]['tf']
                st.session_state.selected_symbol = sel_sym_vr; st.session_state.selected_tf = sel_tf_vr

            if sel_sym_vr is None:
                st.write("Listeden bir hacim sinyali seçin.")
            else:
                 st.markdown(f"**{sel_sym_vr}** • TF: **{sel_tf_vr}**")
                 interval_tv_vr = TV_INTERVAL_MAP.get(sel_tf_vr, '60')
                 show_tradingview(sel_sym_vr, interval_tv_vr, height=400)

                 # Doğru veriyi bul (volume_reversal_list'ten)
                 row_data_vr = next((x for x in volume_reversal_list if x['symbol']==sel_sym_vr and x['tf'] == sel_tf_vr), None)

                 if row_data_vr:
                      st.markdown(f"#### 📈 Hacim Teyitli Sinyal: {row_data_vr['vr_signal']} (Skor: {row_data_vr['vr_score']}/4)")
                      vr_details = row_data_vr.get('vr_details', {})
                      st.markdown(f"""
                      - **Durum:** {row_data_vr.get('vr_status', 'N/A')}
                      - **Anchor Mum:** {vr_details.get('anchor_time', 'N/A')} (High: {vr_details.get('anchor_price_high','N/A')}, Low: {vr_details.get('anchor_price_low','N/A')})
                      - **Onay Mum:** {vr_details.get('confirmation_time', 'N/A')} (Kapanış: {vr_details.get('confirmation_price','N/A')})
                      """)

                      st.markdown("---")
                      st.markdown("#### 🧠 Genel AI Yorumu (O An İçin)")
                      ai_analysis_vr = row_data_vr.get('ai_analysis')
                      if ai_analysis_vr:
                           st.markdown(ai_analysis_vr.get('explanation', 'Genel AI yorumu bulunamadı.'))
                           ti_vr = ai_analysis_vr; entry_vr = ti_vr.get('entry'); stop_vr = ti_vr.get('stop_loss'); target_vr = ti_vr.get('take_profit')
                           # Metrikleri göster (Genel AI'dan alınan seviyeler)
                           if entry_vr is not None and stop_vr is not None and target_vr is not None:
                                c1v, c2v, c3v = st.columns(3); entry_str_v=...; stop_str_v=...; target_str_v=... # Formatlama
                                c1v.metric("AI Giriş", entry_str_v); c2v.metric("AI Stop", stop_str_v); c3v.metric("AI Hedef", target_str_v)
                      else:
                           st.warning("Bu hacim sinyali anı için genel AI analizi bulunamadı.")

                      # --- Takip/Kayıt/İndir Butonları (Hacim Sinyali için) ---
                      # Not: Aynı session state'i (tracked_signals) kullanabiliriz, key'ler zaten TF içeriyor.
                      track_key_vr = f"track_{sel_sym_vr}_{sel_tf_vr}"
                      is_tracked_vr = track_key_vr in st.session_state.tracked_signals
                      # ... (Takip butonu mantığı aynı) ...
                      b1v, b2v, b3v = st.columns([1,1,1])
                      # ... (Kayıt/İndir butonları mantığı aynı, sadece key'leri farklılaştır) ...

                 else:
                      st.warning(f"{sel_sym_vr} ({sel_tf_vr}) için hacim detay verisi bulunamadı.")


    # --- Takip Edilen Sinyaller (Aynı kaldı) ---
    st.markdown("---"); st.markdown("### 📌 Takip Edilen Sinyaller")
    # ... (Gösterim aynı) ...

    # --- Özet Metrikler ve Kayıtlı Tahminler (Aynı kaldı) ---
    st.markdown("---"); cols_summary = st.columns(4)
    # ... (Gösterim aynı) ...
    with st.expander("💾 Kayıtlı Tahminler (Arşiv)"):
        # ... (Gösterim aynı) ...

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
