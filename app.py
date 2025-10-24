# app.py
# Streamlit MEXC contract sinyal uygulaması - (v3.2 - AttributeError Düzeltmesi)

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

# --- Session State Başlatma (EN ÜSTE TAŞINDI) ---
if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'selected_symbol' not in st.session_state: st.session_state.selected_symbol = None
if 'selected_tf' not in st.session_state: st.session_state.selected_tf = '15m' # Varsayılan TF
if 'tracked_signals' not in st.session_state: st.session_state.tracked_signals = {}
if 'last_scan_time' not in st.session_state: st.session_state.last_scan_time = None
# --- Session State Başlatma Sonu ---


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
# ... (CSS aynı kaldı) ...
st.markdown("""
<style>
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
@st.cache_data(ttl=timedelta(hours=1))
def fetch_all_contract_symbols():
    # ... (İçerik aynı) ...
    url = f"{CONTRACT_BASE}/contract/detail"; j = fetch_json(url)
    if not j: return ["BTCUSDT", "ETHUSDT"]
    data = j.get('data', [])
    symbols = [item['symbol'].replace('_USDT', 'USDT') for item in data if isinstance(item, dict) and item.get('symbol', '').endswith('_USDT')]
    logging.info(f"{len(symbols)} adet MEXC sembolü çekildi."); return sorted(list(set(symbols)))

def fetch_json(url, params=None, timeout=15):
    # ... (İçerik aynı) ...
    try: r = requests.get(url, params=params, timeout=timeout); r.raise_for_status(); return r.json()
    except requests.exceptions.Timeout: logging.warning(f"Zaman aşımı: {url}"); st.toast(f"Zaman aşımı: {url.split('/')[-1]}", icon="⏳"); return None
    except requests.exceptions.RequestException as e: logging.error(f"API hatası: {url} - {e}"); return None

@st.cache_data(ttl=timedelta(minutes=1))
def get_top_contracts_by_volume(limit=200):
    # ... (İçerik aynı) ...
    url = f"{CONTRACT_BASE}/contract/ticker"; j = fetch_json(url); data = j.get('data', []) if j else []
    if not data: return []
    def vol(x): try: return float(x.get('volume24') or x.get('amount24') or 0); except: return 0
    items = sorted(data, key=vol, reverse=True); syms = [it.get('symbol') for it in items[:limit]]
    return [s.replace('_USDT','USDT') for s in syms if s and s.endswith('_USDT')]

def mexc_symbol_from(symbol: str) -> str:
    # ... (İçerik aynı) ...
    s = symbol.strip().upper();
    if '_' in s: return s;
    if s.endswith('USDT'): return s[:-4] + "_USDT";
    logging.warning(f"Beklenmeyen format: {symbol}."); return s

@st.cache_data(ttl=timedelta(seconds=30))
def fetch_contract_klines(symbol_mexc, interval_mexc):
    # ... (İçerik aynı) ...
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
    # ... (İçerik aynı) ...
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"; j = fetch_json(url)
    if not j: return {'fundingRate': 0.0}; data = j.get('data') or {}
    try: return {'fundingRate': float(data.get('fundingRate') or 0)}
    except: return {'fundingRate': 0.0}

# --------------- Scan Engine (Error Handling Eklendi) ---------------
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key):
    """Ana tarama fonksiyonu - ai_engine'i kullanır ve hata yönetimi yapar."""
    results = []
    total_symbols = len(symbols_to_scan)
    progress_bar_area = st.sidebar.empty()
    progress_bar = progress_bar_area.progress(0, text="Tarama başlatılıyor...")

    for i, sym in enumerate(symbols_to_scan):
        progress_value = (i + 1) / total_symbols
        progress_text = f"Taranıyor: {sym} ({i+1}/{total_symbols})"
        progress_bar.progress(progress_value, text=progress_text)

        entry = {'symbol': sym, 'details': {}}
        best_ai_confidence = -1; best_tf = None
        mexc_sym = mexc_symbol_from(sym)
        if not mexc_sym.endswith("_USDT"): continue

        try: # Sembol bazında hata yakalama
            funding = fetch_contract_funding_rate(mexc_sym)
            current_tf_results = {}

            for tf in timeframes:
                interval = INTERVAL_MAP.get(tf)
                if interval is None: continue

                scan_mode = "Normal";
                if tf in SCALP_TFS: scan_mode = "Scalp"
                elif tf in SWING_TFS: scan_mode = "Swing"

                df = fetch_contract_klines(mexc_sym, interval)
                if df is None or df.empty or len(df) < 50: continue

                df_ind = ai_engine.compute_indicators(df)
                if df_ind is None or df_ind.empty or len(df_ind) < 3: continue

                latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]

                score, per_scores, reasons = ai_engine.score_signals(latest, prev, funding, weights)
                label = ai_engine.label_from_score(score, thresholds)

                indicators_snapshot = {
                    'symbol': sym, 'timeframe': tf, 'scan_mode': scan_mode,
                    'score': int(score), 'price': float(latest['close']),
                    'rsi14': latest.get('rsi14'), 'macd_hist': latest.get('macd_hist'),
                    'vol_osc': latest.get('vol_osc'), 'atr14': latest.get('atr14'),
                    'nw_slope': latest.get('nw_slope'), 'bb_upper': latest.get('bb_upper'),
                    'bb_lower': latest.get('bb_lower'), 'funding_rate': funding.get('fundingRate')
                }
                indicators_snapshot = {k: v for k, v in indicators_snapshot.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

                ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=(gemini_api_key if gemini_api_key else None))

                current_tf_results[tf] = {
                    'score': int(score), 'label': label, 'price': float(latest['close']),
                    'per_scores': per_scores, 'reasons': reasons,
                    'ai_analysis': ai_analysis
                }

                current_confidence = ai_analysis.get('confidence', 0) if ai_analysis.get('signal') not in ['NEUTRAL', 'ERROR'] else -1
                if current_confidence > best_ai_confidence:
                    best_ai_confidence = current_confidence; best_tf = tf

            entry['details'] = current_tf_results
            entry['best_timeframe'] = best_tf
            entry['best_score'] = int(best_ai_confidence) if best_ai_confidence >= 0 else 0
            entry['buy_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') in ['AL', 'GÜÇLÜ AL'])
            entry['strong_buy_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') == 'GÜÇLÜ AL')
            entry['sell_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') in ['SAT', 'GÜÇLÜ SAT'])
            results.append(entry)

        except Exception as e:
            logging.error(f"Tarama sırasında {sym} için hata oluştu: {e}", exc_info=True) # Tam traceback'i logla
            st.toast(f"{sym} taranırken hata oluştu: {e}", icon="🚨")
            # Hata durumunda bile boş bir entry ekleyebiliriz veya atlayabiliriz. Şimdilik atlıyoruz.
            continue # Sonraki sembole geç

    progress_bar_area.empty()
    if not results: # Eğer hiç başarılı sonuç yoksa
        logging.warning("Tarama tamamlandı ancak hiçbir sembol için sonuç üretilemedi.")
    return pd.DataFrame(results) # Boş olsa bile DataFrame döndür


# ------------- Market Analysis Functions (Aynı kaldı) --------------
@st.cache_data(ttl=timedelta(minutes=30))
def get_market_analysis(api_key, period="current"):
    # ... (İçerik aynı) ...
    if not api_key or not ai_engine.GEMINI_AVAILABLE: return None, None
    period_prompt = "şu anki (çok kısa vadeli)" if period == "current" else "önümüzdeki hafta için (orta vadeli)"; analysis_type = "Mevcut Piyasa Duyarlılığı" if period == "current" else "Haftalık Piyasa Görünümü"
    logging.info(f"{analysis_type} isteniyor...");
    try:
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-pro')
        prompt = f"""Sen deneyimli bir kripto para piyasa analistisin... {period_prompt} genel piyasa yönü tahminini yap... Olası Tahminler:... Cevabını SADECE... Örnek:..."""
        response = model.generate_content(prompt, request_options={'timeout': 120})
        logging.info(f"{analysis_type} alındı."); return analysis_type, response.text.strip()
    except Exception as e: logging.error(f"{analysis_type} alınamadı: {e}"); return analysis_type, f"Tahmin alınamadı: {e}"

# ------------- TradingView GÖMME FONKSİYONU (Aynı kaldı) ------------
# ... (show_tradingview fonksiyonu aynı kaldı) ...
def show_tradingview(symbol: str, interval_tv: str, height: int = 480):
    uid = f"tv_widget_{symbol.replace('/','_')}_{interval_tv}"
    tradingview_html = f"""...""" # HTML içerik aynı
    components.html(tradingview_html, height=height, scrolling=False)


# ------------------- ANA UYGULAMA AKIŞI -------------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli (Hibrit AI)")

# --- Piyasa Analizi Alanı ---
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", help="Gelişmiş AI analizi ve Piyasa Tahmini için.", key="api_key_input")
# ... (Piyasa analizi gösterimi aynı kaldı) ...
if gemini_api_key_ui:
    analysis_col1, analysis_col2 = st.columns(2)
    with analysis_col1:
        current_title, current_analysis = get_market_analysis(gemini_api_key_ui, period="current")
        if current_title and current_analysis: st.markdown(f"""<div class="market-analysis"><div class="market-analysis-title">{current_title} ⏱️</div><div class="market-analysis-content">{current_analysis}</div></div>""", unsafe_allow_html=True)
    with analysis_col2:
        weekly_title, weekly_analysis = get_market_analysis(gemini_api_key_ui, period="weekly")
        if weekly_title and weekly_analysis: st.markdown(f"""<div class="market-analysis"><div class="market-analysis-title">{weekly_title} 📅</div><div class="market-analysis-content">{weekly_analysis}</div></div>""", unsafe_allow_html=True)
    st.markdown("---")
# ...

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
with st.sidebar.expander("Sistem Algoritması Ayarları"): # Algoritma ayarları...
     weights_ui = {...}; thresholds_ui = (...) # Ağırlık/Eşik inputları aynı...


# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    spinner_msg = "Tarama çalışıyor (Algoritma Modu)..."
    if gemini_api_key_ui and ai_engine.GEMINI_AVAILABLE: spinner_msg = "Tarama çalışıyor (Gemini AI Modu)..."
    
    with st.spinner(spinner_msg):
        try:
             # --- HATA YÖNETİMİ EKLENDİ ---
             scan_start_time = time.time()
             st.session_state.scan_results = run_scan(symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui, gemini_api_key_ui)
             scan_duration = time.time() - scan_start_time
             logging.info(f"Tarama tamamlandı. Süre: {scan_duration:.2f} saniye. {len(st.session_state.scan_results)} sonuç bulundu.")
             st.session_state.last_scan_time = datetime.now()
             st.session_state.selected_symbol = None # Taramadan sonra seçimi sıfırla
             st.experimental_rerun() # Sonuçları hemen göstermek için
        except Exception as e:
             # run_scan içindeki hata yakalama buraya düşmemeli ama garanti olsun
             logging.error(f"Beklenmedik tarama hatası: {e}", exc_info=True)
             st.error(f"Tarama sırasında beklenmedik bir hata oluştu: {e}")
             st.session_state.scan_results = pd.DataFrame() # Hata durumunda boşalt


# --- Sonuçları Göster ---
# df_results'ı session_state'den al (EN ÜSTTE İNİTİALİZE EDİLDİ)
df_results = st.session_state.scan_results

if st.session_state.last_scan_time:
    st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Sonuç DataFrame'i var mı ve boş mu kontrolü
if df_results is None or df_results.empty:
    if st.session_state.last_scan_time: # Eğer tarama yapıldıysa ama sonuç yoksa
        st.warning("Tarama tamamlandı ancak seçili kriterlere uygun coin bulunamadı veya verilerde sorun oluştu.")
    else: # Henüz tarama yapılmadıysa
        st.info("Henüz tarama yapılmadı. Lütfen yan panelden ayarları yapılandırıp 'Tara / Yenile' butonuna basın.")
else:
    # --- AI Listesini Hazırla (DataFrame Boş Değilse) ---
    ai_list_display = []
    # ... (ai_list oluşturma mantığı aynı kaldı) ...
    for _, row in df_results.iterrows():
        best_tf = row.get('best_timeframe'); details = row.get('details', {}) or {}
        snapshot = details.get(best_tf) if best_tf and details else None
        if not snapshot: continue; ai_analysis = snapshot.get('ai_analysis')
        if not ai_analysis: continue
        ai_list_display.append({'symbol': row['symbol'], 'best_tf': best_tf, 'price': snapshot.get('price'), 'ai_signal': ai_analysis.get('signal', 'NEUTRAL'), 'ai_confidence': ai_analysis.get('confidence', 0), 'ai_text': ai_analysis.get('explanation', 'Açıklama yok.'), 'target_info': ai_analysis, 'algo_score': snapshot.get('score'), 'algo_label': snapshot.get('label'), 'per_scores': snapshot.get('per_scores'), 'reasons': snapshot.get('reasons', [])})


    if not ai_list_display:
        st.warning("Tarama sonuçları işlenemedi veya geçerli AI sinyali bulunamadı.")
        st.stop()

    ai_df_display = pd.DataFrame(ai_list_display)

    # Layout ve Sol Taraf (Filtreleme)
    left, right = st.columns([1.6, 2.4])
    with left:
        st.markdown("### 🔎 AI Sinyal Listesi")
        filter_signal_ui = st.selectbox("Sinyal Türü", ["All","LONG","SHORT","NEUTRAL", "ERROR"], index=0, key="signal_filter")
        min_confidence_ui = st.slider("AI Minimum Güven (%)", 0, 100, 30, step=5, key="conf_filter")

        filtered_display = ai_df_display.copy()

        # --- KeyError Düzeltmesi (Tekrar Kontrol) ---
        if not filtered_display.empty:
            if filter_signal_ui != "All":
                if 'ai_signal' in filtered_display.columns: filtered_display = filtered_display[filtered_display['ai_signal'] == filter_signal_ui]
                else: filtered_display = pd.DataFrame() # Boşalt
            if 'ai_confidence' in filtered_display.columns: filtered_display = filtered_display[filtered_display['ai_confidence'] >= min_confidence_ui]
            else: filtered_display = pd.DataFrame() # Boşalt
            if not filtered_display.empty: filtered_display = filtered_display.sort_values(by='ai_confidence', ascending=False)
        # --- Bitti ---

        st.caption(f"{len(filtered_display)} sinyal bulundu.")
        # ... (Sinyal listesi gösterimi aynı kaldı) ...
        MAX_SIGNALS_TO_SHOW = 150
        for _, r in filtered_display.head(MAX_SIGNALS_TO_SHOW).iterrows():
            emoji = "..."; algo_info = "..." # Emoji ve algo_info aynı
            cols = st.columns([0.6,2,1])
            # ... (cols içeriği aynı) ...
            if cols[2].button("Detay", key=f"det_{r['symbol']}_{r['best_tf']}"):
                st.session_state.selected_symbol = r['symbol']; st.session_state.selected_tf = r['best_tf']
                st.experimental_rerun()

    # Sağ Taraf (Detay Ekranı)
    with right:
        st.markdown("### 📈 Seçili Coin Detayı")
        sel_sym = st.session_state.selected_symbol
        sel_tf_val = st.session_state.selected_tf

        # Başlangıçta veya tarama sonrası ilk coin'i seç (Filtrelenmişten)
        if sel_sym is None and not filtered_display.empty:
            sel_sym = filtered_display.iloc[0]['symbol']
            sel_tf_val = filtered_display.iloc[0]['best_tf']
            st.session_state.selected_symbol = sel_sym
            st.session_state.selected_tf = sel_tf_val

        if sel_sym is None:
            st.write("Listeden bir coin seçin veya tarama yapın.")
        else:
            # Detay gösterimi... (TradingView, AI Analizi, Metrikler, Butonlar, Algoritma Puanları aynı kaldı) ...
            st.markdown(f"**{sel_sym}** • TF: **{sel_tf_val}**")
            interval_tv_val = TV_INTERVAL_MAP.get(sel_tf_val, '60')
            show_tradingview(sel_sym, interval_tv_val, height=400)
            row_data = next((x for x in ai_list_display if x['symbol']==sel_sym and x['best_tf'] == sel_tf_val), None)
            if row_data is None: row_data = next((x for x in ai_list_display if x['symbol']==sel_sym), None) # Fallback

            if row_data:
                # ... (AI Analizi, Metrikler, Takip Butonu, Kayıt/İndirme Butonları, Algoritma Puanları Expander'ı aynı kaldı) ...
                st.markdown("#### 🧠 AI Analizi..."); st.markdown(row_data['ai_text'])
                ti_data = row_data['target_info']; entry_val = ti_data.get('entry'); stop_val = ti_data.get('stop_loss'); target_val = ti_data.get('take_profit')
                if entry_val is not None and stop_val is not None and target_val is not None:
                     c1,c2,c3=st.columns(3); entry_str = f"{entry_val:.{8 if entry_val < 1 else 5}f}"; stop_str = f"{stop_val:.{8 if stop_val < 1 else 5}f}"; target_str = f"{target_val:.{8 if target_val < 1 else 5}f}"; delta_stop = f"{((stop_val-entry_val)/entry_val*100):.2f}%" if entry_val else "N/A"; delta_target = f"{((target_val-entry_val)/entry_val*100):.2f}%" if entry_val else "N/A"
                     c1.metric("Giriş", entry_str); c2.metric("Stop", stop_str, delta=delta_stop, delta_color="inverse"); c3.metric("Hedef", target_str, delta=delta_target)
                # ... Takip butonu, Kayıt/İndirme butonları ...
                with st.expander("Algoritma Puan Katkıları (Eski Sistem)"): # ... grafik/tablo ...
                      pass # İçerik aynı
            else: st.warning(f"{sel_sym} ({sel_tf_val}) için detay verisi bulunamadı.")


    # --- Takip Edilen Sinyaller (Aynı kaldı) ---
    st.markdown("---"); st.markdown("### 📌 Takip Edilen Sinyaller")
    # ... (Takip edilen sinyallerin gösterimi aynı kaldı) ...
    if st.session_state.tracked_signals: tracked_list = list(st.session_state.tracked_signals.values()); tracked_df = pd.DataFrame(tracked_list); tracked_df_display = tracked_df[...].sort_values(...); st.dataframe(tracked_df_display.style.format({...}), use_container_width=True)
    else: st.info("Henüz takip edilen sinyal yok.")

    # --- Özet Metrikler ve Kayıtlı Tahminler (Aynı kaldı) ---
    st.markdown("---"); cols_summary = st.columns(4)
    # ... (Metriklerin hesaplanması ve gösterimi aynı kaldı) ...
    cols_summary[0].metric("Taranan", f"{len(df_results)}"); #... diğer metrikler ...
    with st.expander("💾 Kayıtlı Tahminler (Arşiv)"):
        # ... (Arşiv gösterimi aynı kaldı) ...
        recs = ai_engine.load_records();
        if recs: st.dataframe(pd.DataFrame(recs).sort_values(by='timestamp', ascending=False), use_container_width=True)
        else: st.write("Henüz kayıtlı tahmin yok.")

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
