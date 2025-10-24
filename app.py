# app.py
# Streamlit MEXC contract sinyal uygulaması - (v3 - KeyError Düzeltmesi + Piyasa Analizi)

import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import requests
from datetime import datetime, timedelta # timedelta eklendi
import ai_engine
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
DEFAULT_WEIGHTS = {'ema':25,'macd':20,'rsi':15,'bb':10,'adx':7,'vol':10,'funding':30,'nw':8}
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
.stProgress > div > div > div > div { background-image: linear-gradient(to right, #00b09b , #96c93d); } /* Progress bar rengi */
.market-analysis { background-color: #0f172a; padding: 10px; border-radius: 8px; margin-bottom: 15px; border: 1px solid #1e293b; }
.market-analysis-title { font-weight: bold; margin-bottom: 5px; color: #cbd5e1; }
.market-analysis-content { font-size: 0.9em; color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ---------------- API Helpers (Cache süreleri ayarlandı) ----------------
@st.cache_data(ttl=timedelta(hours=1)) # Sembol listesini 1 saat cache'le
def fetch_all_contract_symbols():
    url = f"{CONTRACT_BASE}/contract/detail"
    try:
        j = fetch_json(url)
        data = j.get('data', [])
        symbols = [item['symbol'].replace('_USDT', 'USDT') for item in data if isinstance(item, dict) and item.get('symbol', '').endswith('_USDT')]
        logging.info(f"{len(symbols)} adet MEXC vadeli işlem sembolü çekildi.")
        return sorted(list(set(symbols)))
    except Exception as e:
        logging.error(f"Tüm MEXC sembolleri çekilemedi: {e}")
        st.error(f"MEXC sembol listesi alınamadı: {e}")
        return ["BTCUSDT", "ETHUSDT"]

def fetch_json(url, params=None, timeout=15): # Timeout artırıldı
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        logging.warning(f"API isteği zaman aşımına uğradı: {url}")
        st.toast(f"API isteği zaman aşımına uğradı: {url.split('/')[-1]}", icon="⏳")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"API isteği başarısız: {url} - Hata: {e}")
        # st.toast(f"API isteği hatası: {e}", icon="🚨") # Çok fazla toast olabilir
        return None

def fetch_contract_ticker():
    url = f"{CONTRACT_BASE}/contract/ticker"
    j = fetch_json(url)
    return j.get('data', []) if j else []

@st.cache_data(ttl=timedelta(minutes=1)) # Hacim verisini 1 dakika cache'le
def get_top_contracts_by_volume(limit=200):
    # ... (Fonksiyon içeriği aynı) ...
    data = fetch_contract_ticker()
    if not data: return []
    def vol(x):
        try: return float(x.get('volume24') or x.get('amount24') or 0)
        except (ValueError, TypeError): return 0
    items = sorted(data, key=vol, reverse=True)
    syms = [it.get('symbol') for it in items[:limit]]
    return [s.replace('_USDT','USDT') for s in syms if s and s.endswith('_USDT')]

def mexc_symbol_from(symbol: str) -> str:
    # ... (Fonksiyon içeriği aynı) ...
    s = symbol.strip().upper()
    if '_' in s: return s
    if s.endswith('USDT'): return s[:-4] + "_USDT"
    logging.warning(f"Beklenmeyen sembol formatı: {symbol}.")
    return s

@st.cache_data(ttl=timedelta(seconds=30)) # Kline verisini 30 saniye cache'le
def fetch_contract_klines(symbol_mexc, interval_mexc):
    # ... (Hata yönetimi iyileştirildi) ...
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    j = fetch_json(url, params={'interval': interval_mexc})
    if not j: return pd.DataFrame()
    d = j.get('data') or {}
    times = d.get('time', [])
    if not times: return pd.DataFrame()
    try:
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(d.get('time'), unit='s'),
            'open': pd.to_numeric(d.get('open'), errors='coerce'),
            'high': pd.to_numeric(d.get('high'), errors='coerce'),
            'low': pd.to_numeric(d.get('low'), errors='coerce'),
            'close': pd.to_numeric(d.get('close'), errors='coerce'),
            'volume': pd.to_numeric(d.get('vol'), errors='coerce')
        })
        df = df.dropna()
        if len(df) < 50: # Yetersiz veri kontrolü
             logging.warning(f"fetch_contract_klines yetersiz veri döndü: {symbol_mexc} - {interval_mexc} ({len(df)} satır)")
        return df
    except Exception as e:
        logging.error(f"Kline verisi işlenirken hata ({symbol_mexc}, {interval_mexc}): {e}")
        return pd.DataFrame()


@st.cache_data(ttl=timedelta(minutes=1)) # Funding rate 1 dakika cache'le
def fetch_contract_funding_rate(symbol_mexc):
    # ... (Fonksiyon içeriği aynı) ...
    url = f"{CONTRACT_BASE}/contract/funding_rate/{symbol_mexc}"
    j = fetch_json(url)
    if not j: return {'fundingRate': 0.0}
    data = j.get('data') or {}
    try: return {'fundingRate': float(data.get('fundingRate') or 0)}
    except (ValueError, TypeError): return {'fundingRate': 0.0}

# --------------- Indicators & Scoring (Fonksiyonlar aynı kaldı)----------------
# ... (compute_indicators, nw_smooth, label_from_score, score_signals) ...
# Bu fonksiyonlar önceki yanıttaki halleriyle kullanılacak.

# ---------------- Scan Engine (Cache kapatıldı, progress bar eklendi) ----------------
# @st.cache_data(ttl=120) # API anahtarı değişince sorun olmaması için cache kapalı
def run_scan(symbols_to_scan, timeframes, weights, thresholds, gemini_api_key):
    # ... (İçerik büyük ölçüde aynı, progress bar güncellemeleri eklendi) ...
    results = []
    total_symbols = len(symbols_to_scan)
    # Sidebar'da progress bar daha iyi görünebilir
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
            if df is None or df.empty or len(df) < 50:
                # logging.warning(f"Yetersiz kline verisi (run_scan): {sym} - {tf}")
                continue
            df_ind = compute_indicators(df)
            if df_ind is None or df_ind.empty or len(df_ind) < 3:
                # logging.warning(f"İndikatör hesaplama hatası (run_scan): {sym} - {tf}")
                continue

            latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]

            score, per_scores, reasons = score_signals(latest, prev, funding, weights)
            label = label_from_score(score, thresholds)

            indicators_snapshot = {
                'symbol': sym, 'timeframe': tf, 'scan_mode': scan_mode,
                'score': int(score), 'price': float(latest['close']),
                'rsi14': latest.get('rsi14'), 'macd_hist': latest.get('macd_hist'),
                'vol_osc': latest.get('vol_osc'), 'atr14': latest.get('atr14'),
                'nw_slope': latest.get('nw_slope'), 'bb_upper': latest.get('bb_upper'),
                'bb_lower': latest.get('bb_lower'), 'funding_rate': funding.get('fundingRate')
            }
            # None veya NaN değerleri temizle
            indicators_snapshot = {k: v for k, v in indicators_snapshot.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}

            try:
                # API anahtarı boş string ise None gönder
                ai_analysis = ai_engine.get_ai_prediction(indicators_snapshot, api_key=(gemini_api_key if gemini_api_key else None))
            except Exception as e:
                logging.error(f"AI Analiz hatası ({sym}, {tf}): {e}")
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

        # Eski sayımlar
        entry['buy_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') in ['AL', 'GÜÇLÜ AL'])
        entry['strong_buy_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') == 'GÜÇLÜ AL')
        entry['sell_count'] = sum(1 for d in current_tf_results.values() if d and d.get('label') in ['SAT', 'GÜÇLÜ SAT'])

        results.append(entry)

    progress_bar_area.empty() # Tarama bitince barı kaldır
    return pd.DataFrame(results)

# ------------- Market Analysis Functions --------------
@st.cache_data(ttl=timedelta(minutes=30)) # 30 dakikada bir tahmin al
def get_market_analysis(api_key, period="current"):
    """Gemini kullanarak piyasa analizi alır (current veya weekly)."""
    if not api_key or not ai_engine.GEMINI_AVAILABLE:
        return None, None # Hem analiz hem başlık için None döndür

    period_prompt = "şu anki (çok kısa vadeli)" if period == "current" else "önümüzdeki hafta için (orta vadeli)"
    analysis_type = "Mevcut Piyasa Duyarlılığı" if period == "current" else "Haftalık Piyasa Görünümü"
    logging.info(f"{analysis_type} isteniyor...")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Sen deneyimli bir kripto para piyasa analistisin. Genel piyasa koşullarını (BTC fiyat hareketi, dominans, major altcoin trendleri, hacim, açık pozisyonlar (OI), fonlama oranları, global haberler, korku/açgözlülük endeksi gibi faktörleri dikkate alarak) analiz et.

        {period_prompt} genel piyasa yönü tahminini yap.

        Olası Tahminler:
        - GÜÇLÜ YÜKSELİŞ (Boğa Piyasası)
        - YÜKSELİŞ (Hafif Pozitif)
        - NÖTR/KARARSIZ (Yatay/Belirsiz)
        - DÜŞÜŞ (Hafif Negatif)
        - GÜÇLÜ DÜŞÜŞ (Ayı Piyasası)

        Cevabını SADECE bu tahminlerden biri olarak ver ve 1-2 cümlelik kısa bir gerekçe ekle.
        Örnek: YÜKSELİŞ - BTC'nin EMA direncini test etmesi ve OI'nin artması kısa vadeli pozitifliğe işaret ediyor.
        """
        response = model.generate_content(prompt, request_options={'timeout': 120}) # Timeout artırıldı
        logging.info(f"{analysis_type} alındı.")
        return analysis_type, response.text.strip()
    except Exception as e:
        logging.error(f"{analysis_type} alınamadı: {e}")
        return analysis_type, f"Tahmin alınamadı: {e}"

# ------------- GÜVENLİ TradingView GÖMME FONKSİYONU (Değişiklik Yok) ------------
# ... (show_tradingview fonksiyonu aynı kalacak) ...
def show_tradingview(symbol: str, interval_tv: str, height: int = 480):
    uid = f"tv_widget_{symbol.replace('/','_')}_{interval_tv}"
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:{height}px; width:100%;">
      <div id="{uid}" style="height:100%; width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      (function() {{
        try {{
          new TradingView.widget({{
            "container_id": "{uid}",
            "symbol": "BINANCE:{symbol}", // TradingView için genellikle BINANCE kullanılır
            "interval": "{interval_tv}",
            "autosize": true,
            "timezone": "Europe/Istanbul",
            "theme": "dark",
            "style": "1",
            "locale": "tr",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "hide_side_toolbar": false,
            "hideideas": true
          }});
        }} catch(e) {{
          var el = document.getElementById("{uid}");
          if(el) el.innerHTML = "<div style='color:#f66;padding:10px;'>Grafik yüklenemedi: "+e.toString()+"</div>";
        }}
      }})();
      </script>
    </div>
    """
    components.html(tradingview_html, height=height, scrolling=False)

# ------------------- ANA UYGULAMA AKIŞI -------------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli (Hibrit AI)")

# --- Piyasa Analizi Alanı ---
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", help="Gelişmiş AI analizi ve Piyasa Tahmini için.")

# Analizleri al ve göster
if gemini_api_key_ui:
    analysis_col1, analysis_col2 = st.columns(2)
    with analysis_col1:
        current_title, current_analysis = get_market_analysis(gemini_api_key_ui, period="current")
        if current_title and current_analysis:
            st.markdown(f"""
            <div class="market-analysis">
                <div class="market-analysis-title">{current_title} ⏱️</div>
                <div class="market-analysis-content">{current_analysis}</div>
            </div>
            """, unsafe_allow_html=True)
    with analysis_col2:
        weekly_title, weekly_analysis = get_market_analysis(gemini_api_key_ui, period="weekly")
        if weekly_title and weekly_analysis:
             st.markdown(f"""
            <div class="market-analysis">
                <div class="market-analysis-title">{weekly_title} 📅</div>
                <div class="market-analysis-content">{weekly_analysis}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
elif st.sidebar.button("Piyasa Analizi için Anahtar Gerekli", help="Günlük ve Haftalık piyasa analizi için Gemini API anahtarını girin."):
    pass # Buton sadece bilgi amaçlı

# --- Sidebar Ayarları ---
st.sidebar.header("Tarama Ayarları")

# Sembol Seçimi
all_symbols_list = fetch_all_contract_symbols()
mode = st.sidebar.selectbox("Sembol Kaynağı", ["Top Hacim (Max 200)","Özel Liste Seç"])

symbols_to_scan_ui = []
if mode == "Özel Liste Seç":
    # Arama yapılabilen çoklu seçim kutusu
    selected_symbols_ui = st.sidebar.multiselect("Taramak İstediğiniz Coinleri Seçin (Arayabilirsiniz)",
                                                 options=all_symbols_list,
                                                 default=["BTCUSDT", "ETHUSDT"])
    symbols_to_scan_ui = selected_symbols_ui
else: # Top Hacim
    symbols_by_volume_list = get_top_contracts_by_volume(200)
    top_n_ui = st.sidebar.slider("İlk N Coin Taransın", min_value=5, max_value=len(symbols_by_volume_list), value=min(50, len(symbols_by_volume_list)))
    symbols_to_scan_ui = symbols_by_volume_list[:top_n_ui]

if not symbols_to_scan_ui:
    st.sidebar.warning("Taranacak sembol seçilmedi.")
    st.stop()

# Zaman Dilimleri
timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS)
if not timeframes_ui:
     st.sidebar.warning("Lütfen en az bir zaman dilimi seçin.")
     st.stop()


# Algoritma Ayarları (Expander içinde)
with st.sidebar.expander("Sistem Algoritması Ayarları"):
    st.markdown("**Sinyal Mantığı:**")
    st.caption("""
    - **RSI:** Aşırı alım/satım (70/30) ve trend (50).
    - **MACD:** Histogram crossover ve pozitif/negatif bölge.
    - **NW Slope:** Ana trend yönü (en önemli).
    - **Bollinger:** Bant dışına taşma (reversal sinyali).
    - **Hacim:** Trend onayı.
    - **Funding:** Aşırı oranlar (reversal sinyali).
    """)
    st.markdown("**Ağırlıklar (Puanlama İçin):**")
    # ... (Ağırlık inputları) ...
    w_ema = st.number_input("EMA", value=DEFAULT_WEIGHTS['ema'], key="w_ema")
    w_macd = st.number_input("MACD", value=DEFAULT_WEIGHTS['macd'], key="w_macd")
    w_rsi = st.number_input("RSI", value=DEFAULT_WEIGHTS['rsi'], key="w_rsi")
    w_bb = st.number_input("BB", value=DEFAULT_WEIGHTS['bb'], key="w_bb")
    # w_adx = st.number_input("ADX", value=DEFAULT_WEIGHTS['adx'], key="w_adx") # ADX kullanılmıyor gibi
    w_vol = st.number_input("VOL", value=DEFAULT_WEIGHTS['vol'], key="w_vol")
    w_funding = st.number_input("Funding", value=DEFAULT_WEIGHTS['funding'], key="w_funding")
    w_nw = st.number_input("NW slope", value=DEFAULT_WEIGHTS['nw'], key="w_nw")
    weights_ui = {'ema':w_ema,'macd':w_macd,'rsi':w_rsi,'bb':w_bb, 'vol':w_vol,'funding':w_funding,'nw':w_nw}

    st.markdown("**Sinyal Eşikleri (Etiketleme İçin):**")
    # ... (Eşik inputları) ...
    strong_buy_t = st.slider("GÜÇLÜ AL ≥", 10, 100, 60, key="t_sb")
    buy_t = st.slider("AL ≥", 0, 80, 20, key="t_b")
    sell_t = st.slider("SAT ≤", -80, 0, -20, key="t_s")
    strong_sell_t = st.slider("GÜÇLÜ SAT ≤", -100, -10, -60, key="t_ss")
    thresholds_ui = (strong_buy_t, buy_t, sell_t, strong_sell_t)

# --- Session State Başlatma ---
if 'scan_results' not in st.session_state: st.session_state.scan_results = pd.DataFrame()
if 'selected_symbol' not in st.session_state: st.session_state.selected_symbol = None
if 'selected_tf' not in st.session_state: st.session_state.selected_tf = DEFAULT_TFS[0]
if 'tracked_signals' not in st.session_state: st.session_state.tracked_signals = {}
if 'last_scan_time' not in st.session_state: st.session_state.last_scan_time = None

# --- Tarama Butonu ---
scan = st.sidebar.button("🔍 Tara / Yenile")

if scan:
    st.session_state.scan_results = run_scan(symbols_to_scan_ui, timeframes_ui, weights_ui, thresholds_ui, gemini_api_key_ui)
    st.session_state.last_scan_time = datetime.now()
    st.session_state.selected_symbol = None # Taramadan sonra seçimi sıfırla

# --- Sonuçları Göster ---
df_results = st.session_state.scan_results
if st.session_state.last_scan_time:
    st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%Y-%m-%d %H:%M:%S')}")

if df_results is None or df_results.empty:
    st.info("Henüz tarama yapılmadı veya seçili kriterlere uygun coin bulunamadı.")
else:
    # AI Listesini Hazırla
    ai_list_display = []
    # ... (ai_list oluşturma mantığı öncekiyle aynı) ...
    for _, row in df_results.iterrows():
        best_tf = row.get('best_timeframe')
        details = row.get('details', {}) or {}
        snapshot = details.get(best_tf) if best_tf and details else None
        if not snapshot: continue
        ai_analysis = snapshot.get('ai_analysis')
        if not ai_analysis: continue
        ai_list_display.append({
            'symbol': row['symbol'], 'best_tf': best_tf, 'price': snapshot.get('price'),
            'ai_signal': ai_analysis.get('signal', 'NEUTRAL'), 'ai_confidence': ai_analysis.get('confidence', 0),
            'ai_text': ai_analysis.get('explanation', 'Açıklama yok.'), 'target_info': ai_analysis,
            'algo_score': snapshot.get('score'), 'algo_label': snapshot.get('label'),
            'per_scores': snapshot.get('per_scores'), 'reasons': snapshot.get('reasons', [])
        })

    if not ai_list_display:
        st.warning("Tarama tamamlandı ancak gösterilecek geçerli AI sinyali bulunamadı.")
        st.stop() # Gösterilecek bir şey yoksa dur

    ai_df_display = pd.DataFrame(ai_list_display)

    # Layout
    left, right = st.columns([1.6, 2.4])

    with left:
        st.markdown("### 🔎 AI Sinyal Listesi")
        filter_signal_ui = st.selectbox("Sinyal Türü", ["All","LONG","SHORT","NEUTRAL", "ERROR"], index=0, key="signal_filter")
        min_confidence_ui = st.slider("AI Minimum Güven (%)", 0, 100, 30, step=5, key="conf_filter")

        filtered_display = ai_df_display.copy()

        # --- KeyError Düzeltmesi ---
        if not filtered_display.empty:
            if filter_signal_ui != "All":
                # Filtrelemeden önce sütunun var olduğundan emin ol
                if 'ai_signal' in filtered_display.columns:
                     filtered_display = filtered_display[filtered_display['ai_signal'] == filter_signal_ui]
                else:
                     st.warning("`ai_signal` sütunu bulunamadı, filtreleme yapılamıyor.")
                     filtered_display = pd.DataFrame() # Boş DataFrame göster

            if 'ai_confidence' in filtered_display.columns:
                 filtered_display = filtered_display[filtered_display['ai_confidence'] >= min_confidence_ui]
            else:
                 st.warning("`ai_confidence` sütunu bulunamadı, filtreleme yapılamıyor.")
                 filtered_display = pd.DataFrame() # Boş DataFrame göster

            filtered_display = filtered_display.sort_values(by='ai_confidence', ascending=False)
        # --- KeyError Düzeltmesi Sonu ---


        st.caption(f"{len(filtered_display)} sinyal bulundu.")

        # Sinyal listesi gösterimi
        # ... (Önceki for döngüsü ve button mantığı aynı) ...
        MAX_SIGNALS_TO_SHOW = 150 # Çok fazla sinyal varsa yavaşlamayı önle
        for _, r in filtered_display.head(MAX_SIGNALS_TO_SHOW).iterrows():
            emoji = "⚪"
            if r['ai_signal']=='LONG': emoji='🚀'
            elif r['ai_signal']=='SHORT': emoji='🔻'
            elif r['ai_signal']=='ERROR': emoji='⚠️'

            cols = st.columns([0.6,2,1])
            cols[0].markdown(f"<div style='font-size:20px'>{emoji}</div>", unsafe_allow_html=True)
            algo_info = f"Algo: {r.get('algo_label','N/A')} ({r.get('algo_score','N/A')})"
            cols[1].markdown(f"**{r['symbol']}** • {r['best_tf']} \nAI: **{r['ai_signal']}** (%{r['ai_confidence']}) <span style='font-size: 0.8em; color: grey;'>{algo_info}</span>", unsafe_allow_html=True)
            if cols[2].button("Detay", key=f"det_{r['symbol']}_{r['best_tf']}"): # Key'e TF ekleyerek benzersiz yap
                st.session_state.selected_symbol = r['symbol']
                st.session_state.selected_tf = r['best_tf']
                st.experimental_rerun() # Detaya tıklayınca sağ tarafı hemen güncelle

    with right:
        st.markdown("### 📈 Seçili Coin Detayı")
        sel_sym = st.session_state.selected_symbol
        sel_tf_val = st.session_state.selected_tf

        # Başlangıçta veya tarama sonrası ilk coin'i seç
        if sel_sym is None and not filtered_display.empty:
            sel_sym = filtered_display.iloc[0]['symbol']
            sel_tf_val = filtered_display.iloc[0]['best_tf']
            st.session_state.selected_symbol = sel_sym # State'i güncelle
            st.session_state.selected_tf = sel_tf_val  # State'i güncelle

        if sel_sym is None:
            st.write("Listeden bir coin seçin veya tarama yapın.")
        else:
            st.markdown(f"**{sel_sym}** • TF: **{sel_tf_val}**")
            interval_tv_val = TV_INTERVAL_MAP.get(sel_tf_val, '60')

            show_tradingview(sel_sym, interval_tv_val, height=400)

            # Seçili coin için doğru veriyi bul
            row_data = next((x for x in ai_list_display if x['symbol']==sel_sym and x['best_tf'] == sel_tf_val), None)
            # Eğer best_tf ile bulunamazsa, sadece sembole göre ara (ilk bulduğunu al) - nadir durum
            if row_data is None:
                 row_data = next((x for x in ai_list_display if x['symbol']==sel_sym), None)


            if row_data:
                st.markdown("#### 🧠 AI Analizi ve Ticaret Planı")
                st.markdown(row_data['ai_text'])

                ti_data = row_data['target_info']
                entry_val = ti_data.get('entry')
                stop_val = ti_data.get('stop_loss')
                target_val = ti_data.get('take_profit')

                if entry_val is not None and stop_val is not None and target_val is not None:
                    c1, c2, c3 = st.columns(3)
                    # Sayısal değerleri formatla
                    entry_str = f"{entry_val:.{8 if entry_val < 1 else 5}f}"
                    stop_str = f"{stop_val:.{8 if stop_val < 1 else 5}f}"
                    target_str = f"{target_val:.{8 if target_val < 1 else 5}f}"
                    delta_stop = f"{((stop_val-entry_val)/entry_val*100):.2f}%" if entry_val else "N/A"
                    delta_target = f"{((target_val-entry_val)/entry_val*100):.2f}%" if entry_val else "N/A"

                    c1.metric("Giriş (Entry)", entry_str)
                    c2.metric("Stop Loss", stop_str, delta=delta_stop, delta_color="inverse")
                    c3.metric("Hedef (Target)", target_str, delta=delta_target)
                elif entry_val is not None:
                     st.metric("Fiyat", f"{entry_val:.{8 if entry_val < 1 else 5}f}")

                # --- Sinyal Takip Butonu ---
                track_key = f"track_{sel_sym}_{sel_tf_val}" # Key'e TF eklendi
                is_tracked = track_key in st.session_state.tracked_signals
                track_button_label = "❌ Takipten Çıkar" if is_tracked else "📌 Sinyali Takip Et"
                if st.button(track_button_label, key=f"track_btn_{track_key}"):
                    if is_tracked:
                        del st.session_state.tracked_signals[track_key]
                        st.toast(f"{sel_sym} ({sel_tf_val}) takipten çıkarıldı.", icon="🗑️")
                    else:
                        st.session_state.tracked_signals[track_key] = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                            'symbol': sel_sym, 'tf': sel_tf_val,
                            'signal': row_data['ai_signal'], 'confidence': row_data['ai_confidence'],
                            'entry': entry_val, 'stop': stop_val, 'target': target_val,
                        }
                        st.toast(f"{sel_sym} ({sel_tf_val}) takibe alındı!", icon="📌")
                    st.experimental_rerun()

                # Kayıt/İndirme Butonları
                b1, b2, b3 = st.columns([1,1,1])
                # ... (Buton mantığı aynı kaldı, outcome eklendi) ...
                if b1.button("✅ Tahmin Başarılı", key=f"success_{track_key}"):
                     rec = {'symbol': sel_sym, 'tf': row_data['best_tf'], 'entry': entry_val, 'stop': stop_val, 'target': target_val,
                           'price_at_mark': row_data['price'], 'ai_signal': row_data['ai_signal'], 'ai_confidence': row_data['ai_confidence'],
                           'outcome': 'Success', 'timestamp': datetime.utcnow().isoformat()}
                     ok = ai_engine.save_record(rec); st.toast("Başarılı kaydedildi." if ok else "Kayıt hatası!")
                if b2.button("❌ Tahmin Başarısız", key=f"fail_{track_key}"):
                     rec = {'symbol': sel_sym, 'tf': row_data['best_tf'], 'entry': entry_val, 'stop': stop_val, 'target': target_val,
                           'price_at_mark': row_data['price'], 'ai_signal': row_data['ai_signal'], 'ai_confidence': row_data['ai_confidence'],
                           'outcome': 'Failure', 'timestamp': datetime.utcnow().isoformat()}
                     ok = ai_engine.save_record(rec); st.toast("Başarısız kaydedildi." if ok else "Kayıt hatası!")
                if b3.button("📥 Analizi İndir", key=f"dl_{track_key}"):
                     st.download_button("JSON İndir", data=json.dumps(row_data, indent=2, ensure_ascii=False), file_name=f"{sel_sym}_{sel_tf_val}_signal.json")


                # Algoritma Puan Katkıları
                with st.expander("Algoritma Puan Katkıları (Eski Sistem)"):
                    # ... (Grafik/Tablo gösterimi aynı kaldı) ...
                    per = row_data.get('per_scores', {})
                    if per and PLOTLY_AVAILABLE:
                        dfp = pd.DataFrame([{'indicator':k,'points':v} for k,v in per.items()])
                        fig = px.bar(dfp.sort_values('points'), x='points', y='indicator', orientation='h', color='points', color_continuous_scale='RdYlGn')
                        fig.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10), template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    elif per:
                        st.table(pd.DataFrame([{'indicator':k,'points':v} for k,v in per.items()]).set_index('indicator'))
                    else: st.write("Algoritma skor verisi yok.")
            else:
                 st.warning(f"{sel_sym} ({sel_tf_val}) için detay verisi bulunamadı.")


    # --- Takip Edilen Sinyaller ---
    st.markdown("---")
    st.markdown("### 📌 Takip Edilen Sinyaller")
    if st.session_state.tracked_signals:
        tracked_list = list(st.session_state.tracked_signals.values())
        tracked_df = pd.DataFrame(tracked_list)
        # Sütunları yeniden sırala ve formatla
        tracked_df_display = tracked_df[['timestamp', 'symbol', 'tf', 'signal', 'confidence', 'entry', 'stop', 'target']].sort_values(by='timestamp', ascending=False)
        st.dataframe(tracked_df_display.style.format({ # Sayısal formatlama
             'confidence': "{:.0f}%",
             'entry': "{:.5f}",
             'stop': "{:.5f}",
             'target': "{:.5f}"
        }), use_container_width=True)
    else:
        st.info("Henüz takip edilen sinyal yok.")

    # --- Özet Metrikler ve Kayıtlı Tahminler ---
    st.markdown("---")
    cols_summary = st.columns(4)
    cols_summary[0].metric("Taranan Coin", f"{len(df_results)}")
    # Güven eşiği 30% olarak ayarlandı
    valid_ai_df = ai_df_display[ai_df_display['ai_confidence'] >= 30] if not ai_df_display.empty else ai_df_display
    cols_summary[1].metric("LONG Sinyal (>%30)", f"{len(valid_ai_df[valid_ai_df['ai_signal'] == 'LONG'])}" if not valid_ai_df.empty else 0)
    cols_summary[2].metric("SHORT Sinyal (>%30)", f"{len(valid_ai_df[valid_ai_df['ai_signal'] == 'SHORT'])}" if not valid_ai_df.empty else 0)
    cols_summary[3].metric("Kayıtlı Tahmin", f"{len(ai_engine.load_records())}")

    with st.expander("💾 Kayıtlı Tahminler (Arşiv)"):
        # ... (Arşiv gösterimi aynı kaldı) ...
        recs = ai_engine.load_records()
        if recs:
            st.dataframe(pd.DataFrame(recs).sort_values(by='timestamp', ascending=False), use_container_width=True)
        else: st.write("Henüz kayıtlı tahmin yok.")

st.caption("⚠️ Uyarı: Bu araç yalnızca eğitim ve deneme amaçlıdır. Yatırım tavsiyesi değildir.")
