# app.py
# Profesyonel Sinyal Paneli (v6 - Stabil + Tüm Özellikler)

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
import math

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
SCALP_TFS = ['1m', '5m', '15m'] # Scalp için ağırlıklandırma
SWING_TFS = ['4h', '1d'] # Swing için

# ---------------- CSS ----------------
st.markdown("""
<style>
    /* ... (Önceki CSS stilleri eklenebilir) ... */
    [data-testid="stMetricValue"] { font-size: 24px; }
    [data-testid="stMetricLabel"] { font-size: 16px; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; } /* Sekme aralığı */
    .stTabs [data-baseweb="tab"] { background-color: #0E1117; border-radius: 8px 8px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ---------------- API Yardımcı Fonksiyonları (Sağlamlaştırıldı) ----------------
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
    logging.error("fetch_all_contract_symbols: Geçersiz veri.")
    return ["BTCUSDT", "ETHUSDT"] # Fallback

def fetch_json(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"API hatası: {url} - {e}")
        return None

@st.cache_data(ttl=timedelta(minutes=1))
def get_top_contracts_by_volume(limit=200):
    url = f"{CONTRACT_BASE}/contract/ticker"
    data = fetch_json(url)
    if not data or 'data' not in data or not isinstance(data['data'], list):
        logging.error("get_top_contracts_by_volume: Geçersiz veri.")
        return []
    def vol(x):
        try: return float(x.get('volume24') or x.get('amount24') or '0')
        except: return 0
    valid_items = [item for item in data['data'] if isinstance(item, dict)]
    items = sorted(valid_items, key=vol, reverse=True)
    syms = [it.get('symbol') for it in items[:limit] if it.get('symbol')]
    return [s.replace('_USDT','USDT') for s in syms if s.endswith('_USDT')]

def mexc_symbol_from(symbol: str) -> str:
    s = symbol.strip().upper();
    if not s: return ""
    if '_' in s: return s;
    if s.endswith('USDT'): return s[:-4] + "_USDT";
    return s + "_USDT"

@st.cache_data(ttl=timedelta(seconds=30))
def fetch_contract_klines(symbol_mexc, interval_mexc):
    url = f"{CONTRACT_BASE}/contract/kline/{symbol_mexc}"
    data = fetch_json(url, params={'interval': interval_mexc})
    if not data or 'data' not in data or not isinstance(data['data'], dict):
         logging.warning(f"Geçersiz kline verisi: {symbol_mexc} - {interval_mexc}")
         return pd.DataFrame()
    d = data['data']; times = d.get('time')
    if not isinstance(times, list) or not times:
         logging.warning(f"Kline 'time' verisi eksik: {symbol_mexc} - {interval_mexc}")
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
        df = df.dropna(subset=['timestamp', 'close']).reset_index(drop=True)
        if len(df) < 50: logging.warning(f"fetch_klines az veri: {symbol_mexc} ({len(df)})")
        return df
    except Exception as e:
        logging.error(f"Kline işleme hatası ({symbol_mexc}): {e}")
        return pd.DataFrame()

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

                # Scalp/Swing modu belirle
                scan_mode = "Normal"
                if tf in SCALP_TFS: scan_mode = "Scalp"
                elif tf in SWING_TFS: scan_mode = "Swing"

                df = fetch_contract_klines(mexc_sym, interval)
                min_bars_needed = 100 # İndikatörler için minimum veri
                if df is None or df.empty or len(df) < min_bars_needed:
                    logging.debug(f"Yetersiz kline ({sym}-{tf}): {len(df) if df is not None else 0}")
                    continue

                # --- 1. İndikatörleri Hesapla (8 ana indikatör) ---
                df_ind = ai_engine.compute_indicators(df)
                if df_ind is None or df_ind.empty or len(df_ind) < 2:
                    logging.warning(f"İndikatör hesaplanamadı: {sym}-{tf}")
                    continue

                latest = df_ind.iloc[-1]; prev = df_ind.iloc[-2]

                # --- 2. Algoritmik Puanlama ---
                algo_score, algo_label, algo_contributions = ai_engine.score_signals(latest, prev, weights, tf, SCALP_TFS)

                # --- 3. Gemini AI Analizi (API anahtarı varsa) ---
                gemini_analysis = None
                if gemini_api_key:
                    try:
                        # AI'a gönderilecek snapshot
                        indicators_snapshot = {
                            'symbol': sym, 'timeframe': tf, 'scan_mode': scan_mode,
                            'price': float(latest['close']),
                            'rsi': latest.get('rsi'), 'macd_hist': latest.get('macd_hist'),
                            'ema_cross_signal': 1 if latest.get('ema_short',0) > latest.get('ema_long',0) else -1,
                            'bb_percent': latest.get('bbp'), # Bollinger %B
                            'stoch_k': latest.get('stoch_k'), 'stoch_d': latest.get('stoch_d'),
                            'adx': latest.get('adx'), 'dmi_plus': latest.get('dmi_plus'), 'dmi_minus': latest.get('dmi_minus'),
                            'volume_spike': latest.get('volume_spike'),
                            'atr_percent': latest.get('atr_percent'),
                            'algo_score': algo_score # Kendi skorunu da gönder
                        }
                        indicators_snapshot = {k: (v if not (isinstance(v, float) and np.isnan(v)) else None) for k, v in indicators_snapshot.items()}
                        
                        gemini_analysis = ai_engine.get_gemini_analysis(indicators_snapshot, api_key=gemini_api_key)
                    
                    except Exception as e:
                        logging.error(f"Gemini analizi hatası ({sym}-{tf}): {e}")
                        st.toast(f"{sym}-{tf} Gemini analizi başarısız.", icon="⚠️")
                        gemini_analysis = {"error": str(e)}

                # --- Sonuçları Birleştir ---
                results.append({
                    'symbol': sym, 'tf': tf,
                    'price': float(latest['close']),
                    'algo_score': algo_score,
                    'algo_label': algo_label,
                    'algo_contributions': algo_contributions, # Puan grafiği için
                    'gemini_analysis': gemini_analysis, # Gemini sonucu (veya None)
                    'timestamp': datetime.now()
                })

        except Exception as e:
            logging.error(f"Tarama sırasında {sym} için hata: {e}", exc_info=True)
            st.toast(f"{sym} taranırken hata: {e}", icon="🚨")
            continue

    try: progress_bar_area.empty()
    except: pass
    
    if not results: logging.warning("Tarama hiç sonuç üretmedi.")
    return pd.DataFrame(results) # Boş olsa bile DataFrame döndür


# ------------- Market Analysis Functions (Gemini Gerekli) --------------
@st.cache_data(ttl=timedelta(minutes=30))
def get_market_analysis(api_key):
    if not api_key or not ai_engine.GEMINI_AVAILABLE: return None
    try:
        logging.info("Genel piyasa analizi isteniyor...")
        genai.configure(api_key=api_key); model = genai.GenerativeModel('gemini-pro')
        prompt = """
        Sen bir usta kripto para piyasa analistisin. BTC fiyatı, dominans, hacim, fonlama oranları ve genel korku/açgözlülük endeksine bakarak, 
        bugünkü genel piyasa yönü tahminini (örn: YÜKSELİŞ, DÜŞÜŞ, NÖTR/KARARSIZ) ve 1-2 cümlelik kısa bir gerekçe belirt.
        Örnek: YÜKSELİŞ - BTC'nin desteği tutması ve fonlama oranlarının nötr olması kısa vadeli pozitifliğe işaret ediyor.
        """
        response = model.generate_content(prompt, request_options={'timeout': 120})
        logging.info("Genel piyasa analizi alındı.")
        return response.text.strip()
    except Exception as e:
        logging.error(f"Genel piyasa analizi alınamadı: {e}")
        return f"Piyasa analizi alınamadı: {str(e)[:50]}..."

# ------------- UI Yardımcı Fonksiyonları ------------
def show_tradingview(symbol: str, interval_tv: str):
    """TradingView widget'ını güvenli bir şekilde HTML bileşeni olarak basar."""
    # (Önceki yanıttaki güvenli st.components.v1.html kodu)
    uid = f"tv_widget_{symbol.replace('/','_')}_{interval_tv}"
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:450px; width:100%;">
      <div id="{uid}" style="height:100%; width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      (function() {{
        try {{ new TradingView.widget({{
            "container_id": "{uid}", "symbol": "BINANCE:{symbol}", "interval": "{interval_tv}",
            "autosize": true, "timezone": "Europe/Istanbul", "theme": "dark", "style": "1", "locale": "tr",
            "enable_publishing": false, "allow_symbol_change": true, "hide_side_toolbar": false
        }}); }} catch(e) {{ }}
      }})();
      </script>
    </div>
    """
    components.html(tradingview_html, height=450)

def plot_indicator_contributions(contributions: Dict[str, float]):
    """İndikatör katkılarını Plotly bar chart olarak çizer."""
    if not PLOTLY_AVAILABLE or not contributions:
        st.caption("Puan katkı detayı yok veya Plotly yüklü değil.")
        return

    # Sözlüğü DataFrame'e çevir
    df_plot = pd.DataFrame(list(contributions.items()), columns=['İndikatör', 'Puan Katkısı'])
    df_plot = df_plot.sort_values(by='Puan Katkısı', ascending=True)

    # Renkleri belirle (pozitif/negatif)
    colors = ['#00b09b' if x > 0 else '#f44336' for x in df_plot['Puan Katkısı']]

    fig = go.Figure(go.Bar(
        x=df_plot['Puan Katkısı'],
        y=df_plot['İndikatör'],
        orientation='h',
        marker_color=colors,
        text=df_plot['Puan Katkısı'],
        texttemplate='%{text:.0f}',
        textposition='outside'
    ))
    fig.update_layout(
        title="İndikatör Puan Katkıları",
        xaxis_title="Puan", yaxis_title="",
        height=300, margin=dict(l=40,r=40,t=40,b=40),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#e6eef6'
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ------------------- ANA UYGULAMA AKIŞI -------------------
st.title("🔥 MEXC Vadeli — Profesyonel Sinyal Paneli")

# --- 1. SOL: Ayarlar (Sidebar) ---
st.sidebar.header("Tarama Ayarları")
gemini_api_key_ui = st.sidebar.text_input("Gemini API Anahtarı (Opsiyonel)", type="password", key="api_key_input", help="AI Analizi ve Piyasa Yorumu için gereklidir.")

# Zaman Dilimleri (Hata düzeltmesi uygulandı)
timeframes_ui = st.sidebar.multiselect("Zaman Dilimleri", options=ALL_TFS, default=DEFAULT_TFS, key="timeframes_multiselect")
if not timeframes_ui: st.sidebar.warning("Lütfen en az bir zaman dilimi seçin."); st.stop()

# Sembol Kaynağı (Hata düzeltmesi uygulandı)
all_symbols_list = fetch_all_contract_symbols()
mode = st.sidebar.radio("Sembol Kaynağı", ["Top Hacim","Özel Liste"], key="mode_radio")
symbols_to_scan_ui = [];
if mode == "Özel Liste":
    selected_symbols_ui = st.sidebar.multiselect("Coinleri Seçin", options=all_symbols_list, default=["BTCUSDT", "ETHUSDT"])
    symbols_to_scan_ui = selected_symbols_ui
else: # Top Hacim
    symbols_by_volume_list = get_top_contracts_by_volume(200)
    if not symbols_by_volume_list:
        st.sidebar.error("MEXC hacim verisi alınamadı."); st.stop()
    else:
        max_symbols = len(symbols_by_volume_list); min_val_slider = 5; max_val_slider = max(min_val_slider, max_symbols)
        default_val_slider = max(min_val_slider, min(50, max_symbols))
        top_n_ui = st.sidebar.slider("İlk N Coin", min_value=min_val_slider, max_value=max_val_slider, value=default_val_slider)
        symbols_to_scan_ui = symbols_by_volume_list[:top_n_ui]
if not symbols_to_scan_ui: st.sidebar.error("Taranacak sembol seçilmedi!"); st.stop()

# Ağırlık Ayarları (İsteğe bağlı expander)
with st.sidebar.expander("Gelişmiş Ağırlık Ayarları"):
    st.caption("Scalp (kısa vade) veya Swing (uzun vade) ağırlıklarını ayarlayın.")
    weights_ui = {}
    weights_ui['rsi_weight'] = st.slider("RSI Ağırlığı (Scalp için önemli)", 0, 50, 25, help="Momentum gücü.")
    weights_ui['stoch_weight'] = st.slider("Stochastic Ağırlığı (Scalp için önemli)", 0, 50, 20, help="Hızlı momentum dönüşü.")
    weights_ui['macd_weight'] = st.slider("MACD Ağırlığı (Orta vade)", 0, 50, 20, help="Trend değişimi.")
    weights_ui['ema_cross_weight'] = st.slider("EMA Cross Ağırlığı", 0, 50, 15, help="Kısa/Uzun trend dengesi.")
    weights_ui['bb_weight'] = st.slider("Bollinger Ağırlığı", 0, 50, 10, help="Volatilite / Aşırı alım-satım.")
    weights_ui['adx_weight'] = st.slider("ADX Ağırlığı", 0, 50, 10, help="Trendin gücü (yönü değil).")
    weights_ui['volume_weight'] = st.slider("Hacim Artışı Ağırlığı", 0, 50, 15, help="Sinyali hacimle teyit etme.")
    # ATR puanlama için kullanılmaz, seviye belirleme içindir.

scan = st.sidebar.button("🔍 Tara / Yenile")
if st.session_state.last_scan_time:
    st.sidebar.caption(f"Son Tarama: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

# --- 2. ÜST: Piyasa Analizi ---
if gemini_api_key_ui:
    with st.spinner("Piyasa analizi alınıyor..."):
        market_analysis_text = get_market_analysis(gemini_api_key_ui)
    if market_analysis_text:
        st.markdown(f"**🧠 Günlük Piyasa Yorumu:** {market_analysis_text}")
    st.markdown("---")


# --- 3. ORTA ve SAĞ: Detay Alanı (Layout) ---
col_mid, col_right = st.columns([1.8, 1.2])

# Başlangıçta veya seçim temizlendiğinde yer tutucular
if st.session_state.selected_signal_data is None:
    with col_mid:
        st.info("Detayları görmek için aşağıdaki tarama sonuçları listesinden bir sinyale tıklayın.")
    with col_right:
        st.info("Algoritmik puan, indikatör katkıları ve AI analizi burada görünecek.")

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
        st.caption(f"{len(df_results)} sinyal bulundu.")
        # Filtreleme UI
        filter_col1, filter_col2 = st.columns(2)
        filter_label = filter_col1.selectbox("Sinyale Göre Filtrele", ["Tümü", "GÜÇLÜ AL", "AL", "SAT", "GÜÇLÜ SAT", "TUT"], key="filter_label")
        sort_by = filter_col2.selectbox("Sırala", ["Puana Göre (En Yüksek)", "Puana Göre (En Düşük)", "Fiyata Göre"], key="sort_by")

        # Filtreleme ve Sıralama
        filtered_df = df_results.copy()
        if filter_label != "Tümü":
            filtered_df = filtered_df[filtered_df['algo_label'] == filter_label]

        if sort_by == "Puana Göre (En Yüksek)":
            filtered_df = filtered_df.sort_values(by='algo_score', ascending=False)
        elif sort_by == "Puana Göre (En Düşük)":
            filtered_df = filtered_df.sort_values(by='algo_score', ascending=True)
        else: # Fiyata Göre (varsayılan)
             filtered_df = filtered_df.sort_values(by='price', ascending=False)

        # Sonuçları satır satır bas ve tıklanabilir yap
        for _, row in filtered_df.iterrows():
            cols = st.columns([1, 1, 1, 1, 1, 0.5])
            color = "green" if row['algo_score'] > 50 else ("red" if row['algo_score'] < 50 else "gray")
            cols[0].markdown(f"**{row['symbol']}**")
            cols[1].markdown(f"**{row['tf']}**")
            cols[2].markdown(f"**{row['price']:.5f}**")
            cols[3].markdown(f"<span style='color:{color}; font-weight:bold;'>{row['algo_label']} ({row['algo_score']})</span>", unsafe_allow_html=True)
            # Gemini puanı varsa göster
            if row['gemini_analysis'] and 'score' in row['gemini_analysis']:
                g_score = row['gemini_analysis']['score']
                g_color = "green" if g_score > 60 else ("red" if g_score < 40 else "gray")
                cols[4].markdown(f"AI: <span style='color:{g_color};'>%{g_score}</span>", unsafe_allow_html=True)
            
            if cols[5].button("Detay", key=f"btn_{row['symbol']}_{row['tf']}"):
                st.session_state.selected_signal_data = row # Tüm satır verisini state'e kaydet
                st.experimental_rerun() # Sayfayı yeniden çalıştırarak Orta/Sağ panelleri güncelle

with tab_history:
    st.markdown("### 📌 Takip Edilen Sinyaller")
    # ... (Buraya 'tracked_signals' listesini gösterme ve kaydetme mantığı eklenecek) ...
    st.info("Sinyal takip sistemi yakında eklenecektir.")


# --- 5. Detay Panellerini Doldur (Eğer bir sinyal seçildiyse) ---
selected_data = st.session_state.get('selected_signal_data')

if selected_data is not None:
    # --- ORTA: Grafik ---
    with col_mid:
        st.subheader(f"Grafik: {selected_data['symbol']} ({selected_data['tf']})")
        tv_tf = TV_INTERVAL_MAP.get(selected_data['tf'], '15') # Varsayılan 15m
        show_tradingview(selected_data['symbol'], tv_tf)

    # --- SAĞ: Analiz ---
    with col_right:
        st.subheader("Sinyal Analizi")
        
        # 1. Puan Metrikleri
        st.metric(label="Algoritma Sinyali", value=f"{selected_data['algo_label']} (%{selected_data['algo_score']})")
        
        # 2. Gemini Karşılaştırması
        gemini_data = selected_data.get('gemini_analysis')
        if gemini_data:
            if 'error' in gemini_data:
                st.error(f"AI Analizi Hatası: {gemini_data['error']}")
            elif 'score' in gemini_data:
                g_score = gemini_data['score']
                g_text = gemini_data['text']
                # Uyum kontrolü
                algo_buy = selected_data['algo_score'] > 50
                gemini_buy = g_score > 55 # Eşik
                uyum_mesaji = " (Uyumlu)" if algo_buy == gemini_buy else " (Uyumsuz!)"
                
                st.metric(label="Gemini AI Sinyali", value=f"%{g_score}{uyum_mesaji}")
                st.text_area("Gemini AI Yorumu", value=g_text, height=150, disabled=True, help="AI tarafından üretilen otomatik yorum.")
            else:
                 st.info("Gemini AI bu sinyal için bir puan veya yorum üretmedi.")
        else:
             st.info("Gemini AI analizi (API Anahtarı) aktif değil.")

        # 3. İndikatör Katkı Grafiği
        if PLOTLY_AVAILABLE:
            contributions = selected_data.get('algo_contributions')
            if contributions:
                plot_indicator_contributions(contributions)
            else:
                st.caption("Puan katkı detayı bulunamadı.")
        else:
            st.warning("Grafik için Plotly kütüphanesi gerekli.")
