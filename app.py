import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
import requests
import io

# -------------------------------
# Sayfa Yapılandırması (Modern Dashboard)
# -------------------------------
st.set_page_config(
    page_title="Pro Kripto Vadeli Sinyal Paneli",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CSS Stilleri (İsteğe Bağlı Görsellik)
# -------------------------------
st.markdown("""
<style>
    /* Ana arkaplan */
    .stApp {
        background-color: #0F172A; /* Koyu tema */
    }
    
    /* Başlık */
    h1 {
        color: #F8FAFC;
        font-weight: 700;
    }
    h2, h3 {
        color: #E2E8F0;
    }

    /* Yan menü */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
    }
    
    /* Bilgi kutusu (yeni özet için) */
    .stAlert {
        background-color: #1E293B;
        border: 1px solid #334155;
        color: #E2E8F0;
    }

    /* Sonuç tablosu satırları */
    .result-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1.5fr 1fr 1fr 1fr 0.5fr;
        align-items: center;
        padding: 8px 12px;
        border-bottom: 1px solid #334155;
        transition: background-color 0.2s;
    }
    .result-row:hover {
        background-color: #334155;
    }
    .result-header {
        font-weight: 700;
        color: #94A3B8;
        background-color: #1E293B;
    }
    
    /* Coin logosu ve ismi */
    .coin-logo {
        width: 28px;
        height: 28px;
        margin-right: 10px;
        vertical-align: middle;
    }
    .coin-name {
        font-size: 1.1em;
        font-weight: 600;
        color: #F1F5F9;
        vertical-align: middle;
    }
    
    /* Skor çubuğu */
    .score-bar-container {
        width: 100%;
        background-color: #374151;
        border-radius: 4px;
        height: 18px;
        overflow: hidden;
    }
    .score-bar {
        height: 100%;
        color: #111827;
        font-size: 12px;
        font-weight: 600;
        text-align: center;
        line-height: 18px;
    }
    
    /* Renkler */
    .green { background-color: #22C55E; }
    .light-green { background-color: #84CC16; }
    .yellow { background-color: #EAB308; }
    .orange { background-color: #F97316; }
    .red { background-color: #EF4444; }
    
    .text-green { color: #22C55E; }
    .text-light-green { color: #84CC16; }
    .text-yellow { color: #EAB308; }
    .text-orange { color: #F97316; }
    .text-red { color: #EF4444; }
    .text-gray { color: #94A3B8; }

</style>
""", unsafe_allow_html=True)


# -------------------------------
# Veri Çekme Fonksiyonları (Futures API ile)
# -------------------------------

@st.cache_data(ttl=600)
def get_all_futures_symbols():
    """Tüm Binance Vadeli (USDT Perpetual) coin listesini çeker."""
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        data = requests.get(url, timeout=10).json()
        symbols = [
            s['symbol'] for s in data['symbols']
            if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT' and s['contractType'] == 'PERPETUAL'
        ]
        return sorted(symbols)
    except Exception as e:
        st.sidebar.error(f"Sembol listesi alınamadı: {e}")
        return ["BTCUSDT", "ETHUSDT"]

@st.cache_data(ttl=300)
def get_top_symbols_by_volume(top_n=100):
    """Hacme göre en iyi N coini çeker."""
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data)
        df['quoteVolume'] = pd.to_numeric(df['quoteVolume'])
        usdt_pairs = df[df['symbol'].str.endswith('USDT')].sort_values(by="quoteVolume", ascending=False)
        return usdt_pairs.head(top_n)['symbol'].tolist()
    except Exception as e:
        st.sidebar.error(f"Top coinler alınamadı: {e}")
        return get_all_futures_symbols()[:top_n]

@st.cache_data(ttl=60)
def fetch_futures_klines(symbol, interval, limit=400):
    """Binance Futures API'sinden kline verisi çeker."""
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url, timeout=10).json()
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "trades", "tb_base_volume", "tb_quote_volume", "ignore"
        ])
        df = df.astype(float)
        df = df[["time", "open", "high", "low", "close", "volume"]]
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_funding_rate(symbol):
    """Tek bir coin için anlık fonlama oranını çeker."""
    try:
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
        data = requests.get(url, timeout=5).json()
        return float(data.get('lastFundingRate', 0))
    except Exception:
        return 0.0

def get_coin_logo_url(symbol):
    """Coin logosu için URL döndürür."""
    base_symbol = symbol.replace("USDT", "").lower()
    return f"https://raw.githubusercontent.com/atomiclabs/cryptocurrency-icons/master/128/color/{base_symbol}.png"

# -------------------------------
# Analiz ve Puanlama Fonksiyonları
# -------------------------------

def compute_indicators(df):
    """Gelişmiş göstergeleri hesaplar (Vortex ve CHOP eklendi)."""
    df = df.copy()
    
    # Trend
    df['ema20'] = ta.ema(df['close'], length=20)
    df['ema50'] = ta.ema(df['close'], length=50)
    df['ema200'] = ta.ema(df['close'], length=200)
    df['adx14'] = ta.adx(df['high'], df['low'], df['close']).iloc[:, 0] # Sadece ADX hattı

    # Momentum
    macd = ta.macd(df['close'])
    if isinstance(macd, pd.DataFrame) and not macd.empty:
        df['macd_line'] = macd.iloc[:, 0]
        df['macd_hist'] = macd.iloc[:, 1]
    
    df['rsi14'] = ta.rsi(df['close'], length=14)
    df['mfi14'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)

    # Volatilite / Aşırılık
    bb = ta.bbands(df['close'])
    if isinstance(bb, pd.DataFrame) and not bb.empty:
        df['bb_lower'] = bb.iloc[:, 0]
        df['bb_upper'] = bb.iloc[:, 2]

    # YENİ: Vortex Indicator (Trend Yönü)
    vortex = ta.vortex(df['high'], df['low'], df['close'])
    if isinstance(vortex, pd.DataFrame) and not vortex.empty:
        df['vortex_pos'] = vortex.iloc[:, 0] # VI+
        df['vortex_neg'] = vortex.iloc[:, 1] # VI-
    
    # YENİ: Choppiness Index (Piyasa Durumu: Trend / Yönsüz)
    df['chop'] = ta.chop(df['high'], df['low'], df['close'], length=14)

    df.dropna(inplace=True)
    return df

def map_score_label(score):
    if score >= 70: return "GÜÇLÜ AL"
    elif score >= 30: return "AL"
    elif score <= -70: return "GÜÇLÜ SAT"
    elif score <= -30: return "SAT"
    else: return "NÖTR"

def get_score_color(score):
    if score >= 70: return "green", "#22C55E"
    elif score >= 30: return "light-green", "#84CC16"
    elif score > -30: return "yellow", "#EAB308"
    elif score > -70: return "orange", "#F97316"
    else: return "red", "#EF4444"

def get_trend_icon(score):
    if score >= 30: return "📈"
    elif score <= -30: return "📉"
    else: return "📊"

def get_detail_summary(scores, latest_data):
    """(YENİ) Puanlara ve verilere göre bilgilendirici bir özet oluşturur."""
    
    summary_parts = []
    
    # 1. Trend Durumu
    ema_score = scores.get('EMA', 0)
    vortex_score = scores.get('Vortex', 0)
    adx_val = latest_data.get('adx14', 0)
    
    if ema_score > 0 and vortex_score > 0:
        summary_parts.append("güçlü bir yükseliş trendi")
    elif ema_score < 0 and vortex_score < 0:
        summary_parts.append("güçlü bir düşüş trendi")
    elif ema_score > 0 or vortex_score > 0:
        summary_parts.append("zayıf bir yükseliş eğilimi")
    elif ema_score < 0 or vortex_score < 0:
        summary_parts.append("zayıf bir düşüş eğilimi")
    else:
        summary_parts.append("yönsüz bir trend")
        
    if adx_val > 25:
        summary_parts.append(f"(ADX {adx_val:.0f} ile trendi onaylıyor)")
    else:
        summary_parts.append(f"(ADX {adx_val:.0f} ile zayıf trend)")

    # 2. Piyasa Yapısı (CHOP)
    chop_val = latest_data.get('chop', 50)
    if chop_val < 38.2:
        summary_parts.append("Piyasa 'Trend' modunda (CHOP < 38.2),")
    elif chop_val > 61.8:
        summary_parts.append("Piyasa 'Yönsüz/Sıkışık' (CHOP > 61.8),")
    else:
        summary_parts.append("Piyasa 'Kararsız' yapıda,")

    # 3. Momentum
    rsi_val = latest_data.get('rsi14', 50)
    if rsi_val > 70:
        summary_parts.append("momentum aşırı alımda.")
    elif rsi_val < 30:
        summary_parts.append("momentum aşırı satımda.")
    else:
        summary_parts.append("momentum nötr bölgede.")
        
    # 4. Sentiment
    funding_score = scores.get('Funding', 0)
    if funding_score > 0:
        summary_parts.append("Piyasa geneli (kontra) alım yönlü bir baskı yaratıyor.")
    elif funding_score < 0:
        summary_parts.append("Piyasa geneli (kontra) satım yönlü bir baskı yaratıyor.")
        
    return " ".join(summary_parts)


def score_latest_signals(df_latest, df_prev, funding_rate, weights):
    """İndikatörlere göre puanlama yapar (Vortex ve CHOP eklendi)."""
    total, scores, reasons = 0, {}, []
    
    try:
        # EMA Puanı (Trend)
        ema_score = 0
        if df_latest['ema20'] > df_latest['ema50'] > df_latest['ema200']:
            ema_score = weights['ema']
            reasons.append(f"EMA Yükseliş Dizilimi (+{weights['ema']})")
        elif df_latest['ema20'] < df_latest['ema50'] < df_latest['ema200']:
            ema_score = -weights['ema']
            reasons.append(f"EMA Düşüş Dizilimi (-{weights['ema']})")
        scores['EMA'] = ema_score
        total += ema_score
    except Exception: pass

    try:
        # RSI Puanı (Momentum/Aşırılık)
        rsi = df_latest['rsi14']
        rsi_score = 0
        if rsi < 30:
            rsi_score = weights['rsi']
            reasons.append(f"RSI Aşırı Satım (< 30) (+{weights['rsi']})")
        elif rsi > 70:
            rsi_score = -weights['rsi']
            reasons.append(f"RSI Aşırı Alım (> 70) (-{weights['rsi']})")
        scores['RSI'] = rsi_score
        total += rsi_score
    except Exception: pass

    try:
        # MACD Puanı (Momentum Kesişim)
        macd_score = 0
        if df_prev['macd_hist'] < 0 and df_latest['macd_hist'] > 0:
            macd_score = weights['macd']
            reasons.append(f"MACD Pozitif Kesişim (+{weights['macd']})")
        elif df_prev['macd_hist'] > 0 and df_latest['macd_hist'] < 0:
            macd_score = -weights['macd']
            reasons.append(f"MACD Negatif Kesişim (-{weights['macd']})")
        scores['MACD'] = macd_score
        total += macd_score
    except Exception: pass

    try:
        # Bollinger Puanı (Volatilite/Aşırılık)
        bb_score = 0
        if df_latest['close'] > df_latest['bb_upper']:
            bb_score = -int(weights['bb'] * 0.5) # Aşırı alım, geri çekilme
            reasons.append(f"Bollinger Üst Bandı Aşıldı (-{int(weights['bb'] * 0.5)})")
        elif df_latest['close'] < df_latest['bb_lower']:
            bb_score = weights['bb']
            reasons.append(f"Bollinger Alt Bandı Aşıldı (+{weights['bb']})")
        scores['Bollinger'] = bb_score
        total += bb_score
    except Exception: pass

    try:
        # ADX Puanı (Trend Gücü Filtresi)
        adx_score = 0
        if df_latest['adx14'] > 25:
            # Puanı, mevcut trend yönünde (EMA skoru) güçlendir
            adx_score = weights['adx'] * np.sign(scores.get('EMA', 0))
            reasons.append(f"ADX Güçlü Trend (> 25) ({'+' if adx_score >= 0 else ''}{adx_score})")
        scores['ADX'] = adx_score
        total += adx_score
    except Exception: pass
    
    try:
        # YENİ: Vortex Puanı (Trend Kesişimi)
        vortex_score = 0
        if df_prev['vortex_pos'] < df_prev['vortex_neg'] and df_latest['vortex_pos'] > df_latest['vortex_neg']:
            vortex_score = weights['vortex'] # Bullish kesişim
            reasons.append(f"Vortex Pozitif Kesişim (+{weights['vortex']})")
        elif df_prev['vortex_pos'] > df_prev['vortex_neg'] and df_latest['vortex_pos'] < df_latest['vortex_neg']:
            vortex_score = -weights['vortex'] # Bearish kesişim
            reasons.append(f"Vortex Negatif Kesişim (-{weights['vortex']})")
        scores['Vortex'] = vortex_score
        total += vortex_score
    except Exception: pass
        
    try:
        # YENİ: Choppiness Puanı (Piyasa Yapısı Filtresi)
        chop_score = 0
        chop = df_latest['chop']
        trend_direction = np.sign(scores.get('EMA', 0) + scores.get('MACD', 0))
        
        if chop < 38.2: # Güçlü Trend
            chop_score = int(weights['chop'] * trend_direction)
            reasons.append(f"Güçlü Trend (CHOP < 38.2) ({'+' if chop_score >= 0 else ''}{chop_score})")
        elif chop > 61.8: # Yönsüz Piyasa
            chop_score = -int(weights['chop'] * trend_direction) # Trend sinyallerini zayıflat
            reasons.append(f"Yönsüz Piyasa (CHOP > 61.8) ({'+' if chop_score >= 0 else ''}{chop_score})")
        scores['CHOP'] = chop_score
        total += chop_score
    except Exception: pass
    
    try:
        # Fonlama Oranı Puanı (Kontra-Sentiment)
        funding_score = 0
        if funding_rate > 0.0005: # Piyasada longlar baskın
            funding_score = -weights['funding'] # Kontra (ters) sinyal
            reasons.append(f"Yüksek Pozitif Fonlama (Kontra SAT) (-{weights['funding']})")
        elif funding_rate < -0.0005: # Piyasada shortlar baskın
            funding_score = weights['funding'] # Kontra (ters) sinyal
            reasons.append(f"Yüksek Negatif Fonlama (Kontra AL) (+{weights['funding']})")
        scores['Funding'] = funding_score
        total += funding_score
    except Exception: pass

    total = int(max(min(total, 100), -100)) # Skoru -100 ile +100 arasında sınırla
    label = map_score_label(total)
    trend_icon = get_trend_icon(total)
    
    # Bilgilendirici özet için 'latest_data' ve 'scores' kullanılır
    summary = get_detail_summary(scores, df_latest)
    
    return total, scores, reasons, label, trend_icon, summary, df_latest

@st.cache_data
def convert_df_to_csv(df):
   return df.to_csv(index=False).encode('utf-8')

# -------------------------------
# Streamlit UI - Yan Menü (Sidebar)
# -------------------------------

st.sidebar.title("Ayarlar ve Tarama")
st.sidebar.markdown("---")

# 1. Coin Seçimi
st.sidebar.subheader("1. Coin Listesi Seçimi")
scan_mode = st.sidebar.radio(
    "Hangi coinler taransın?",
    ["En Popüler (Hacim)", "Tüm Vadeli Coinler", "Özel Liste"],
    index=0,
    key="scan_mode"
)

all_symbols_list = get_all_futures_symbols()
symbols_to_scan = []

if scan_mode == "En Popüler (Hacim)":
    top_n = st.sidebar.selectbox("Popüler Coin Sayısı", [50, 100, 200], index=1)
    symbols_to_scan = get_top_symbols_by_volume(top_n)
elif scan_mode == "Tüm Vadeli Coinler":
    symbols_to_scan = all_symbols_list
    st.sidebar.info(f"{len(symbols_to_scan)} adet coin taranacak. Bu işlem uzun sürebilir.")
else: # Özel Liste
    symbols_to_scan = st.sidebar.multiselect(
        "Coinleri Seçin",
        all_symbols_list,
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
    )

# 2. Zaman Dilimi
st.sidebar.subheader("2. Zaman Dilimi Seçimi")
scan_timeframes = st.sidebar.multiselect(
    "Tarama zaman dilimleri:",
    ["5m", "15m", "30m", "1h", "4h", "1d"],
    default=["1h", "4h", "1d"]
)

# 3. Filtreler
st.sidebar.subheader("3. Sonuç Filtreleri")
filter_strong_only = st.sidebar.checkbox("Yalnızca 'Güçlü Al / Sat' göster")
filter_min_tf = st.sidebar.checkbox("Sadece '1h' ve üzeri TF'leri göster", value=True)

# 4. Tarama Butonu
st.sidebar.markdown("---")
scan_button = st.sidebar.button("🔍 Taramayı Başlat", type="primary", use_container_width=True)

# -------------------------------
# Ana Panel - Başlık
# -------------------------------
st.title("💹 Kripto Vadeli Sinyal Paneli")

# İndikatör Ağırlıkları (Gelişmiş ayar, expander içinde)
with st.expander("Gelişmiş: İndikatör Ağırlık Ayarları"):
    st.markdown("##### Trend ve Momentum Ağırlıkları")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        w_ema = st.slider("EMA Ağırlığı", 0, 30, 15, key="w_ema")
        w_rsi = st.slider("RSI Ağırlığı", 0, 30, 10, key="w_rsi")
    with col_w2:
        w_macd = st.slider("MACD Ağırlığı", 0, 30, 15, key="w_macd")
        w_bb = st.slider("Bollinger Ağırlığı", 0, 30, 5, key="w_bb")
    with col_w3:
        w_adx = st.slider("ADX (Trend Gücü) Ağırlığı", 0, 30, 10, key="w_adx")
        w_mfi = st.slider("MFI Ağırlığı (dahil değil)", 0, 30, 0, key="w_mfi", disabled=True) # MFI şu an skorda yok

    st.markdown("##### Komplike İndikatör Ağırlıkları")
    col_w4, col_w5, col_w6 = st.columns(3)
    with col_w4:
        w_vortex = st.slider("Vortex Ağırlığı (Kesişim)", 0, 30, 20, key="w_vortex")
    with col_w5:
        w_chop = st.slider("CHOP Ağırlığı (Filtre)", 0, 30, 10, key="w_chop")
    with col_w6:
        w_funding = st.slider("Fonlama Ağırlığı (Kontra)", 0, 30, 15, key="w_funding")
    
    weights = {
        'ema': w_ema, 'rsi': w_rsi, 'macd': w_macd, 'bb': w_bb, 'adx': w_adx,
        'vortex': w_vortex, 'chop': w_chop, 'funding': w_funding
    }

# -------------------------------
# Tarama Mantığı
# -------------------------------
if "results" not in st.session_state:
    st.session_state.results = []

if scan_button:
    if not symbols_to_scan or not scan_timeframes:
        st.error("Lütfen taranacak en az bir coin ve bir zaman dilimi seçin.")
    else:
        st.session_state.results = []
        progress_bar = st.progress(0, text="Tarama başlatılıyor...")
        total_symbols = len(symbols_to_scan)
        
        for i, sym in enumerate(symbols_to_scan):
            progress_bar.progress((i + 1) / total_symbols, text=f"[{i+1}/{total_symbols}] {sym} taranıyor...")
            
            try:
                funding_rate = fetch_funding_rate(sym)
                
                best_tf, best_score, details = None, -999, {}
                strong_buys, strong_sells = 0, 0
                
                for tf in scan_timeframes:
                    df = fetch_futures_klines(sym, tf)
                    if df.empty or len(df) < 200: 
                        continue
                    
                    df = compute_indicators(df)
                    if df.empty:
                        continue
                    
                    total, scores, reasons, label, trend_icon, summary, latest_data = score_latest_signals(
                        df.iloc[-1], df.iloc[-2], funding_rate, weights
                    )
                    
                    # Detay paneli için veri sakla
                    latest_data_dict = latest_data.to_dict()
                    latest_data_dict['funding_rate'] = funding_rate # Fonlama oranını da ekle
                    
                    details[tf] = {
                        "label": label, 
                        "score": total, 
                        "reasons": reasons, 
                        "scores": scores,
                        "summary": summary,
                        "latest_data": latest_data_dict
                    }
                    
                    if label == "GÜÇLÜ AL": strong_buys += 1
                    if label == "GÜÇLÜ SAT": strong_sells += 1
                    
                    if total > best_score:
                        best_tf, best_score = tf, total
                
                if best_tf:
                    final_label = map_score_label(best_score)
                    final_trend = get_trend_icon(best_score)
                    st.session_state.results.append({
                        "Coin": sym,
                        "Logo": get_coin_logo_url(sym),
                        "En İyi Zaman Dilimi": best_tf,
                        "Skor": best_score,
                        "Etiket": final_label,
                        "SB Sayısı": strong_buys,
                        "SS Sayısı": strong_sells,
                        "Trend": final_trend,
                        "Detay": details
                    })
            except Exception as e:
                st.warning(f"{sym} taranırken bir hata oluştu: {e}")
        
        progress_bar.empty()

# -------------------------------
# Sonuçların Gösterimi
# -------------------------------
if st.session_state.results:
    st.markdown("## 📊 Tarama Sonuçları")
    results_df = pd.DataFrame(st.session_state.results)
    
    # --- Filtreleri Uygula ---
    filtered_df = results_df.copy()
    if filter_strong_only:
        filtered_df = filtered_df[
            (filtered_df['Etiket'] == "GÜÇLÜ AL") | (filtered_df['Etiket'] == "GÜÇLÜ SAT")
        ]
    if filter_min_tf:
        filtered_df = filtered_df[~filtered_df['En İyi Zaman Dilimi'].str.contains('m')]

    st.info(f"{len(results_df)} coin tarandı. {len(filtered_df)} sonuç filtrelendi.")
    
    # --- CSV İndirme Butonu ---
    csv_data = convert_df_to_csv(filtered_df.drop(columns=['Logo', 'Detay']))
    st.download_button(
        label="📥 Sonuçları CSV Olarak İndir",
        data=csv_data,
        file_name=f"tarama_sonuclari_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
    
    # --- Modern Sonuç Tablosu ---
    st.markdown(
        """
        <div class="result-row result-header">
            <div>Coin</div>
            <div>En İyi TF</div>
            <div>Skor</div>
            <div>Etiket</div>
            <div>Trend</div>
            <div title="Kaç zaman diliminde 'Güçlü Al/Sat' sinyali verdi?">SB/SS</div>
            <div>Detay</div>
        </div>
        """, unsafe_allow_html=True
    )

    for _, row in filtered_df.sort_values(by="Skor", ascending=False).iterrows():
        color_class, color_hex = get_score_color(row['Skor'])
        
        score_bar_html = f"""
        <div class="score-bar-container" title="Skor: {row['Skor']}">
            <div class="score-bar {color_class}" style="width: {max(5, (abs(row['Skor']) / 100) * 100)}%; background-color: {color_hex};">
                {row['Skor']}
            </div>
        </div>
        """
        
        row_html = f"""
        <div class="result-row">
            <div>
                <img src="{row['Logo']}" class="coin-logo" onerror="this.style.display='none'">
                <span class="coin-name">{row['Coin']}</span>
            </div>
            <div>{row['En İyi Zaman Dilimi']}</div>
            <div>{score_bar_html}</div>
            <div class="text-{color_class}" style="font-weight: 600;">{row['Etiket']}</div>
            <div style="font-size: 1.5em;" title="{row['Trend']}">{row['Trend']}</div>
            <div title="Güçlü Al: {row['SB Sayısı']} | Güçlü Sat: {row['SS Sayısı']}">
                <span class="text-green">{row['SB Sayısı']}</span> / <span class="text-red">{row['SS Sayısı']}</span>
            </div>
        </div>
        """
        st.markdown(row_html, unsafe_allow_html=True)
        
        # --- Detay Paneli (Expander) ---
        with st.expander(f" detayları"):
            st.subheader(f"📊 {row['Coin']} — Detaylı Analiz")
            
            tf_tabs = st.tabs(list(row['Detay'].keys()))
            
            for i, tf in enumerate(row['Detay'].keys()):
                with tf_tabs[i]:
                    det = row['Detay'][tf]
                    latest_data = det['latest_data']
                    
                    st.markdown(f"#### {tf} Sinyali: <span class='text-{get_score_color(det['score'])[0]}'>{det['label']} (Skor: {det['score']})</span>", unsafe_allow_html=True)
                    
                    # YENİ: Bilgilendirici Özet
                    st.info(f"**Algoritmik Özet:** {det['summary']}")
                    
                    det_col1, det_col2 = st.columns([1, 1.5])
                    
                    with det_col1:
                        st.markdown("**Ana Metrikler:**")
                        st.metric("Kapanış Fiyatı", f"{latest_data.get('close', 0):,.4f} USDT")
                        st.metric("RSI (14)", f"{latest_data.get('rsi14', 0):.2f}")
                        st.metric("ADX (14)", f"{latest_data.get('adx14', 0):.2f}")
                        st.metric("CHOP (14)", f"{latest_data.get('chop', 0):.2f}", help="< 38.2 = Trend, > 61.8 = Yönsüz")
                        
                        v_pos = latest_data.get('vortex_pos', 0)
                        v_neg = latest_data.get('vortex_neg', 0)
                        v_delta = "Yükseliş" if v_pos > v_neg else "Düşüş"
                        st.metric("Vortex (VI+/VI-)", f"{v_pos:.2f} / {v_neg:.2f}", delta=v_delta)
                        
                        fr = latest_data.get('funding_rate', 0)
                        fr_delta = "Pozitif (Kontra Sat)" if fr > 0.0001 else ("Negatif (Kontra Al)" if fr < -0.0001 else "Nötr")
                        st.metric("Fonlama Oranı", f"{fr:.4%}", delta=fr_delta)

                    with det_col2:
                        st.markdown("**Puanlama Dağılımı:**")
                        score_df = pd.DataFrame.from_dict(det['scores'], orient='index', columns=['Puan'])
                        st.bar_chart(score_df)
            
            # TradingView Grafiği
            st.markdown("---")
            st.subheader("TradingView Grafiği")
            
            tf_map = {"5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
            tv_interval = tf_map.get(row['En İyi Zaman Dilimi'], "60")
            
            st.components.v1.html(f'''
                <div class="tradingview-widget-container" style="height:480px; width:100%">
                  <div id="tv_{row['Coin']}_{tv_interval}"></div>
                  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                  <script type="text/javascript">
                  new TradingView.widget({{
                      "container_id": "tv_{row['Coin']}_{tv_interval}",
                      "symbol": "BINANCE:{row['Coin']}",
                      "interval": "{tv_interval}",
                      "theme": "dark",
                      "style": "1",
                      "locale": "tr",
                      "hide_top_toolbar": false,
                      "allow_symbol_change": true
                  }});
                  </script>
                </div>
            ''', height=490)

else:
    st.info("Ayarları seçtikten sonra 'Taramayı Başlat' butonuna basın.")