import streamlit as st
import pandas as pd
import pandas_ta as ta
import random
from datetime import datetime
import time
import requests
import json

# --- Configuration and Initialization ---
# Initialize session state for tracking signals (FIXED: Must use 'not in' to initialize only once)
if 'saved_signals' not in st.session_state:
    st.session_state.saved_signals = []

if 'market_summary' not in st.session_state:
    st.session_state.market_summary = None

SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT", "ADA/USDT", "SOL/USDT", "DOT/USDT", "AVAX/USDT"]
SCAN_RESULTS_KEY = 'scan_results'

# Gemini API Configuration (Leave as empty string, Canvas will provide the key)
API_KEY = "" 
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
IMAGEN_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict"

st.set_page_config(layout="wide", page_title="AI Destekli Kripto Sinyal Paneli")

# --- Gemini API Functions ---

def call_gemini_api(prompt, use_search=False, system_instruction=""):
    """
    Makes a POST request to the Gemini API with exponential backoff.
    """
    url = f"{GEMINI_API_URL}?key={API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }
    
    if use_search:
        payload["tools"] = [{"google_search": {}}]

    headers = {'Content-Type': 'application/json'}
    
    # Simple retry mechanism (exponential backoff not fully implemented here for brevity, 
    # but essential for production)
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            result = response.json()
            candidate = result.get('candidates', [None])[0]
            
            if candidate and candidate.get('content') and candidate['content'].get('parts'):
                text = candidate['content']['parts'][0]['text']
                return text
            
            return "API'den geçerli yanıt alınamadı."
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            return f"API Hatası ({response.status_code}): {e}. Gemini çağrısı başarısız oldu."
        except Exception as e:
            return f"İstekte hata: {e}"
            
    return "API'ye erişim sağlanamadı."


def generate_market_summary(strong_signals_df):
    """ Generates a market summary based on strong signals using Gemini. """
    st.toast("✨ Piyasa Özeti Oluşturuluyor...")
    
    if strong_signals_df.empty:
        summary_text = "Şu anda güçlü bir sinyal bulunamadı, bu nedenle piyasa özeti oluşturulamıyor."
        st.session_state.market_summary = summary_text
        return

    # Prepare data for the prompt
    signal_data = strong_signals_df[['Sembol', 'Sinyal', 'Açıklama']].to_markdown(index=False)
    
    system_prompt = (
        "Sen deneyimli bir kripto piyasası analistisin. Verilen güçlü teknik sinyallere ve genel piyasa "
        "trendlerine (Google Search'ten gelen güncel bilgilere dayanarak) odaklanarak Türkçe, 150 kelimeyi geçmeyen, "
        "kısa, nesnel ve profesyonel bir piyasa özeti yaz."
    )
    
    user_query = (
        "Aşağıdaki teknik analiz tarama sonuçlarını ve Google Arama'dan gelen güncel piyasa verilerini kullanarak genel bir "
        "kripto piyasası özeti oluştur:\n\n"
        f"Güçlü Sinyaller:\n{signal_data}"
    )
    
    result = call_gemini_api(user_query, use_search=True, system_instruction=system_prompt)
    st.session_state.market_summary = result
    st.toast("✨ Piyasa Özeti Tamamlandı!", icon='✅')


def get_signal_context(symbol, signal_type, entry_price):
    """ Generates a contextual explanation and risk analysis for a specific symbol using Gemini. """
    
    system_prompt = (
        "Sen profesyonel bir finansal risk uzmanısın. Kullanıcının takip listesine eklediği kripto para birimi sinyali "
        "hakkında Google Search'ten gelen güncel haberleri ve risk faktörlerini analiz et. Türkçe, 3-4 cümlelik, "
        "sembolün temel durumu, olası riskleri ve sinyali destekleyen/çürüten güncel olayları özetleyen bir metin oluştur."
    )
    
    user_query = (
        f"Sembol: {symbol}, Sinyal Tipi: {signal_type}, Giriş Fiyatı: {entry_price}. "
        f"Bu sinyalin geçerliliğini ve bu sembolle ilişkili güncel piyasa risklerini (düzenleyici haberler, teknolojik gelişmeler, önemli ortaklıklar vb.) analiz et."
    )
    
    result = call_gemini_api(user_query, use_search=True, system_instruction=system_prompt)
    return result

# --- Core Financial Logic (Reused) ---

def fetch_mock_data(symbol, period=200):
    """
    Simulates fetching OHLCV data.
    In a real app, this would be replaced by an API call (e.g., ccxt).
    """
    base_price = 1000 + random.randint(-50, 50)
    data = []
    
    for i in range(period):
        open_p = base_price * (1 + random.uniform(-0.01, 0.01))
        close_p = open_p * (1 + random.uniform(-0.01, 0.01))
        high_p = max(open_p, close_p) * (1 + random.uniform(0.001, 0.005))
        low_p = min(open_p, close_p) * (1 - random.uniform(0.001, 0.005))
        volume = 100000 + random.randint(-50000, 50000)
        
        data.append([
            datetime.now() - pd.Timedelta(days=period - 1 - i),
            open_p, high_p, low_p, close_p, volume
        ])
        base_price = close_p
    
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df.set_index('Date', inplace=True)
    return df

@st.cache_data
def generate_composite_signal(df):
    """
    Generates 'Güçlü Al' or 'Güçlü Sat' signal based on multiple indicators (RSI, MACD, BBands).
    """
    if df.empty:
        return {"signal": "Nötr", "reason": "Veri yok"}

    # Indicators calculation
    df.ta.rsi(append=True)
    last_rsi = df['RSI_14'].iloc[-1]
    df.ta.macd(append=True)
    last_macd = df['MACD_12_26_9'].iloc[-1]
    
    # BBands Calculation and Dynamic Name Extraction
    bbands_output = df.ta.bbands(close=df['Close'], length=20, std=2.0, append=True)
    
    # Varsayılan adlar: BBL_20_2.0, BBU_20_2.0
    lower_band_col = f"BBL_{20}_2.0"
    upper_band_col = f"BBU_{20}_2.0"
    
    # Kontrol et ve verileri çek
    try:
        last_close = df['Close'].iloc[-1]
        last_lower_band = df[lower_band_col].iloc[-1]
        last_upper_band = df[upper_band_col].iloc[-1]
    except KeyError:
        # Eğer adlar yanlışsa, BBands'in oluşturduğu sütunları bulmaya çalış
        bbands_cols = [col for col in df.columns if col.startswith('BBL_')]
        if not bbands_cols:
             return {"signal": "Hata", "reason": "BBands sütunları bulunamadı.", "price": df['Close'].iloc[-1] if not df.empty else 0}
        
        lower_band_col = bbands_cols[0]
        upper_band_col = lower_band_col.replace('BBL', 'BBU')
        
        last_lower_band = df[lower_band_col].iloc[-1]
        last_upper_band = df[upper_band_col].iloc[-1]


    # --- Signal Algorithm (Advanced Logic) ---
    is_oversold = last_rsi < 30
    is_macd_positive = last_macd > 0
    is_near_lower_band = last_close <= last_lower_band * 1.01

    if is_oversold and is_macd_positive and is_near_lower_band:
        return {
            "signal": "GÜÇLÜ AL", 
            "reason": f"RSI({last_rsi:.2f}) aşırı satım bölgesinde ve fiyat alt banda yakın. MACD yükseliş gösteriyor.", 
            "price": last_close
        }
    
    is_overbought = last_rsi > 70
    is_macd_negative = last_macd < 0
    is_near_upper_band = last_close >= last_upper_band * 0.99

    if is_overbought and is_macd_negative and is_near_upper_band:
        return {
            "signal": "GÜÇLÜ SAT", 
            "reason": f"RSI({last_rsi:.2f}) aşırı alım bölgesinde ve fiyat üst banda yakın. MACD düşüş gösteriyor.", 
            "price": last_close
        }

    # Weaker signals (can be filtered out)
    if last_rsi < 40 and is_macd_positive:
         return {"signal": "Al", "reason": f"RSI({last_rsi:.2f}) ve MACD yükseliş gösteriyor.", "price": last_close}
    if last_rsi > 60 and is_macd_negative:
         return {"signal": "Sat", "reason": f"RSI({last_rsi:.2f}) ve MACD düşüş gösteriyor.", "price": last_close}

    return {"signal": "Nötr", "reason": "Güçlü bir sinyal koşulu sağlanmadı.", "price": last_close}

# --- Streamlit Action Functions ---

def run_scan():
    """ Runs the scanner and updates session state. """
    st.toast("Tarama Başlatılıyor...", icon='🔍')
    
    new_results = []
    
    # Use st.progress for visual feedback
    progress_bar = st.progress(0, text="Semboller Analiz Ediliyor...")
    
    for i, symbol in enumerate(SYMBOLS):
        df = fetch_mock_data(symbol)
        analysis = generate_composite_signal(df)
        
        signal_type = analysis['signal']
        price = analysis.get('price', 0)
        
        if "GÜÇLÜ" in signal_type:
            new_results.append({
                "Sembol": symbol,
                "Sinyal": signal_type,
                "Fiyat": f"{price:.4f}",
                "Açıklama": analysis['reason']
            })
            
        progress_bar.progress((i + 1) / len(SYMBOLS), text=f"Semboller Analiz Ediliyor: {symbol}")

    # Store only the strong signals in the session state
    df_new_results = pd.DataFrame(new_results)
    st.session_state[SCAN_RESULTS_KEY] = df_new_results
    progress_bar.empty()
    st.toast(f"Tarama Tamamlandı. {len(new_results)} güçlü sinyal bulundu.", icon='✅')

    # Immediately generate market summary if results exist
    if not df_new_results.empty:
        generate_market_summary(df_new_results)


def check_targets():
    """ Checks if saved signals have hit their target prices. """
    st.toast("Hedefler Kontrol Ediliyor...", icon='🎯')
    updated_count = 0
    
    for signal in st.session_state.saved_signals:
        if signal['status'] == 'Takip Ediliyor':
            # Mock data fetch for current price
            df = fetch_mock_data(signal['symbol'], period=1)
            if df.empty: continue
            current_price = df['Close'].iloc[-1]
            
            entry = signal['entry_price']
            target = signal['target_price']
            
            # Target Hit Logic
            if "AL" in signal['signal_type'] and current_price >= target:
                signal['status'] = 'Hedefe Ulaşıldı'
                updated_count += 1
            elif "SAT" in signal['signal_type'] and current_price <= target:
                signal['status'] = 'Hedefe Ulaşıldı'
                updated_count += 1
            # Stop-Loss Mock Logic (e.g., 3% loss)
            elif "AL" in signal['signal_type'] and current_price <= entry * 0.97:
                 signal['status'] = 'Zarar Durumu'
                 updated_count += 1
            elif "SAT" in signal['signal_type'] and current_price >= entry * 1.03:
                 signal['status'] = 'Zarar Durumu'
                 updated_count += 1

    if updated_count > 0:
        st.toast(f"🎉 {updated_count} sinyalin durumu güncellendi!", icon='⬆️')
    else:
        st.toast("Takip edilen sinyallerde durum değişikliği olmadı.", icon='➖')

def save_signal_from_scan(row_data):
    """ Adds a selected signal from the scan results to the saved list. """
    
    symbol = row_data["Sembol"]
    signal = row_data["Sinyal"]
    entry_price = float(row_data["Fiyat"])
    
    # Check if already tracking
    if any(s['symbol'] == symbol and s['status'] == 'Takip Ediliyor' for s in st.session_state.saved_signals):
        st.warning(f"{symbol} zaten takip listenizde mevcut.")
        return

    # Set Target Price (5% profit goal)
    if "AL" in signal:
        target_price = entry_price * 1.05
    else: # GÜÇLÜ SAT
        target_price = entry_price * 0.95
        
    # GEMINI: Get contextual analysis
    context_analysis = get_signal_context(symbol, signal, entry_price)

    st.session_state.saved_signals.append({
        "symbol": symbol,
        "entry_price": entry_price,
        "target_price": target_price,
        "signal_type": signal,
        "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "status": "Takip Ediliyor",
        "context": context_analysis # New Gemini-powered field
    })
    st.toast(f"{symbol} sinyali kaydedildi! Hedef: {target_price:.4f}", icon='➕')

# --- Main Streamlit Layout ---

st.title("💰 AI Destekli Kripto Sinyal Paneli")
st.markdown("Güçlü indikatörlerle taranan piyasa sinyallerini listeleyin ve hedeflerinizi takip edin.")

# Top button to initiate scan
st.button("Tüm Sembolleri Tara (Yeni Sinyal Üret)", on_click=run_scan, type="primary")

# Gemini Powered Market Summary Display
st.markdown("---")
st.subheader("✨ Güncel Piyasa Özeti (Gemini Analizi)")

if st.session_state.market_summary:
    st.info(st.session_state.market_summary)
else:
    st.info("Piyasa özetini görmek için taramayı çalıştırın.")

st.markdown("---")

# Create two columns for the main layout
col1, col2 = st.columns([2, 1])

# --- Left Column: Scan Results ---
with col1:
    st.header("🔍 Tarama Sonuçları")
    st.subheader("Güçlü Al/Sat Sinyalleri")
    
    if SCAN_RESULTS_KEY not in st.session_state or st.session_state[SCAN_RESULTS_KEY].empty:
        st.info("Lütfen taramayı başlatın. Güçlü sinyal bulunduğunda burada listelenecektir.")
    else:
        df_scan = st.session_state[SCAN_RESULTS_KEY]
        
        # Streamlit Data Editor allows selecting rows
        edited_df = st.data_editor(
            df_scan, 
            key="scan_editor", 
            column_config={
                "Sinyal": st.column_config.TextColumn(
                    "Sinyal",
                    help="Güçlü Al/Sat Sinyali",
                    max_chars=15,
                ),
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
        )
        
        # Check if any row is selected to enable saving
        selected_rows = st.session_state.scan_editor.get("selection", {}).get("rows", [])
        
        if selected_rows:
            # Get the data of the first selected row
            selected_index = selected_rows[0]
            selected_data = df_scan.iloc[selected_index].to_dict()
            
            st.button(f"'{selected_data['Sembol']}' Sinyalini Kaydet ve Takibe Başla", 
                      on_click=save_signal_from_scan, 
                      args=(selected_data,), 
                      type="primary",
                      key="save_button")
        else:
             st.info("Kaydetmek istediğiniz sinyali tablodan seçiniz.")


# --- Right Column: Tracking Signals ---
with col2:
    st.header("🎯 Hedef Takip Listesi")
    st.subheader("Kaydedilen Sinyaller")
    
    # Button to check targets (This must be here to enable rerunning the check_targets logic)
    st.button("Hedefleri Kontrol Et (Anlık Simülasyon)", on_click=check_targets, key="check_target_button")
    
    if not st.session_state.saved_signals:
        st.info("Takip edilecek kaydedilmiş sinyal bulunmamaktadır.")
    else:
        # Convert list of dicts to DataFrame for display
        df_tracking = pd.DataFrame(st.session_state.saved_signals)
        
        # Clean up columns for display
        df_tracking_display = df_tracking[['symbol', 'entry_price', 'target_price', 'status', 'signal_type', 'entry_time', 'context']].copy()
        df_tracking_display.columns = ['Sembol', 'Giriş Fiyatı', 'Hedef Fiyatı', 'Durum', 'Tip', 'Giriş Zamanı', '✨ Risk Analizi']

        # Format numerical columns
        df_tracking_display['Giriş Fiyatı'] = df_tracking_display['Giriş Fiyatı'].apply(lambda x: f"{x:.4f}")
        df_tracking_display['Hedef Fiyatı'] = df_tracking_display['Hedef Fiyatı'].apply(lambda x: f"{x:.4f}")
        
        # Apply visual styling to the tracking table
        def style_status(val):
            if val == 'Hedefe Ulaşıldı':
                color = 'background-color: #d4edda; color: #155724' # Green
            elif val == 'Zarar Durumu':
                color = 'background-color: #f8d7da; color: #721c24' # Red
            elif val == 'Takip Ediliyor':
                color = 'background-color: #fff3cd; color: #856404' # Yellow
            else:
                color = ''
            return color

        st.dataframe(
            df_tracking_display.style.applymap(style_status, subset=['Durum']),
            use_container_width=True,
            hide_index=True
        )
