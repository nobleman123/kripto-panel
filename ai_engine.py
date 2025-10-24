import google.generativeai as genai
import pandas as pd

# ----------------------------
# 🔑 Gemini AI Ayarları
# ----------------------------
def configure_gemini(api_key: str):
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print("Gemini yapılandırılamadı:", e)
        return False


# ----------------------------
# 🧠 AI Analiz Fonksiyonu
# ----------------------------
def analyze_with_gemini(df_dict, api_key, strategy, symbol, timeframe, signal):
    """
    Gemini API'yi kullanarak kripto sinyal verilerini yorumlar.
    df_dict: son mum verilerinin dict hali
    signal: manuel hesaplanmış (BUY / SELL / HOLD)
    """
    try:
        if not configure_gemini(api_key):
            return "Gemini API anahtarı yapılandırılamadı."

        df = pd.DataFrame(df_dict)
        last_close = df["close"].iloc[-1]
        avg_volume = df["volume"].tail(20).mean()
        recent_trend = (
            "yükseliş eğiliminde" if df["close"].iloc[-1] > df["close"].iloc[-5]
            else "düşüş eğiliminde"
        )

        context = f"""
        Sen bir kripto analiz uzmanısın.
        {symbol} paritesine ait {timeframe} zaman diliminde {strategy} stratejisi için
        teknik veriler aşağıda yer alıyor.

        - Son fiyat: {last_close:.2f}
        - Ortalama hacim: {avg_volume:.2f}
        - Genel eğilim: {recent_trend}
        - Üretilen sinyal: {signal}

        Görev:
        1️⃣ Bu verileri kısa, teknik ama anlaşılır bir şekilde açıkla.  
        2️⃣ Stratejiye (Scalp / Swing / Long-term) göre kısa vadeli veya uzun vadeli beklentiyi belirt.  
        3️⃣ Risk / ödül oranına göre mantıklı bir öngörü sun.  
        4️⃣ Tahmini kazanç yüzdesi aralığı belirt (%2 - %5 gibi).  
        """

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(context)
        return response.text.strip() if response and response.text else "Yapay zekâdan geçerli yanıt alınamadı."

    except Exception as e:
        return f"Gemini yorumlaması başarısız: {e}"
