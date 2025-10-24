# ai_engine.py
# Profesyonel Hibrit Analiz Motoru (Heuristic + Gemini AI) - v2

import math
import json
from pathlib import Path
from typing import Dict, Any
import os
import logging

# Gemini AI kütüphanesini içe aktarmayı dene
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Gemini kütüphanesi yüklü değil. 'pip install google-generativeai' ile yükleyin. Yalnızca heuristic mod kullanılabilir.")

RECORDS_FILE = Path("prediction_records.json")

def logistic(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def normalize(v, lo, hi):
    if v is None: return 0.0
    try: v = float(v)
    except Exception: return 0.0
    if hi == lo: return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))

def compute_trade_levels(price: float, atr: float, direction: str = 'LONG', risk_reward_ratio: float = 2.0, atr_multiplier: float = 1.5):
    """Hem LONG hem SHORT yönler için giriş, stop ve hedef hesaplar."""
    if not isinstance(price, (int, float)) or price <= 0: return {'entry': None, 'stop_loss': None, 'take_profit': None}
    
    atr = float(atr) if atr is not None and isinstance(atr, (int, float)) and atr > 0 else price * 0.02 # ATR yoksa veya geçersizse %2 varsay
    
    stop_distance = atr * atr_multiplier
    target_distance = stop_distance * risk_reward_ratio
    
    if direction == 'LONG':
        stop_loss = price - stop_distance
        take_profit = price + target_distance
    elif direction == 'SHORT':
        stop_loss = price + stop_distance
        take_profit = price - target_distance
    else: # NEUTRAL veya ERROR
        return {'entry': price, 'stop_loss': None, 'take_profit': None}

    return {
        'entry': price,
        'stop_loss': max(0.0, stop_loss),
        'take_profit': max(0.0, take_profit)
    }

def get_heuristic_analysis(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gelişmiş kural-bazlı (heuristic) analiz motoru.
    İndikatör katkılarını puanlar ve bir sinyal üretir.
    """
    long_score = 0
    short_score = 0
    explanation_points = []
    
    weights = { # Bu ağırlıklar app.py'den gelen 'score_signals' fonksiyonunda kullanılır. Burası sadece AI için mantık.
        'rsi_extreme': 25, 'rsi_trend': 15, 'macd_trend': 15, 'nw_slope': 30,
        'vol_spike': 10, 'score_bonus': 20, 'bb_reversal': 20
    }
    
    # Bu fonksiyon artık daha çok AI için bir yedek veya karşılaştırma aracı.
    # Ana puanlama 'score_signals' içinde yapılıyor.
    # Burada sadece basitleştirilmiş bir sinyal çıkarımı yapalım.
    
    score = indicators.get('score', 0) # app.py'den gelen ana skoru alalım

    signal = "NEUTRAL"
    confidence = 0
    if score > 50: signal = "LONG"; confidence = int(normalize(score, 50, 100)*50 + 50)
    elif score > 20: signal = "LONG"; confidence = int(normalize(score, 20, 50)*50)
    elif score < -50: signal = "SHORT"; confidence = int(normalize(abs(score), 50, 100)*50 + 50)
    elif score < -20: signal = "SHORT"; confidence = int(normalize(abs(score), 20, 50)*50)
    else: confidence = int(normalize(abs(score), 0, 20)*30)

    levels = compute_trade_levels(
        price=indicators.get('price'),
        atr=indicators.get('atr14'),
        direction=signal
    )

    explanation = f"**Algoritma Sinyali: {signal} (Skor: {score}, Güven Yaklaşık: {confidence}%)**\n"
    explanation += f"Bu sinyal, `app.py` içindeki `score_signals` fonksiyonunda tanımlanan ağırlıklara göre hesaplanan `{score}` puanına dayanmaktadır."

    return {
        "signal": signal,
        "confidence": confidence,
        "explanation": explanation,
        **levels
    }


def get_gemini_analysis(indicators: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Gemini AI kullanarak gelişmiş teknik analiz yapar (v2 - Scalp/Swing Modu).
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("Gemini AI kütüphanesi (google-generativeai) yüklü değil.")
        
    genai.configure(api_key=api_key)
    # Güvenlik ayarlarını daha az kısıtlayıcı yap (isteğe bağlı, riskli olabilir)
    # safety_settings = [
    #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    # ]
    # model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings)
    model = genai.GenerativeModel('gemini-pro')

    # Gemini'ye gönderilecek ham veriler
    # Sembol ve TF'i de ekleyelim
    symbol = indicators.get('symbol', 'Bilinmeyen')
    tf = indicators.get('timeframe', 'Bilinmeyen')
    scan_mode = indicators.get('scan_mode', 'Normal') # Scalp, Swing veya Normal

    data_context = json.dumps(indicators, indent=2)
    
    # --- Geliştirilmiş Prompt v2 (Scalp/Swing Modu) ---
    prompt = f"""
    Sen, {symbol} ({tf}) paritesi üzerinde uzmanlaşmış, kantitatif bir kripto para analistisin.
    Mevcut analiz modun: **{scan_mode}**.

    Görevin, aşağıdaki JSON verilerini analiz ederek bu moda uygun, mantıksal ve bilimsel bir ticaret planı oluşturmaktır.

    İNDİKATÖR VERİLERİ ({symbol} - {tf}):
    ```json
    {data_context}
    ```

    İNDİKATÖR AÇIKLAMALARI VE ÖNCELİKLER:
    1.  **`nw_slope` (Nadaraya-Watson Eğimi - ANA TREND):** En önemli gösterge. Pozitif ise yükseliş, negatif ise düşüş ana trendini gösterir. **Trendin tersine işlemden KESİNLİKLE kaçın.**
    2.  **`rsi14` ve `macd_hist` (Momentum):** Trend yönündeki momentumu onaylamak için kullanılır. Aşırı alım/satım bölgeleri (RSI 70/30) veya MACD Histogramının gücü önemlidir.
    3.  **`bb_upper` / `bb_lower` (Bollinger Bantları):** Volatilite ve potansiyel destek/direnç/hedef seviyeleri. Fiyatın bant dışına çıkması, trend yönünde bir kırılım VEYA tersine bir geri çekilme sinyali olabilir. Trendle birlikte yorumla.
    4.  **`atr14` (Volatilite):** SADECE Stop Loss mesafesini belirlemek için kullanılır.
    5.  **`vol_osc` (Hacim):** Trend veya kırılım yönündeki hacim artışı (pozitif değer) sinyali güçlendirir.
    6.  **`funding_rate` (Fonlama Oranı):** Aşırı pozitif/negatif değerler, piyasanın aşırı ısındığını ve tersine bir hareketin (geri çekilme) olasılığını artırabilir. Trendle birlikte değerlendir.
    7.  **`score` (Ön Skor):** Diğer algoritmanın genel puanı, ek bir teyit olarak kullanılabilir ama ana belirleyici DEĞİLDİR.

    ANALİZ MODU TALİMATLARI:
    - **{scan_mode} Modu:**
        - `{ 'Scalp' if scan_mode == 'Scalp' else ('Swing' if scan_mode == 'Swing' else 'Normal')}` stratejisine odaklan.
        - `{ 'Kısa vadeli (birkaç mum) fiyat hareketlerini ve hızlı momentum değişimlerini yakala.' if scan_mode == 'Scalp' else ('Orta/uzun vadeli (günler/haftalar) ana trendleri ve önemli destek/direnç seviyelerini takip et.' if scan_mode == 'Swing' else 'Genel trend ve momentumu dengeli bir şekilde değerlendir.')}`
        - Stop Loss ve Hedefler `{ 'daha dar' if scan_mode == 'Scalp' else ('daha geniş' if scan_mode == 'Swing' else 'standart')}` olmalı. (ATR çarpanını veya R:R oranını buna göre ayarla).

    TALEPLER:
    1.  Verileri ve `{scan_mode}` modunu dikkate alarak net bir SİNYAL belirle: "LONG", "SHORT" veya "NEUTRAL". **Ana trend (`nw_slope`) yönünde olmayan sinyal VERME.**
    2.  Bu sinyale olan GÜVENİNİ 0 ile 100 arasında belirt (Birden fazla göstergenin aynı yönü teyit etmesi güveni artırır).
    3.  Detaylı bir AÇIKLAMA yaz: Ana trendi belirt, momentumu yorumla, hacmi değerlendir, Bollinger bantlarının durumunu açıkla, fonlama oranının etkisini belirt. Sinyali destekleyen (Confirmation) ve zayıflatan (Contradiction) faktörleri listele. Riskleri vurgula.
    4.  Profesyonel bir GİRİŞ (entry), STOP LOSS (stop_loss) ve HEDEF KÂR (take_profit) seviyesi belirle. Stop `atr14`'e göre (mod'a uygun çarpanla), Hedef ise mantıklı bir R:R oranı (örn: 1:1.5 veya 1:2) veya önemli bir seviye (örn: karşı Bollinger Bandı) olmalı.

    CEVAP FORMATI (SADECE JSON):
    ```json
    {{
      "signal": "LONG",
      "confidence": 75,
      "explanation": "Ana trend (nw_slope) pozitif. RSI ve MACD momentumu LONG yönlü teyit ediyor...",
      "entry": 12345.67,
      "stop_loss": 12300.00,
      "take_profit": 12450.00
    }}
    ```
    """
    
    try:
        response = model.generate_content(prompt)
        logging.info(f"Gemini analizi başarılı: {symbol} - {tf}")
        # Gemini'den gelen yanıtı temizle ve JSON'a dönüştür
        # Bazen yanıt hatalı formatta veya eksik gelebilir, daha sağlam ayrıştırma
        try:
            # JSON bloğunu bulmaya çalış
            json_start = response.text.find('{')
            json_end = response.text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                 cleaned_response = response.text[json_start:json_end]
                 ai_plan = json.loads(cleaned_response)
            else:
                 # JSON bloğu bulunamazsa, eski temizleme yöntemini dene
                 cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
                 ai_plan = json.loads(cleaned_response)

        except (json.JSONDecodeError, AttributeError) as json_e:
             logging.error(f"Gemini yanıtı JSON ayrıştırma hatası ({symbol}, {tf}): {json_e}\nYanıt: {response.text}")
             raise ConnectionError(f"Gemini yanıtı JSON formatında değil: {json_e}")


        # Gemini'nin açıklamasını formatla
        ai_plan['explanation'] = f"**Gemini AI ({scan_mode}) Sinyal: {ai_plan.get('signal','N/A')} (Güven: {ai_plan.get('confidence','N/A')}%)**\n{ai_plan.get('explanation','Açıklama alınamadı.')}"
        
        # Seviyeleri doğrula ve float yap
        for key in ['entry', 'stop_loss', 'take_profit']:
            if key in ai_plan and isinstance(ai_plan[key], (int, float, str)):
                try:
                    ai_plan[key] = float(ai_plan[key])
                except (ValueError, TypeError):
                    ai_plan[key] = None # Geçersizse None yap
            else:
                ai_plan[key] = None # Eksikse None yap

        return ai_plan
        
    except Exception as e:
        raw_response_text = "Yanıt yok"
        if 'response' in locals() and hasattr(response, 'text'):
            raw_response_text = response.text
        logging.error(f"Gemini API veya İşlem Hatası ({symbol}, {tf}): {e}\nYanıt: {raw_response_text}")
        # Hata durumunda, hatayı içeren bir sözlük döndür
        return {
             "signal": "ERROR", "confidence": 0,
             "explanation": f"**Gemini AI Hatası:** {e}\nRaw Response:\n{raw_response_text[:500]}...", # Yanıtın başını göster
             "entry": indicators.get('price'), "stop_loss": None, "take_profit": None
        }


def get_ai_prediction(indicators: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    """
    Ana analiz fonksiyonu. API anahtarı varsa Gemini'yi, yoksa heuristic motoru kullanır.
    """
    if api_key and GEMINI_AVAILABLE:
        try:
            # Gemini'den gelen yanıtta seviyeler None ise, heuristic ile hesapla
            gemini_result = get_gemini_analysis(indicators, api_key)
            if gemini_result.get("signal") not in ["ERROR", "NEUTRAL"] and \
               (gemini_result.get("stop_loss") is None or gemini_result.get("take_profit") is None):
                logging.warning(f"Gemini seviye hesaplayamadı ({indicators.get('symbol')}, {indicators.get('timeframe')}), heuristic seviyeler kullanılıyor.")
                heuristic_levels = compute_trade_levels(
                    price=indicators.get('price'),
                    atr=indicators.get('atr14'),
                    direction=gemini_result.get("signal")
                )
                gemini_result.update(heuristic_levels) # Eksik seviyeleri güncelle
            return gemini_result
        except Exception as e:
            logging.warning(f"Gemini analizi genel hatası ({indicators.get('symbol')}), heuristic moda geçiliyor: {e}")
            return get_heuristic_analysis(indicators) # Genel hata durumunda heuristic'e dön
    else:
        # API anahtarı yoksa veya kütüphane yüklü değilse heuristic kullanılır
        return get_heuristic_analysis(indicators)

# --- Kayıt Fonksiyonları (Değişiklik Yok) ---
# ... (load_records, save_record, clear_records fonksiyonları aynı kalacak) ...
def load_records():
    if not RECORDS_FILE.exists(): return []
    try:
        with open(RECORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Kayıtlar yüklenemedi: {e}")
        return []

def save_record(record: Dict[str, Any]):
    recs = load_records()
    # Aynı sembol ve zaman dilimi için eski kaydı güncelle (opsiyonel)
    # updated = False
    # for i, r in enumerate(recs):
    #     if r.get('symbol') == record.get('symbol') and r.get('tf') == record.get('tf'):
    #         recs[i] = record
    #         updated = True
    #         break
    # if not updated:
    recs.append(record)
    try:
        with open(RECORDS_FILE, 'w', encoding='utf-8') as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"Kayıt kaydedilemedi: {e}")
        return False

def clear_records():
    try:
        if RECORDS_FILE.exists():
            RECORDS_FILE.unlink()
        return True
    except Exception as e:
        logging.error(f"Kayıtlar silinemedi: {e}")
        return False
