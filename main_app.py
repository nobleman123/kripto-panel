# --- Coin Detay Görünümü (TradingView Grafiği + Skorlar + Bilgiler) ---
import streamlit as st

def show_coin_details(symbol, timeframe, score, ai_score, entry_price, target_price, stop_loss):
    """
    Seçilen coin için detaylı görünüm — TradingView grafiği, skorlar ve açıklamalar dahil.
    """
    
    # Ekranı temizle (önceki grafik DOM'dan silinir, removeChild hatası engellenir)
    placeholder = st.empty()
    with placeholder.container():
        st.markdown(f"""
            <h2 style='text-align:center; color:#0dcaf0;'>
                {symbol.upper()} - {timeframe} Zaman Dilimi
            </h2>
            <hr>
        """, unsafe_allow_html=True)

        # --- TradingView Grafik ---
        tradingview_html = f"""
        <div class="tradingview-widget-container">
            <div id="tradingview_chart"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
            new TradingView.widget({{
              "width": "100%",
              "height": 380,
              "symbol": "BINANCE:{symbol.upper()}USDT",
              "interval": "{timeframe}",
              "timezone": "Etc/UTC",
              "theme": "dark",
              "style": "1",
              "locale": "tr",
              "toolbar_bg": "#f1f3f6",
              "enable_publishing": false,
              "allow_symbol_change": true,
              "hide_top_toolbar": false,
              "hide_legend": false
            }});
            </script>
        </div>
        """

        # --- Grafik render işlemi ---
        st.components.v1.html(tradingview_html, height=400)

        # --- Skor Bilgileri ---
        st.markdown("""
        <style>
        .score-card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background-color: #111827;
            border-radius: 12px;
            padding: 10px;
            margin: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.4);
            color: #fff;
            width: 100%;
        }}
        .score-title {{ font-size: 16px; color: #999; }}
        .score-value {{ font-size: 22px; font-weight: bold; color: #0dcaf0; }}
        </style>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='score-card'><div class='score-title'>Sinyal Skoru</div><div class='score-value'>{score:.2f}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='score-card'><div class='score-title'>AI Tahmini</div><div class='score-value'>{ai_score:.1f}%</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='score-card'><div class='score-title'>Zaman Dilimi</div><div class='score-value'>{timeframe}</div></div>", unsafe_allow_html=True)

        # --- Giriş / Hedef / Stop ---
        st.markdown(f"""
        <div style='text-align:center; margin-top:10px;'>
            <b>Giriş:</b> {entry_price:.4f} | 
            <b>Hedef:</b> <span style='color:#0f0'>{target_price:.4f}</span> | 
            <b>Stop:</b> <span style='color:#f33'>{stop_loss:.4f}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.info("💡 Grafiği yakınlaştırabilir veya farklı zaman dilimlerini seçebilirsiniz.", icon="ℹ️")
