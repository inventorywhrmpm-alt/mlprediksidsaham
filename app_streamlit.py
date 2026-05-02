import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Stock Predictor RF", page_icon="📈")

# Load Model
# --- DI DALAM APP.PY ---

# Load model versi 2
model = joblib.load("model_saham_rf_v2.joblib")

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        rsi = st.number_input("RSI Hari Ini:", value=50.0)
        rsi_lag1 = st.number_input("RSI Kemarin:", value=50.0)
        price_now = st.number_input("Harga Close Hari Ini:", value=5000)
    with col2:
        macd = st.number_input("MACD (12,26,9):", value=0.0)
        ma10 = st.number_input("Nilai MA10 Hari Ini:", value=5000)
        price_yesterday = st.number_input("Harga Close Kemarin:", value=5000)

# Hitung Fitur Tambahan secara otomatis
price_lag1 = (price_now - price_yesterday) / price_yesterday
dist_ma10 = (price_now - ma10) / ma10

if st.button("Prediksi Arah Besok", type="primary"):
    # Susun data sesuai urutan 'features' di script training v2
    # features = ['RSI', 'RSI_Lag1', 'MACD_12_26_9', 'Price_Lag1', 'Dist_MA10']
    data_baru = pd.DataFrame(
        [[rsi, rsi_lag1, macd, price_lag1, dist_ma10]], 
        columns=['RSI', 'RSI_Lag1', 'MACD_12_26_9', 'Price_Lag1', 'Dist_MA10']
    )

    hasil = model.predict(data_baru)[0]
    prob = model.predict_proba(data_baru)[0]
    
    # Tampilkan Hasil (1 = NAIK, 0 = TURUN/TETAP)
    st.divider()
    label = "NAIK 🚀" if hasil == 1 else "TURUN/TETAP 📉"
    st.header(f"Prediksi Besok: {label}")
    st.write(f"Keyakinan Model: {max(prob)*100:.2f}%")
st.divider()
st.caption("Dibuat oleh Yonz Suharyono | Menggunakan Random Forest Classifier")
