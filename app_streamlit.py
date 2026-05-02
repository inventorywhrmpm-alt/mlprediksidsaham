import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman
st.set_page_config(page_title="Stock Predictor RF", page_icon="📈")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load("model_saham_rf.joblib")

try:
    model = load_model()
except:
    st.error("Model tidak ditemukan! Jalankan script training dulu.")
    st.stop()

st.title("📈 Stock Movement Predictor")
st.markdown("Prediksi arah pergerakan harga **besok** menggunakan Random Forest.")

# Form Input
with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        volume = st.number_input("Volume Saat Ini:", min_value=0.0, format="%.0f")
        vol_rata2 = st.number_input("Rata-rata Volume (14 Hari):", min_value=0.0, format="%.0f")
    with col2:
        rsi = st.number_input("RSI (14):", min_value=0.0, max_value=100.0, value=50.0)
        macd = st.number_input("MACD (12, 26, 9):", value=0.0, format="%.4f")

    vol_status_text = st.pills("Status Volume Hari Ini", ["HIGH", "LOW", "NORMAL"], default="NORMAL")

# Tombol Prediksi
if st.button("Prediksi Arah Besok", type="primary", use_container_width=True):
    # Mapping status volume ke angka agar sesuai dengan training (HIGH=0, LOW=1, NORMAL=2)
    map_status = {"HIGH": 0, "LOW": 1, "NORMAL": 2}
    vol_encoded = map_status[vol_status_text]

    # Buat DataFrame Input
    data_baru = pd.DataFrame(
        [[volume, rsi, macd, vol_rata2, vol_encoded]], 
        columns=['Volume', 'RSI', 'MACD_12_26_9', 'Vol_Avg', 'Volume_Status_Encoded']
    )

    # Eksekusi Prediksi
    hasil = model.predict(data_baru)[0]
    prob = model.predict_proba(data_baru)[0]
    confidence = max(prob) * 100

    # Tampilkan Hasil
    st.divider()
    if hasil == "NAIK":
        st.success(f"### HASIL: {hasil} 🚀")
    elif hasil == "TURUN":
        st.error(f"### HASIL: {hasil} 📉")
    else:
        st.warning(f"### HASIL: {hasil} ⚖️")
        
    st.write(f"Tingkat Keyakinan Model: **{confidence:.2f}%**")
    
    if confidence < 55:
        st.info("💡 Keyakinan rendah, sebaiknya gabungkan dengan analisa manual.")

st.divider()
st.caption("Dibuat oleh Yonz Suharyono | Menggunakan Random Forest Classifier")
