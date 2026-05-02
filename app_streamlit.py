import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi Pergerakan Saham",
    page_icon="📈"
)

# Load Model
try:
    model = joblib.load("model_belajar1.joblib")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

st.title("📈 Analisa Prediksi Saham")
st.markdown("Masukkan data teknikal untuk memprediksi arah harga (Naik/Turun/Tetap)")

# Ganti text_input menjadi number_input agar tipe datanya otomatis FLOAT/INT
volume = st.number_input("Volume (Angka):", min_value=0.0, step=1.0)
rsi = st.number_input("RSI Value:", min_value=0.0, max_value=100.0, value=50.0)
macd = st.number_input("MACD Value:", value=0.0)
vol_rata2 = st.number_input("Vol Avg (Periode):", min_value=0.0, step=1.0)

# Pastikan pill ini sesuai dengan kolom "Volume_Status" yang ada di dataset training
vol_status = st.pills("Status Volume", ["LOW", "NORMAL", "HIGHT"], default="NORMAL")

if st.button("Prediksi", type="primary"):
    # Pastikan urutan dan nama kolom SAMA PERSIS dengan saat training model
    # Jika saat training menggunakan kolom 'Volume_Status' sebagai kategori, 
    # pastikan model kamu menggunakan Pipeline/Encoder.
    
    data_baru = pd.DataFrame(
        [[volume, rsi, macd, vol_rata2, vol_status]], 
        columns=["Volume", "RSI", "MACD_12_26_9", "Vol_Avg", "Volume_Status"]
    )

    try:
        # Lakukan prediksi
        prediksi = model.predict(data_baru)[0]
        
        # Ambil probabilitas (keyakinan)
        probabilitas = model.predict_proba(data_baru)[0]
        presentase = max(probabilitas)
        
        # Mapping hasil jika label_num digunakan (misal: 1 = Naik, dst)
        hasil_map = {1: "NAIK", -1: "TURUN", 0: "TETAP"}
        hasil_final = hasil_map.get(prediksi, prediksi)

        st.success(f"Model memprediksi pergerakan: **{hasil_final}**")
        st.info(f"Tingkat keyakinan: **{presentase*100:.2f}%**")
        st.balloons()
        
    except ValueError as ve:
        st.error("Terjadi kesalahan input data. Pastikan format kolom sama dengan saat training.")
        st.write(ve)
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.caption("Dibuat dengan ❤️ oleh **Yonz Suharyono**")
