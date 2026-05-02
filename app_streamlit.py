import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Prediksi Pergerakan Saham",
    page_icon="📈"
)

# Load Model
try:
   model = joblib.load("model_saham_rf.joblib")
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
vol_Status = st.pills("Status Volume", ["LOW", "NORMAL", "HIGH"], default="NORMAL")

if st.button("Prediksi", type="primary"):
    # Ubah teks status volume ke angka sesuai urutan alfabet (khas LabelEncoder)
    # Biasanya: HIGH=0, LOW=1, NORMAL=2 (cek hasil training)
    status_map = {"HIGH": 0, "LOW": 1, "NORMAL": 2}
    vol_encoded = status_map[vol_Status]

    # Buat DataFrame
    data_baru = pd.DataFrame(
        [[volume, rsi, macd, vol_rata2, vol_encoded]], 
        columns=["Volume", "RSI", "MACD_12_26_9", "Vol_Avg", "Volume_Status_Encoded"]
    )

    # Prediksi
    prediksi = model.predict(data_baru)[0]
    prob = model.predict_proba(data_baru)[0]
    confidence = max(prob)

    st.success(f"Prediksi Pergerakan Besok: **{prediksi}**")
    st.info(f"Keyakinan Model: {confidence*100:.2f}%")
        st.balloons()
        
    except ValueError as ve:
        st.error("Terjadi kesalahan input data. Pastikan format kolom sama dengan saat training.")
        st.write(ve)
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.caption("Dibuat dengan ❤️ oleh **Yonz Suharyono**")
