import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
	page_title = "predksi harga saham",
	page_icon = ":tangerine:"
)

model = joblib.load("model_belajar1.joblib")

st.title(":tangerine: Belajar Klasifikasi Jeruk")
st.markdown("Aplikasi machine learning classification untuk memprediksi kualitas jeruk")

volume = st.text_input("Volume:", value="")
rsi = st.text_input("RSI:", value="")
macd = st.text_input("MACD:", value="")
vol_rata2 = st.text_input("Vol Avg:", value="")
vol_Status = st.pills("Asal Daerah", ["LOW", "NORMAL", "HIGHT"], default="NORMAL" )

if st.button("Prediksi", type="primary"):
	data_baru = pd.DataFrame([[volume,rsi,macd,vol_rata2,vol_Status]], columns=["Volume","RSI","MACD_12_26_9","Vol_Avg","Volume_Status"])
	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat dengan :tangerine: oleh **Yonz Suharyono**")
















