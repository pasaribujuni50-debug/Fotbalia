import streamlit as st
import pandas as pd
import math
import numpy as np
from textblob import TextBlob
from datetime import datetime
import os

# ==========================================
# 1. ENGINE: LOGIKA MACHINE LEARNING
# ==========================================
def poisson_prob(lmbda, k):
    return (math.pow(lmbda, k) * math.exp(-lmbda)) / math.factorial(k)

def save_to_csv(data):
    filename = "database_prediksi.csv"
    df_new = pd.DataFrame([data])
    if not os.path.isfile(filename):
        df_new.to_csv(filename, index=False)
    else:
        df_new.to_csv(filename, mode='a', header=False, index=False)

# ==========================================
# 2. DASHBOARD: TAMPILAN INTERAKTIF (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Pro Football AI", layout="wide")
st.title("🏆 AI Match Predictor Engine")

# Sidebar untuk Input
st.sidebar.header("Konfigurasi Pertandingan")
home_name = st.sidebar.text_input("Tim Tuan Rumah", "Home Team")
away_name = st.sidebar.text_input("Tim Tamu", "Away Team")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"🏠 {home_name}")
    h_lambda = st.number_input("Rata-rata Gol (Home)", value=1.5)
    h_news = st.text_area("Berita/Sentimen Home", "Pemain fit, moral tinggi")
    h_sent = TextBlob(h_news).sentiment.polarity
    h_adj = h_lambda * (1 + (h_sent * 0.1))
    st.info(f"Sentimen: {h_sent:.2f} | Adj Lambda: {h_adj:.2f}")

with col2:
    st.subheader(f"🚌 {away_name}")
    a_lambda = st.number_input("Rata-rata Gol (Away)", value=1.0)
    a_news = st.text_area("Berita/Sentimen Away", "Kiper utama cedera")
    a_sent = TextBlob(a_news).sentiment.polarity
    a_adj = a_lambda * (1 + (a_sent * 0.1))
    st.info(f"Sentimen: {a_sent:.2f} | Adj Lambda: {a_adj:.2f}")

# ==========================================
# 3. ANALISIS: MATRIKS & PROBABILITAS
# ==========================================
st.divider()
matrix = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        matrix[i, j] = (poisson_prob(h_adj, i) * poisson_prob(a_adj, j)) * 100

df_matrix = pd.DataFrame(matrix, 
                         columns=[f"Away {i}" for i in range(5)], 
                         index=[f"Home {i}" for i in range(5)])

st.subheader("🎯 Matriks Peluang Skor (%)")
st.dataframe(df_matrix.style.background_gradient(cmap='Greens'))

# Hitung Win/Draw/Loss
h_win = np.sum(np.tril(matrix, -1))
draw = np.sum(np.diag(matrix))
a_win = np.sum(np.triu(matrix, 1))

res_c1, res_c2, res_c3 = st.columns(3)
res_c1.metric(f"{home_name} Menang", f"{h_win:.1f}%")
res_c2.metric("Seri", f"{draw:.1f}%")
res_c3.metric(f"{away_name} Menang", f"{a_win:.1f}%")

# Simpan Data
if st.button("Simpan Prediksi ke CSV"):
    hasil = {
        "Tanggal": datetime.now().strftime("%Y-%m-%d"),
        "Match": f"{home_name} vs {away_name}",
        "H_Prob": f"{h_win:.1f}%",
        "A_Prob": f"{a_win:.1f}%",
        "Draw_Prob": f"{draw:.1f}%"
    }
    save_to_csv(hasil)
    st.success("Prediksi berhasil diarsipkan!")
