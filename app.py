import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard Riset Wistar mTOR", layout="wide")

st.title("Analisis Resistance Training pada 8 Tikus Wistar")
st.markdown("**Data Kuantitatif:** Gambaran Histologi mTOR *Gastrocnemius* | **Data Kualitatif:** mTOR ELISA, Kekuatan Otot, Endurance & VO2 Max")

df = pd.read_csv("dataset_wistar_dummy.csv")

# Filter kelompok
pilihan = st.sidebar.radio("Pilih Kelompok Tikus:", ["Semua (8 Ekor)", "Kontrol (4 Ekor)", "Resistance (4 Ekor)"])
if pilihan == "Kontrol (4 Ekor)":
    df_tampil = df[df["kelompok"] == "Kontrol"]
elif pilihan == "Resistance (4 Ekor)":
    df_tampil = df[df["kelompok"] == "Resistance"]
else:
    df_tampil = df

# Ringkasan angka utama
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Sampel", f"{len(df_tampil)} Ekor")
col2.metric("Rata-rata Histologi mTOR (Kuantitatif)", f"{df_tampil['histologi_mtor_persen'].mean():.1f}%")
col3.metric("Rata-rata Luas Serabut Otot", f"{df_tampil['luas_serabut_csa_um2'].mean():.0f} µm²")

st.divider()

# Grafik & Tabel
kiri, kanan = st.columns(2)
with kiri:
    st.subheader("Grafik Kuantitatif: Histologi mTOR (%)")
    st.bar_chart(df_tampil.set_index("rat_id")["histologi_mtor_persen"])

with kanan:
    st.subheader("Tabel Kualitatif: ELISA, Kekuatan & VO2 Max")
    st.dataframe(df_tampil[["rat_id", "kelompok", "mtor_elisa_kualitatif", "kekuatan_otot_kualitatif", "endurance_vo2max_kualitatif"]])
