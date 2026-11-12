import streamlit as st
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Dashboard Riset Wistar mTOR", layout="wide")

st.title("Analisis Resistance Training pada 8 Tikus Wistar")
st.markdown("**Kuantitatif:** Histologi mTOR *Gastrocnemius* | **Kualitatif:** mTOR ELISA, Kekuatan Otot, Endurance & VO2 Max")

# 1. Membaca Data
df = pd.read_csv("dataset_wistar_dummy.csv")

# Filter Kelompok di Sidebar
pilihan = st.sidebar.radio("Filter Kelompok Tikus:", ["Semua (8 Ekor)", "Kontrol (4 Ekor)", "Resistance (4 Ekor)"])
if pilihan == "Kontrol (4 Ekor)":
    df_tampil = df[df["kelompok"] == "Kontrol"]
elif pilihan == "Resistance (4 Ekor)":
    df_tampil = df[df["kelompok"] == "Resistance"]
else:
    df_tampil = df

# Uji Statistik (Uji-T Independen)
kontrol = df[df["kelompok"] == "Kontrol"]["histologi_mtor_persen"]
resistance = df[df["kelompok"] == "Resistance"]["histologi_mtor_persen"]
_, p_val = stats.ttest_ind(resistance, kontrol)

# Ringkasan Metrik Utama
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Sampel", f"{len(df_tampil)} Ekor")
col2.metric("Rata-rata Histologi mTOR", f"{df_tampil['histologi_mtor_persen'].mean():.1f}%")
col3.metric("Rata-rata Luas Serabut", f"{df_tampil['luas_serabut_csa_um2'].mean():.0f} µm²")
col4.metric("Uji-T (p-value)", f"{p_val:.5f}", delta="Signifikan (<0.05)" if p_val < 0.05 else "Tidak Signifikan")

st.divider()

# Grafik & Tabel
kiri, kanan = st.columns(2)
with kiri:
    st.subheader("1. Data Kuantitatif: Histologi mTOR (%)")
    st.bar_chart(df_tampil.set_index("rat_id")["histologi_mtor_persen"])

with kanan:
    st.subheader("2. Data Kualitatif: ELISA, Kekuatan & VO2 Max")
    st.dataframe(df_tampil[["rat_id", "kelompok", "mtor_elisa_kualitatif", "kekuatan_otot_kualitatif", "endurance_vo2max_kualitatif"]], use_container_width=True)

st.divider()

# 3. FITUR MACHINE LEARNING: KALKULATOR PREDIKSI INTERAKTIF
st.subheader("3. Simulasi Prediksi Machine Learning (Random Forest)")
st.write("Pilih profil kualitatif tikus baru di bawah ini untuk memprediksi persentase ekspresi kuantitatif histologi mTOR *gastrocnemius*:")

# Melatih Model Machine Learning dari 8 data tikus
skala = {
    'Rendah': 1, 'Lemah': 1, 'Kurang': 1,
    'Sedang': 2, 'Normal': 2, 'Cukup': 2,
    'Tinggi': 3, 'Kuat': 3, 'Baik': 3,
    'Sangat Kuat': 4, 'Sangat Baik': 4
}
X = df[['mtor_elisa_kualitatif', 'kekuatan_otot_kualitatif', 'endurance_vo2max_kualitatif']].replace(skala)
y = df['histologi_mtor_persen']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Input Pengguna
c_elisa, c_kuat, c_vo2 = st.columns(3)
with c_elisa:
    in_elisa = st.selectbox("Hasil mTOR ELISA (Kualitatif):", ["Rendah", "Sedang", "Tinggi"])
with c_kuat:
    in_kuat = st.selectbox("Kekuatan Otot (Kualitatif):", ["Lemah", "Normal", "Kuat", "Sangat Kuat"])
with c_vo2:
    in_vo2 = st.selectbox("Endurance & VO2 Max (Kualitatif):", ["Kurang", "Cukup", "Baik", "Sangat Baik"])

# Hitung Prediksi Otomatis
input_df = pd.DataFrame([[skala[in_elisa], skala[in_kuat], skala[in_vo2]]],
                        columns=['mtor_elisa_kualitatif', 'kekuatan_otot_kualitatif', 'endurance_vo2max_kualitatif'])
hasil_prediksi = model.predict(input_df)[0]

st.success(f"**Hasil Prediksi Model AI:** Perkiraan Ekspresi Kuantitatif Histologi mTOR adalah **{hasil_prediksi:.2f}%**")
