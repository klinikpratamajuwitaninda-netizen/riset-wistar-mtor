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

# ==============================================================================
# TAMBAHAN BARU: PEMODELAN STATISTIK GAYA SPSS (STATISTICAL MODELLING)
# ==============================================================================
from scipy import stats

st.divider()
st.header("Pemodelan Statistik Gaya SPSS (Statistical Modelling)")

col_spss_1, col_spss_2 = st.columns(2)

# --- BAGIAN KIRI: UJI KUANTITATIF (HISTOLOGI mTOR) ---
with col_spss_1:
    st.subheader("A. Uji Kuantitatif (Histologi mTOR Gastrocnemius)")
    ctrl = df[df["kelompok"] == "Kontrol"]["histologi_mtor_persen"]
    res = df[df["kelompok"] == "Resistance"]["histologi_mtor_persen"]

    # Hitung statistik kuantitatif
    _, p_sw_ctrl = stats.shapiro(ctrl)
    _, p_sw_res = stats.shapiro(res)
    t_stat, p_ttest = stats.ttest_ind(res, ctrl)
    u_stat, p_mann = stats.mannwhitneyu(res, ctrl)

    tabel_quant = pd.DataFrame({
        "Parameter Uji (Kuantitatif)": [
            "Mean ± SD (Kontrol)",
            "Mean ± SD (Resistance)",
            "Uji Normalitas Shapiro-Wilk (Kontrol)",
            "Uji Normalitas Shapiro-Wilk (Resistance)",
            "Independent T-Test (p-value)",
            "Mann-Whitney U Test (p-value)"
        ],
        "Hasil": [
            f"{ctrl.mean():.2f}% ± {ctrl.std():.2f}",
            f"{res.mean():.2f}% ± {res.std():.2f}",
            f"p = {p_sw_ctrl:.4f} (Normal)",
            f"p = {p_sw_res:.4f} (Normal)",
            f"t = {t_stat:.3f} (p = {p_ttest:.5f} | Signifikan)",
            f"U = {u_stat:.1f} (p = {p_mann:.5f} | Signifikan)"
        ]
    })
    st.dataframe(tabel_quant, use_container_width=True, hide_index=True)

# --- BAGIAN KANAN: UJI KUALITATIF (NON-HISTOLOGI) ---
with col_spss_2:
    st.subheader("B. Uji Kualitatif (ELISA, Kekuatan, Endurance & VO2 Max)")
    pilih_qual = st.selectbox(
        "Pilih Variabel Non-Histologi untuk Tabulasi Silang:",
        ["mtor_elisa_kualitatif", "kekuatan_otot_kualitatif", "endurance_vo2max_kualitatif"],
        format_func=lambda x: {
            "mtor_elisa_kualitatif": "mTOR ELISA (Kualitatif)",
            "kekuatan_otot_kualitatif": "Kekuatan Otot (Kualitatif)",
            "endurance_vo2max_kualitatif": "Endurance & VO2 Max (Kualitatif)"
        }[x]
    )

    crosstab_df = pd.crosstab(df["kelompok"], df[pilih_qual])
    chi2, p_chi, _, _ = stats.chi2_contingency(crosstab_df)

    st.markdown("**Tabel Crosstabs (Distribusi Frekuensi):**")
    st.dataframe(crosstab_df, use_container_width=True)
    st.info(f"**Hasil Uji Chi-Square:** $\\chi^2 = {chi2:.3f}$, **p-value = {p_chi:.4f}** ({'Asosiasi Signifikan' if p_chi < 0.05 else 'Belum Signifikan pada N=8'})")
input_df = pd.DataFrame([[skala[in_elisa], skala[in_kuat], skala[in_vo2]]],
                        columns=['mtor_elisa_kualitatif', 'kekuatan_otot_kualitatif', 'endurance_vo2max_kualitatif'])
hasil_prediksi = model.predict(input_df)[0]

st.success(f"**Hasil Prediksi Model AI:** Perkiraan Ekspresi Kuantitatif Histologi mTOR adalah **{hasil_prediksi:.2f}%**")


# ==============================================================================
# TAMBAHAN: ALGORITMA AUTO-SPSS & FEATURE IMPORTANCE MACHINE LEARNING
# ==============================================================================
from sklearn.tree import DecisionTreeRegressor, export_text

st.divider()
st.header("Algoritma Auto-SPSS & Analisis Pengaruh Faktor (Feature Importance)")

col_alg1, col_alg2 = st.columns(2)

# 1. Algoritma Pemilihan Uji SPSS Otomatis
with col_alg1:
    st.subheader("1. Algoritma Keputusan Uji Statistik (Auto-SPSS)")
    ctrl_data = df[df["kelompok"] == "Kontrol"]["histologi_mtor_persen"]
    res_data = df[df["kelompok"] == "Resistance"]["histologi_mtor_persen"]

    _, p_norm_ctrl = stats.shapiro(ctrl_data)
    _, p_norm_res = stats.shapiro(res_data)

    if p_norm_ctrl > 0.05 and p_norm_res > 0.05:
        metode_terpilih = "Independent Samples T-Test (Parametrik)"
        stat_val, p_final = stats.ttest_ind(res_data, ctrl_data)
        alasan = f"Kedua kelompok berdistribusi normal (Shapiro-Wilk p = {p_norm_ctrl:.3f} & {p_norm_res:.3f} > 0.05)."
    else:
        metode_terpilih = "Mann-Whitney U Test (Non-Parametrik)"
        stat_val, p_final = stats.mannwhitneyu(res_data, ctrl_data)
        alasan = "Salah satu kelompok tidak berdistribusi normal (p <= 0.05)."

    st.success(f"**Uji Terpilih Otomatis:** {metode_terpilih}")
    st.write(f"**Alasan Algoritma:** {alasan}")
    st.write(f"**Hasil Akhir:** Statistik = `{stat_val:.4f}` | **p-value = `{p_final:.6f}`**")

# 2. Algoritma Machine Learning: Bobot Pengaruh Variabel Kualitatif
with col_alg2:
    st.subheader("2. Algoritma Pengaruh Faktor Kualitatif (ML Weights)")
    skala_ml = {'Rendah': 1, 'Lemah': 1, 'Kurang': 1, 'Sedang': 2, 'Normal': 2, 'Cukup': 2, 'Tinggi': 3, 'Kuat': 3, 'Baik': 3, 'Sangat Kuat': 4, 'Sangat Baik': 4}
    X_ml = df[['mtor_elisa_kualitatif', 'kekuatan_otot_kualitatif', 'endurance_vo2max_kualitatif']].replace(skala_ml)
    y_ml = df['histologi_mtor_persen']

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_ml, y_ml)

    bobot_df = pd.DataFrame({
        "Variabel Kualitatif": ["mTOR ELISA", "Kekuatan Otot", "Endurance & VO2 Max"],
        "Kontribusi Pengaruh (%)": (rf_model.feature_importances_ * 100).round(1)
    }).set_index("Variabel Kualitatif")

    st.bar_chart(bobot_df)
    st.caption("Grafik menunjukkan seberapa besar pengaruh tiap indikator kualitatif dalam memprediksi skor histologi mTOR.")
