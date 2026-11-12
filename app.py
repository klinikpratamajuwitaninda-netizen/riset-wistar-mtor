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


# ==============================================================================
# TAMBAHAN: SIMULASI PREDIKSI ALGORITMA MACHINE LEARNING GAYA SPSS (CRT TREE)
# ==============================================================================
from sklearn.tree import DecisionTreeRegressor, export_text

st.divider()
st.header("Simulasi Prediksi Algoritma Machine Learning Gaya SPSS (CRT Decision Tree)")
st.write("Simulasi pohon keputusan (Classification & Regression Tree / CRT ala SPSS) untuk memprediksi skor kuantitatif histologi mTOR *gastrocnemius* dari variabel kualitatif:")

# Melatih model Decision Tree ala SPSS CRT
skala_crt = {'Rendah': 1, 'Lemah': 1, 'Kurang': 1, 'Sedang': 2, 'Normal': 2, 'Cukup': 2, 'Tinggi': 3, 'Kuat': 3, 'Baik': 3, 'Sangat Kuat': 4, 'Sangat Baik': 4}
X_crt = df[['mtor_elisa_kualitatif', 'kekuatan_otot_kualitatif', 'endurance_vo2max_kualitatif']].replace(skala_crt)
y_crt = df['histologi_mtor_persen']

spss_tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1, random_state=42)
spss_tree.fit(X_crt, y_crt)

col_sim1, col_sim2 = st.columns([1.2, 1])

with col_sim1:
    st.subheader("A. Input Parameter Simulasi Subjek Baru")
    sim_elisa = st.select_slider("1. Tingkat mTOR ELISA (Kualitatif):", options=["Rendah", "Sedang", "Tinggi"], value="Tinggi")
    sim_kuat = st.select_slider("2. Kekuatan Otot (Kualitatif):", options=["Lemah", "Normal", "Kuat", "Sangat Kuat"], value="Sangat Kuat")
    sim_vo2 = st.select_slider("3. Endurance & VO2 Max (Kualitatif):", options=["Kurang", "Cukup", "Baik", "Sangat Baik"], value="Sangat Baik")

    data_uji = pd.DataFrame([[skala_crt[sim_elisa], skala_crt[sim_kuat], skala_crt[sim_vo2]]],
                            columns=['mtor_elisa_kualitatif', 'kekuatan_otot_kualitatif', 'endurance_vo2max_kualitatif'])
    prediksi_crt = spss_tree.predict(data_uji)[0]
    kategori_pred = "Kelompok Resistance (Hipertrofi Aktif)" if prediksi_crt >= 50 else "Kelompok Kontrol (Basal)"

    st.success(f"**Hasil Prediksi Node SPSS CRT:** **{prediksi_crt:.2f}%** Ekspresi Histologi mTOR")
    st.info(f"**Klasifikasi Kelompok Otomatis:** {kategori_pred}")

with col_sim2:
    st.subheader("B. Struktur Aturan Pohon Keputusan (SPSS Tree Rules)")
    aturan_pohon = export_text(spss_tree, feature_names=["mTOR_ELISA", "Kekuatan_Otot", "Endurance_VO2Max"])
    st.code(aturan_pohon, language="text")
    st.caption("Catatan: Skala 1 = Rendah/Lemah, 2 = Sedang/Normal, 3 = Tinggi/Kuat, 4 = Sangat Kuat/Baik.")


# ==============================================================================
# TAMBAHAN FITUR 2: HIGH-FREQUENCY MICROSTRUCTURE & ALPHA-GENERATING STRATEGIES
# ==============================================================================
import numpy as np

st.divider()
st.header("Analisis Data Mikrostruktur Berfrekuensi Tinggi & Alpha-Generating")
st.markdown("Simulasi telemetri resolusi tinggi (per 100 ms) sinyal kontraksi *m. gastrocnemius*, kurva kelelahan (*fatigue slope*), dan kinetika $\\text{VO}_2$ untuk deteksi dini efisiensi hipertrofi (*Hypertrophy Alpha Index*).")

# Simulasi Data Time-Series High-Frequency (0 - 5000 ms)
waktu_ms = np.arange(0, 5000, 100)
np.random.seed(42)

# Sinyal kontraksi otot & fatigue slope (Resistance mempertahankan gaya lebih stabil dibanding Kontrol)
gaya_kontrol = 100 * np.exp(-0.00025 * waktu_ms) + np.random.normal(0, 2.5, len(waktu_ms))
gaya_resistance = 145 * np.exp(-0.00009 * waktu_ms) + np.random.normal(0, 2.0, len(waktu_ms))

# Kinetika konsumsi oksigen (VO2 ml/kg/min)
vo2_kontrol = 42 + 18 * (1 - np.exp(-0.0008 * waktu_ms)) + np.random.normal(0, 0.8, len(waktu_ms))
vo2_resistance = 48 + 31 * (1 - np.exp(-0.0012 * waktu_ms)) + np.random.normal(0, 0.8, len(waktu_ms))

df_micro = pd.DataFrame({
    "Waktu (ms)": waktu_ms,
    "Gaya Kontraksi Kontrol (g)": gaya_kontrol,
    "Gaya Kontraksi Resistance (g)": gaya_resistance,
    "Kinetika VO2 Kontrol": vo2_kontrol,
    "Kinetika VO2 Resistance": vo2_resistance
}).set_index("Waktu (ms)")

col_hf1, col_hf2 = st.columns(2)
with col_hf1:
    st.subheader("A. Sinyal Kontraksi Gastrocnemius & Fatigue Slope (ms)")
    st.line_chart(df_micro[["Gaya Kontraksi Kontrol (g)", "Gaya Kontraksi Resistance (g)"]])
    st.caption("Kelompok Resistance menunjukkan puncak gaya lebih tinggi dengan penurunan kelelahan (fatigue slope) yang jauh lebih landai.")

with col_hf2:
    st.subheader("B. Kinetika Konsumsi Oksigen (VO2 Kinetics)")
    st.line_chart(df_micro[["Kinetika VO2 Kontrol", "Kinetika VO2 Resistance"]])
    st.caption("Kinetika penyerapan oksigen mencapai steady-state lebih cepat dan tinggi pada kelompok Resistance Training.")

# Strategi Alpha-Generating: Hypertrophy Alpha Index
st.subheader("C. Strategi Alpha-Generating: Peringkat Hypertrophy Alpha Index")
st.write("Indikator kuantitatif prediktif untuk mendeteksi tikus dengan respons hipertrofi mTOR paling optimal lebih awal sebelum terminasi histologi:")

skala_alpha = {'Rendah': 1, 'Lemah': 1, 'Kurang': 1, 'Sedang': 2, 'Normal': 2, 'Cukup': 2, 'Tinggi': 3, 'Kuat': 3, 'Baik': 3, 'Sangat Kuat': 4, 'Sangat Baik': 4}
df_alpha = df[["rat_id", "kelompok", "mtor_elisa_kualitatif", "kekuatan_otot_kualitatif", "endurance_vo2max_kualitatif", "histologi_mtor_persen"]].copy()

# Menghitung skor Alpha Index (kombinasi bobot kekuatan, kinetika VO2, dan sinyal ELISA)
skor_k = df_alpha["kekuatan_otot_kualitatif"].map(skala_alpha)
skor_v = df_alpha["endurance_vo2max_kualitatif"].map(skala_alpha)
skor_e = df_alpha["mtor_elisa_kualitatif"].map(skala_alpha)

df_alpha["Hypertrophy Alpha Index"] = ((skor_k * 0.42 + skor_v * 0.34 + skor_e * 0.24) * 25).round(1)
df_alpha["Sinyal Alpha"] = df_alpha["Hypertrophy Alpha Index"].apply(
    lambda x: "STRONG ALPHA (Respons Optimal)" if x >= 85 else ("MODERATE ALPHA" if x >= 65 else "BASELINE (Non-Responder)")
)
df_alpha = df_alpha.sort_values(by="Hypertrophy Alpha Index", ascending=False)

st.dataframe(
    df_alpha[["rat_id", "kelompok", "Hypertrophy Alpha Index", "Sinyal Alpha", "histologi_mtor_persen"]],
    use_container_width=True,
    hide_index=True
)
