import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

st.set_page_config(page_title="🎓 Prediksi Kelulusan Mahasiswa", layout="wide")

# Sidebar
with st.sidebar:
    st.title("🔧 Konfigurasi")
    uploaded_file = st.file_uploader("📁 Upload Dataset Mahasiswa (CSV)", type=["csv"])
    run = st.button("🚀 Jalankan Prediksi")

# Judul utama
st.title("🎓 Aplikasi Prediksi Kelulusan Mahasiswa")
st.markdown("Gunakan model **Logistic Regression** untuk memprediksi kelulusan berdasarkan faktor sosial dan akademik.")

# Alur Sistem
with st.expander("🧭 Lihat Alur Sistem"):
    st.markdown("""
    **Alur Sistem:**
    1. Pengguna mengunggah file dataset (.csv) atau mengisi data manual
    2. Data ditampilkan dalam tabel (preview)
    3. Fitur dan target dipisahkan (fitur: semua kolom numerik + kategorikal; target: kolom status kelulusan)
    4. Dataset dibagi menjadi 80% data latih dan 20% data uji
    5. Model Logistic Regression dilatih dengan data latih
    6. Model diuji dengan data uji
    7. Evaluasi dilakukan dengan akurasi, confusion matrix, dan classification report
    8. Sistem menampilkan prediksi dan performa model
    9. Sistem menampilkan koefisien regresi dan interpretasi pengaruh fitur
    """)

# Input manual data
with st.expander("✍️ Input Data Mahasiswa Manual"):
    col1, col2, col3 = st.columns(3)
    with col1:
        ipk = st.number_input("IPK", 0.0, 4.0, step=0.01)
        hadir = st.slider("Kehadiran (%)", 0, 100, 75)
    with col2:
        sks = st.number_input("Jumlah SKS", 0, 160, step=1)
        organisasi = st.selectbox("Aktif Organisasi", ["Ya", "Tidak"])
    with col3:
        kerja = st.selectbox("Status Kerja", ["Tidak bekerja", "Paruh waktu", "Penuh waktu"])
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

    predict_manual = st.button("🔍 Prediksi Kelulusan dari Input Manual")

# Fungsi utama
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Mahasiswa (5 Data Teratas)")
    st.dataframe(df.head())

    if run:
        possible_targets = [col for col in df.columns if 'lulus' in col.lower() or 'kelulusan' in col.lower()]
        if possible_targets:
            target_column = possible_targets[0]
        else:
            st.error("❌ Kolom target kelulusan tidak ditemukan di dataset.")
            st.stop()

        if target_column not in df.columns:
            st.error(f"❌ Kolom '{target_column}' tidak ditemukan dalam dataset.")
            st.stop()

        X = df.drop(columns=[target_column])
        y = df[target_column]

        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        st.session_state.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        st.session_state.model = model  # Simpan model

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.subheader("✅ Hasil Evaluasi Model")
        st.metric(label="🎯 Akurasi", value=f"{acc*100:.2f}%")

        st.markdown("### 🔍 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="coolwarm", ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

        st.markdown("### 📝 Classification Report")
        st.code(classification_report(y_test, y_pred), language='text')

        st.markdown("### 💡 Interpretasi Faktor Berpengaruh")
        coef = model.coef_[0]
        feature_importance = pd.DataFrame({
            'Fitur': X.columns,
            'Koefisien': coef,
            'Pengaruh Absolut': np.abs(coef)
        }).sort_values(by='Pengaruh Absolut', ascending=False)

        st.dataframe(feature_importance[['Fitur', 'Koefisien']])

        st.markdown("### 📊 Visualisasi Pengaruh Fitur")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Pengaruh Absolut', y='Fitur', data=feature_importance, palette='crest', ax=ax2)
        ax2.set_title("Pengaruh Fitur terhadap Prediksi Kelulusan")
        st.pyplot(fig2)

        st.markdown("---")
        st.markdown("📌 **Penjelasan:**")
        st.markdown("- Nilai **koefisien positif** menunjukkan fitur tersebut meningkatkan kemungkinan kelulusan.")
        st.markdown("- Nilai **koefisien negatif** menunjukkan fitur tersebut mengurangi kemungkinan kelulusan.")
        st.markdown("- Fitur dengan nilai absolut koefisien terbesar adalah faktor paling berpengaruh.")

# Prediksi input manual
if predict_manual and 'feature_names' in st.session_state and 'model' in st.session_state:
    input_df = pd.DataFrame({
        'IPK_rata_rata': [ipk],
        'Kehadiran_%': [hadir],
        'Jumlah_SKS': [sks],
        'Aktif_Organisasi': [1 if organisasi == "Ya" else 0],
        'Status_Kerja': ["Tidak bekerja", "Paruh waktu", "Penuh waktu"].index(kerja),
        'Jenis_Kelamin': [0 if gender == "Laki-laki" else 1]
    })

    try:
        input_df = input_df[st.session_state.feature_names]
        prediction = st.session_state.model.predict(input_df)[0]
        result_text = "✅ Diprediksi **LULUS**" if prediction == 1 else "❌ Diprediksi **TIDAK LULUS**"

        if prediction == 1:
            st.balloons()
        else:
            st.warning("🎈 Mahasiswa diprediksi belum lulus. Perlu peningkatan faktor pendukung.")

        st.success(f"Hasil Prediksi: {result_text}")
    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
else:
    st.info("📥 Silakan upload dataset CSV dan jalankan prediksi terlebih dahulu.")
