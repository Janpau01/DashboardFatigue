import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Dashboard Kelelahan Kognitif",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# LOAD DATASET CORE
# =========================================================

@st.cache_data
def load_data():

    # DATASET CORE
    df = pd.read_csv(
        "data/screen_time_mentalwellness.csv"
    )

    # =====================================================
    # RENAME KOLOM
    # =====================================================

    df = df.rename(columns={

        'screen_time_hours':
        'screen_time',

        'sleep_hours':
        'sleep_hours',

        'stress_level_0_10':
        'stress_level',

        'productivity_0_100':
        'productivity',

        'mental_wellness_index_0_100':
        'wellness_index',

        'daily_social_media_hours':
        'social_media',

        'daily_exercise_minutes':
        'exercise_minutes',

        'caffeine_intake_mg_per_day':
        'caffeine',

        'age':
        'age'
    })

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    df['fatigue_score'] = (
        (df['screen_time'] * 0.35) +
        ((10 - df['sleep_hours']) * 0.30) +
        (df['stress_level'] * 0.35)
    )

    # =====================================================
    # KATEGORI FATIGUE
    # =====================================================

    df['fatigue_category'] = np.where(
        df['fatigue_score'] < 5,
        'Rendah',

        np.where(
            df['fatigue_score'] < 7,
            'Sedang',
            'Tinggi'
        )
    )

    # =====================================================
    # SAMPLE DATA AGAR STREAMLIT CEPAT
    # =====================================================

    if len(df) > 5000:

        df = df.sample(
            5000,
            random_state=42
        )

    return df


df = load_data()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.metric-box {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 15px;
}

.insight-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #FF4B4B;
}

.recommend-box {
    background-color: #1F2937;
    padding: 20px;
    border-radius: 15px;
    border-left: 5px solid #00CC96;
}

.big-font {
    font-size: 28px;
    font-weight: bold;
}

.medium-font {
    font-size: 18px;
    line-height: 1.8;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("🧠 Dashboard Analisis Kelelahan Kognitif")

st.markdown("""
Dashboard ini digunakan untuk menganalisis hubungan aktivitas digital harian Anda terhadap:

- tingkat stres,
- kualitas tidur,
- produktivitas,
- dan risiko kelelahan kognitif.
""")

st.markdown("---")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📌 Ringkasan",
    "📊 Analisis",
    "🤖 Deteksi AI"
])

# =========================================================
# TAB 1 — RINGKASAN
# =========================================================

with tab1:

    st.header("📌 Ringkasan Utama")

    avg_screen = round(
        df['screen_time'].mean(),
        2
    )

    avg_sleep = round(
        df['sleep_hours'].mean(),
        2
    )

    avg_stress = round(
        df['stress_level'].mean(),
        2
    )

    high_risk = len(
        df[df['fatigue_category'] == 'Tinggi']
    )

    risk_percent = round(
        (high_risk / len(df)) * 100,
        1
    )

    # =====================================================
    # METRIC
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rata-rata Screen Time",
            f"{avg_screen} jam"
        )

    with col2:
        st.metric(
            "Rata-rata Durasi Tidur",
            f"{avg_sleep} jam"
        )

    with col3:
        st.metric(
            "Rata-rata Tingkat Stres",
            f"{avg_stress}/10"
        )

    with col4:
        st.metric(
            "Risiko Fatigue Tinggi",
            f"{risk_percent}%"
        )

    st.markdown("---")

    # =====================================================
    # INSIGHT
    # =====================================================

    st.subheader("🔍 Insight Utama")

    col1, col2 = st.columns([1.5, 1])

    # =====================================================
    # VISUALISASI
    # =====================================================

    with col1:

        fig = px.scatter(
            df,
            x='screen_time',
            y='fatigue_score',
            color='stress_level',
            title='Hubungan Screen Time dan Kelelahan Kognitif',
            opacity=0.6
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # INSIGHT BOX
    # =====================================================

    with col2:

        st.markdown(f"""
        <div class="insight-box">

        <p class="big-font">
        Hasil Analisis
        </p>

        <p class="medium-font">

        • {risk_percent}% pengguna memiliki risiko fatigue tinggi.

        <br>

        • Pengguna dengan screen time tinggi mengalami peningkatan stres.

        <br>

        • Durasi tidur rendah menyebabkan fatigue lebih tinggi.

        <br>

        • Produktivitas menurun ketika fatigue meningkat.

        </p>

        </div>
        """, unsafe_allow_html=True)

        # =================================================
        # PENJELASAN FATIGUE
        # =================================================

        st.markdown("""
        <div style="
        background-color:#111827;
        padding:20px;
        border-radius:15px;
        border-left:5px solid #3B82F6;
        margin-top:20px;
        ">

        <p style="
        font-size:26px;
        font-weight:bold;
        margin-bottom:15px;
        ">

        📘 Penjelasan Cognitive Fatigue

        </p>

        <p style="
        font-size:17px;
        line-height:1.8;
        ">

        <b>Cognitive fatigue</b> atau kelelahan kognitif merupakan kondisi
        penurunan kemampuan mental akibat aktivitas digital berlebihan,
        kurang tidur, peningkatan stres, serta overload informasi digital.

        <br><br>

        Kondisi ini dapat menyebabkan:

        <ul>
        <li>Penurunan fokus dan konsentrasi</li>
        <li>Produktivitas kerja menurun</li>
        <li>Kesulitan mengambil keputusan</li>
        <li>Kelelahan mental (mental exhaustion)</li>
        <li>Peningkatan risiko burnout</li>
        </ul>

        Semakin tinggi tingkat fatigue pengguna, maka semakin besar risiko
        terjadinya gangguan performa kognitif dan kesehatan mental.

        </p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2 — ANALISIS
# =========================================================

with tab2:

    st.header("📊 Analisis Kelelahan Kognitif")

    col1, col2 = st.columns(2)

    # =====================================================
    # VISUALISASI 1
    # =====================================================

    with col1:

        fig2 = px.scatter(
            df,
            x='sleep_hours',
            y='stress_level',
            color='fatigue_category',
            title='Durasi Tidur vs Tingkat Stres'
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # =====================================================
    # VISUALISASI 2
    # =====================================================

    with col2:

        fig3 = px.scatter(
            df,
            x='fatigue_score',
            y='productivity',
            color='fatigue_category',
            title='Fatigue vs Produktivitas'
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    st.markdown("---")

    # =====================================================
    # DISTRIBUSI FATIGUE
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        fatigue_distribution = (
            df['fatigue_category']
            .value_counts()
            .reset_index()
        )

        fatigue_distribution.columns = [
            'Kategori',
            'Jumlah'
        ]

        fig4 = px.pie(
            fatigue_distribution,
            values='Jumlah',
            names='Kategori',
            title='Distribusi Tingkat Fatigue'
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    # =====================================================
    # KESIMPULAN ANALISIS
    # =====================================================

    with col2:

        total_data = len(df)

        tinggi = round(
            (
                len(df[df['fatigue_category'] == 'Tinggi'])
                / total_data
            ) * 100,
            1
        )

        sedang = round(
            (
                len(df[df['fatigue_category'] == 'Sedang'])
                / total_data
            ) * 100,
            1
        )

        rendah = round(
            (
                len(df[df['fatigue_category'] == 'Rendah'])
                / total_data
            ) * 100,
            1
        )

        st.markdown(f"""
        <div class="recommend-box">

        <p class="big-font">
        Kesimpulan Analisis
        </p>

        <p class="medium-font">

        • Sebanyak <b>{tinggi}%</b> pengguna berada pada kategori 
        <b>fatigue tinggi</b>, yang menunjukkan tingginya risiko 
        kelelahan mental akibat aktivitas digital berlebihan.

        <br><br>

        • Sebanyak <b>{sedang}%</b> pengguna berada pada kategori 
        <b>fatigue sedang</b>, yang menunjukkan mulai munculnya 
        penurunan fokus dan produktivitas.

        <br><br>

        • Sebanyak <b>{rendah}%</b> pengguna berada pada kategori 
        <b>fatigue rendah</b>, yang menunjukkan kondisi mental 
        relatif stabil dan sehat.

        <br><br>

        • Aktivitas digital berlebihan, durasi tidur rendah, dan tingkat stres tinggi 
        merupakan faktor utama peningkatan cognitive fatigue.

        </p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3 — DETEKSI AI
# =========================================================

with tab3:

    st.header("🤖 Deteksi Dini Kelelahan Kognitif")

    st.markdown("""
    Masukkan aktivitas harian Anda untuk mendeteksi tingkat kelelahan kognitif.
    """)


    # =====================================================
    # FORM
    # =====================================================

    with st.form("fatigue_form"):

        col1, col2 = st.columns(2)

        with col1:

            screen_time = st.slider(
                "Durasi Screen Time (jam/hari)",
                0.0,
                24.0,
                7.0
            )

            sleep_hours = st.slider(
                "Durasi Tidur",
                0.0,
                12.0,
                6.0
            )

            stress_level = st.slider(
                "Tingkat Stres",
                1,
                10,
                5
            )

        with col2:

            social_media = st.slider(
                "Penggunaan Media Sosial",
                0.0,
                15.0,
                4.0
            )

            productivity = st.slider(
                "Produktivitas",
                1,
                100,
                70
            )

            exercise = st.slider(
                "Durasi Olahraga (menit)",
                0,
                120,
                30
            )

        submitted = st.form_submit_button(
            "🔍 Analisis Kelelahan"
        )

    # =====================================================
    # HASIL
    # =====================================================

    if submitted:

        fatigue_score = (
            (screen_time * 0.35) +
            ((10 - sleep_hours) * 0.30) +
            (stress_level * 0.35)
        )

        fatigue_percent = min(
            int(fatigue_score * 10),
            100
        )

        st.subheader("📊 Hasil Deteksi")

        st.progress(fatigue_percent)

        st.metric(
            "Persentase Kelelahan Kognitif",
            f"{fatigue_percent}%"
        )

        # =================================================
        # KATEGORI
        # =================================================

        if fatigue_percent < 40:

            category = "🟢 Fatigue Rendah"

            explanation = """
            Kondisi pengguna masih cukup sehat dan stabil.
            """

        elif fatigue_percent < 70:

            category = "🟡 Fatigue Sedang"

            explanation = """
            Pengguna mulai mengalami kelelahan kognitif ringan.
            """

        else:

            category = "🔴 Fatigue Tinggi"

            explanation = """
            Pengguna mengalami tingkat kelelahan kognitif tinggi.
            """

        st.markdown(f"## {category}")

        st.info(explanation)

        # =================================================
        # REKOMENDASI
        # =================================================

        st.header("💡 Rekomendasi Cerdas")

        recommendations = []

        if screen_time > 8:
            recommendations.append(
                "Kurangi screen time harian."
            )

        if sleep_hours < 6:
            recommendations.append(
                "Tingkatkan kualitas tidur menjadi 7–8 jam."
            )

        if stress_level > 7:
            recommendations.append(
                "Lakukan manajemen stres dan relaksasi."
            )

        if social_media > 6:
            recommendations.append(
                "Batasi penggunaan media sosial."
            )

        if productivity < 60:
            recommendations.append(
                "Tingkatkan manajemen waktu dan fokus kerja."
            )

        if exercise < 20:
            recommendations.append(
                "Tambahkan aktivitas olahraga harian."
            )

        if len(recommendations) == 0:

            st.success("""
            Pengguna memiliki pola aktivitas digital yang cukup sehat.
            """)

        else:

            for rec in recommendations:
                st.warning(rec)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Dashboard Pengembangan Sistem Deteksi Dini Kelelahan Kognitif
Berbasis Aktivitas Harian
""")