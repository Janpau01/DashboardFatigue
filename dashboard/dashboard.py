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
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

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
        'caffeine'
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
    # SAMPLE DATA
    # =====================================================

    if len(df) > 1000:

        df = df.sample(
            1000,
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
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("🧠 Dashboard Analisis Kelelahan Kognitif")

st.markdown("""
Dashboard ini digunakan untuk menganalisis hubungan aktivitas digital harian terhadap:

- tingkat stres,
- kualitas tidur,
- produktivitas,
- dan risiko kelelahan kognitif.
""")

st.markdown("---")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Ringkasan",
    "📊 Analisis",
    "🧠 Deteksi Kelelahan",
    "📝 Catatan Dashboard"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.header("📌 Ringkasan Utama")

    avg_screen = round(df['screen_time'].mean(), 2)
    avg_sleep = round(df['sleep_hours'].mean(), 2)
    avg_stress = round(df['stress_level'].mean(), 2)

    high_risk = len(
        df[df['fatigue_category'] == 'Tinggi']
    )

    risk_percent = round(
        (high_risk / len(df)) * 100,
        1
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rata-rata Penggunaan Gadget",
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

    st.subheader("🔍 Insight Utama")

    col1, col2 = st.columns([1.5, 1])

    # =====================================================
    # CHART
    # =====================================================

    with col1:

        fig = px.scatter(
            df,
            x='screen_time',
            y='fatigue_score',
            color='stress_level',
            title='Hubungan Penggunaan Gadget dan Kelelahan Kognitif',
            opacity=0.6
        )

        fig.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # INSIGHT
    # =====================================================

    with col2:

        st.markdown(f"""
        <div class="insight-box">

        <p class="big-font">
        Hasil Analisis
        </p>

        <p class="medium-font">

        • {risk_percent}% responden memiliki risiko fatigue tinggi.

        <br>

        • Responden dengan penggunaan gadget tinggi mengalami peningkatan stres.

        <br>

        • Durasi tidur rendah menyebabkan fatigue lebih tinggi.

        <br>

        • Produktivitas menurun ketika fatigue meningkat.

        </p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 2
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

        fig2.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="white"
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

        fig3.update_layout(
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="white"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

# =====================================================
# PIE CHART
# =====================================================

with col1:

    fatigue_distribution = (
        df['fatigue_category']
        .value_counts()
        .reset_index()
    )

    fatigue_distribution.columns = [
        'Kategori_Asli',
        'Jumlah'
    ]

    # =================================================
    # RENAME LABEL
    # =================================================

    fatigue_distribution['Kategori'] = (
        fatigue_distribution['Kategori_Asli']
        .replace({

            'Tinggi':
            '🔴 Near-Burnout',

            'Sedang':
            '🟡 Strained',

            'Rendah':
            '🟢 Refreshed'
        })
    )

    fig4 = px.pie(
        fatigue_distribution,
        values='Jumlah',
        names='Kategori',
        title='Distribusi Kondisi Mental'
    )

    fig4.update_layout(
        paper_bgcolor="#0E1117",
        font_color="white"
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

        • Sebanyak <b>{tinggi}%</b> responden berada pada kondisi
        <b>🔴 Near-Burnout</b>, yang menunjukkan tingkat
        kelelahan mental tinggi akibat aktivitas digital berlebihan.

        <br><br>

        • Sebanyak <b>{sedang}%</b> responden berada pada kondisi
        <b>🟡 Strained</b>, yang menunjukkan mulai munculnya
        tekanan mental dan penurunan fokus.

        <br><br>

        • Sebanyak <b>{rendah}%</b> responden berada pada kondisi
        <b>🟢 Refreshed</b>, yang menunjukkan kondisi mental
        relatif stabil dan sehat.

        <br><br>

        • Aktivitas digital berlebihan dan kurang tidur
        menjadi faktor utama peningkatan cognitive fatigue.

        </p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.header("🧠 Deteksi Dini Kelelahan Kognitif")

    st.markdown("""
    Masukkan aktivitas harian Anda untuk mendeteksi tingkat kelelahan kognitif.
    """)

    with st.form("fatigue_form"):

        col1, col2 = st.columns(2)

        with col1:

            screen_time = st.slider(
                "Durasi Penggunaan Gadget (jam/hari)",
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
    # OUTPUT
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
        # STATUS FATIGUE
        # =================================================

        if fatigue_percent < 40:

            category = "🟢 Refreshed"

            explanation = """
            Kondisi mental Anda masih stabil,
            fokus masih terjaga,
            dan aktivitas digital belum memberikan
            tekanan kognitif berlebihan.
            """

        elif fatigue_percent < 70:

            category = "🟡 Strained"

            explanation = """
            Anda mulai mengalami tekanan mental
            dan kelelahan kognitif ringan akibat
            aktivitas digital dan stres harian.
            """

        else:

            category = "🔴 Near-Burnout"

            explanation = """
            Kondisi mental Anda menunjukkan tanda-tanda
            kelelahan tinggi dan mendekati burnout.

            Disarankan untuk segera melakukan
            recovery dan mengurangi overstimulasi digital.
            """

        st.markdown(f"## {category}")

        st.warning(explanation)

        # =================================================
        # REKOMENDASI CERDAS
        # =================================================

        st.header("💡 Rekomendasi Cerdas")

        recommendations = []

        if screen_time > 8:

            recommendations.append(
                "Kurangi penggunaan gadget harian."
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
                "Gunakan teknik manajemen waktu."
            )

        if exercise < 20:

            recommendations.append(
                "Tambahkan aktivitas olahraga."
            )

        if len(recommendations) == 0:

            st.success("""
            Anda memiliki pola aktivitas digital yang cukup sehat.
            """)

        else:

            for rec in recommendations:

                st.info(rec)

        # =================================================
        # BRAIN RECOVERY SYSTEM
        # =================================================

        st.markdown("---")

        st.header("🧠 Brain Recovery System")

        brainrot_score = 0

        if screen_time > 8:
            brainrot_score += 30

        if social_media > 6:
            brainrot_score += 25

        if sleep_hours < 6:
            brainrot_score += 25

        if stress_level > 7:
            brainrot_score += 20

        if brainrot_score < 30:

            brainrot_category = "🟢 Risiko Brainrot Rendah"

            brainrot_desc = """
            Pola aktivitas digital Anda masih relatif sehat.
            """

        elif brainrot_score < 60:

            brainrot_category = "🟡 Risiko Brainrot Sedang"

            brainrot_desc = """
            Anda mulai menunjukkan gejala overstimulasi digital.
            """

        else:

            brainrot_category = "🔴 Risiko Brainrot Tinggi"

            brainrot_desc = """
            Anda menunjukkan indikasi brainrot tinggi.
            """

        st.subheader(brainrot_category)

        st.warning(brainrot_desc)

        st.markdown("""
        ### 🧘 Rekomendasi Pemulihan Otak
        """)

        recovery = []

        if screen_time > 8:

            recovery.append(
                "📵 Lakukan pembatasan digital minimal 1–2 jam tanpa gadget."
            )

        if social_media > 6:

            recovery.append(
                "📱 Kurangi konsumsi short-form content."
            )

        if sleep_hours < 6:

            recovery.append(
                "😴 Tingkatkan kualitas tidur menjadi 7–8 jam."
            )

        if stress_level > 7:

            recovery.append(
                "🧘 Lakukan mindfulness atau relaksasi."
            )

        if productivity < 60:

            recovery.append(
                "🎯 Gunakan teknik deep work atau Pomodoro."
            )

        if exercise < 20:

            recovery.append(
                "🏃 Lakukan olahraga ringan."
            )

        if len(recovery) == 0:

            st.success("""
            Anda memiliki pola digital yang cukup sehat.
            """)

        else:

            for item in recovery:

                st.success(item)

# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.header("📝 Catatan Dashboard")

    st.markdown("""
    Halaman ini berisi informasi tambahan mengenai
    kelelahan kognitif, dampak overstimulasi digital,
    serta proses pemulihan fungsi otak.
    """)

    st.markdown("---")

    st.markdown("""
    <div style="
    background-color:#111827;
    padding:25px;
    border-radius:15px;
    border-left:5px solid #3B82F6;
    margin-bottom:20px;
    ">

    <p style="
    font-size:28px;
    font-weight:bold;
    margin-bottom:15px;
    ">

    📘 Penjelasan Cognitive Fatigue

    </p>

    <p style="
    font-size:18px;
    line-height:1.5;
    margin:0;
    padding:0;
    ">

    Cognitive fatigue atau kelelahan kognitif merupakan kondisi
    penurunan kemampuan mental akibat aktivitas digital berlebihan,
    kurang tidur, peningkatan stres, serta overload informasi digital.

    <br>

    Kondisi ini dapat menyebabkan:

    <ul style="
    line-height:1.5;
    margin-top:10px;
    ">

    <li>Penurunan fokus dan konsentrasi</li>
    <li>Produktivitas kerja menurun</li>
    <li>Kesulitan mengambil keputusan</li>
    <li>Kelelahan mental (mental exhaustion)</li>
    <li>Peningkatan risiko burnout</li>

    </ul>

    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
    background-color:#1F2937;
    padding:25px;
    border-radius:15px;
    border-left:5px solid #EF4444;
    margin-bottom:20px;
    ">

    <p style="
    font-size:28px;
    font-weight:bold;
    margin-bottom:15px;
    ">

    ⚠️ Dampak Potensial

    </p>

    <p style="
    font-size:18px;
    line-height:1.5;
    margin:0;
    padding:0;
    ">

    Konsumsi konten digital berlebihan dapat menyebabkan:

    <ul style="
    line-height:1.5;
    margin-top:10px;
    ">

    <li>Penurunan fokus dan konsentrasi</li>
    <li>Overstimulasi dopamin akibat konten instan</li>
    <li>Mental exhaustion atau kelelahan mental</li>
    <li>Kesulitan melakukan deep work</li>
    <li>Motivasi belajar dan produktivitas menurun</li>
    <li>Gangguan kualitas tidur</li>
    <li>Peningkatan stres dan kecemasan</li>

    </ul>

    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
    background-color:#111827;
    padding:25px;
    border-radius:15px;
    border-left:5px solid #10B981;
    ">

    <p style="
    font-size:28px;
    font-weight:bold;
    margin-bottom:15px;
    ">

    🧠 Neuroplasticity Recovery Insight

    </p>

    <p style="
    font-size:18px;
    line-height:1.5;
    margin:0;
    padding:0;
    ">

    Otak manusia memiliki kemampuan neuroplasticity,
    yaitu kemampuan untuk membentuk ulang jalur saraf
    berdasarkan kebiasaan baru.

    <br>

    Artinya:
    brainrot atau kelelahan akibat overstimulasi digital
    bukan kondisi permanen.

    <br>

    Kebiasaan sehat seperti:

    <ul style="
    line-height:1.5;
    margin-top:10px;
    ">

    <li>Tidur cukup 7–8 jam</li>
    <li>Mengurangi durasi penggunaan gadget berlebihan</li>
    <li>Olahraga rutin</li>
    <li>Membaca buku</li>
    <li>Melatih deep work dan fokus</li>
    <li>Mengurangi short-form content</li>

    </ul>

    dapat membantu memulihkan fokus,
    konsentrasi, dan kesehatan mental secara bertahap.

    </p>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Dashboard Pengembangan Sistem Deteksi Dini Kelelahan Kognitif
Berbasis Aktivitas Harian
""")