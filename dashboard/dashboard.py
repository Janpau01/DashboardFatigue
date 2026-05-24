import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from datetime import datetime

# =====================================================
# FILE HISTORY
# =====================================================

history_file = "progress_history.csv"

# =====================================================
# SESSION STATE
# =====================================================

if 'wellness_result' not in st.session_state:

    st.session_state.wellness_result = None

# =====================================================
# LOAD HISTORY
# =====================================================

if 'progress_history' not in st.session_state:

    if os.path.exists(history_file):

        history_df = pd.read_csv(
            history_file
        )

        st.session_state.progress_history = (
            history_df.to_dict('records')
        )

    else:

        st.session_state.progress_history = []

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

# =====================================================
# LOAD MODEL MACHINE LEARNING
# =====================================================

model = joblib.load(
    "model/fatigue_model.pkl"
)

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
Dashboard ini membantu Anda memahami hubungan
antara penggunaan gadget, kualitas tidur,
tingkat stres, dan kondisi mental harian.

Sistem juga memberikan insight wellness
serta rekomendasi untuk menjaga fokus
dan kesehatan digital.
""")

st.markdown("---")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Ringkasan",
    "📊 Insight",
    "🌿 Wellness Check",
    "🌱 Recovery Center",
    "📈 Recovery Journey",
    "📘 Panduan & Insight"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.header("🏠 Ringkasan Utama")

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

    # =====================================================
    # LAYOUT INSIGHT
    # =====================================================

    col1, col2 = st.columns([2, 1])

    # =====================================================
    # CHART ANALISIS
    # =====================================================

    with col1:

        # ================================================
        # KATEGORI PENGGUNAAN GADGET
        # ================================================

        df['kategori_gadget'] = pd.cut(
            df['screen_time'],
            bins=[0, 5, 8, 24],
            labels=[
                'Rendah',
                'Sedang',
                'Tinggi'
            ]
        )

        # ================================================
        # RATA-RATA FATIGUE
        # ================================================

        fatigue_summary = (
            df.groupby('kategori_gadget')['fatigue_score']
            .mean()
            .reset_index()
        )

        fatigue_summary.columns = [
            'Kategori Penggunaan Gadget',
            'Rata-rata Fatigue'
        ]

        # ================================================
        # BAR CHART
        # ================================================

        fig = px.bar(
            fatigue_summary,
            x='Kategori Penggunaan Gadget',
            y='Rata-rata Fatigue',
            color='Kategori Penggunaan Gadget',
            title='Penggunaan Gadget Meningkatkan Risiko Kelelahan Mental',
            text_auto='.2f'
        )

        fig.update_layout(

            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="white",

            xaxis_title="Kategori Penggunaan Gadget",
            yaxis_title="Tingkat Kelelahan",

            showlegend=False,

            height=500
        )

        fig.update_traces(
            textposition='outside'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # HASIL ANALISIS
    # =====================================================

    with col2:

        st.markdown(f"""
        <div class="insight-box">

        <p class="big-font">
        Hasil Analisis
        </p>

        <p class="medium-font">

        • {risk_percent}% responden memiliki risiko kelelahan mental tinggi.

        <br>

        • Responden dengan penggunaan gadget tinggi mengalami peningkatan stres.

        <br>

        • Durasi tidur rendah menyebabkan kelelahan mental lebih tinggi.

        <br>

        • Produktivitas menurun ketika kelelahan mental meningkat.
        
        <br>
        
        • Semakin tinggi durasi penggunaan gadget, maka risiko kelelahan mental juga meningkat.

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
        <b>🔴 Near-Burnout</b>.

        <br>

        • Sebanyak <b>{sedang}%</b> responden berada pada kondisi
        <b>🟡 Strained</b>.

        <br>

        • Sebanyak <b>{rendah}%</b> responden berada pada kondisi
        <b>🟢 Refreshed</b>.

        <br>

        • Aktivitas digital berlebihan dan kurang tidur
        menjadi faktor utama peningkatan cognitive fatigue.

        </p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.header("🌿 Daily Mind Check")

    st.markdown("""
    Masukkan aktivitas harian Anda untuk melihat kondisi
    keseimbangan mental dan penggunaan digital sehari-hari.
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

        with st.spinner(
            "🤖 Sistem sedang menganalisis kondisi mental Anda..."
        ):

        # =====================================================
        # INPUT DATA MACHINE LEARNING
        # =====================================================

            input_data = pd.DataFrame([{

                'screen_time': screen_time,
                'sleep_hours': sleep_hours,
                'stress_level': stress_level,
                'digital_balance': 50,
                'physical_activity': exercise,
                'work_hours': 8
            }])

            # =================================================
            # PREDIKSI MACHINE LEARNING
            # =================================================

            prediction = model.predict(
                input_data
            )[0]

            # =================================================
            # SKOR RISIKO KELELAHAN MENTAL
            # =================================================

            risk_score = (
                (screen_time * 0.35) +
                ((10 - sleep_hours) * 0.30) +
                (stress_level * 0.35)
            )

            # =================================================
            # NORMALISASI PERSENTASE RISIKO
            # =================================================

            fatigue_percent = min(
                int(risk_score * 8.5),
                95
            )
            
            # =====================================================
            # SAVE RESULT TO SESSION
            # =====================================================

            st.session_state.wellness_result = {

                'screen_time': screen_time,
                'sleep_hours': sleep_hours,
                'stress_level': stress_level,
                'exercise': exercise,
                'social_media': social_media,

                'fatigue_percent': fatigue_percent,
                'prediction': prediction
            }
            
            # =====================================================
            # SAVE HISTORY
            # =====================================================

            new_data = {

                'Date': datetime.now().strftime(
                "%d-%m-%Y %H:%M"
                ),
                
                'Fatigue Risk': fatigue_percent,
                'Screen Time': screen_time,
                'Stress': stress_level,
                'Sleep': sleep_hours,
                'Exercise': exercise
            }

            if len(st.session_state.progress_history) == 0 or \
            st.session_state.progress_history[-1] != new_data:

                st.session_state.progress_history.append(
                    new_data
            )

            # =====================================================
            # SAVE TO CSV
            # =====================================================

            history_df = pd.DataFrame(
                st.session_state.progress_history
            )

            history_df.to_csv(
                history_file,
                index=False
            )
            
            # =================================================
            # INTERPRETASI RISIKO
            # =================================================

            if fatigue_percent <= 35:

                risk_label = "🟢 Stabil"

                risk_desc = """
                Kondisi mental Anda masih stabil
                dan aktivitas digital masih dalam batas aman.
                """

            elif fatigue_percent <= 65:

                risk_label = "🟡 Mulai Lelah"

                risk_desc = """
                Aktivitas digital dan stres harian mulai memberikan dampak pada fokus dan energi mental Anda.
                """

            elif fatigue_percent <= 85:

                risk_label = "🟠 Risiko Tinggi"

                risk_desc = """
                Tingkat kelelahan mental Anda cukup tinggi dan mulai mempengaruhi kualitas aktivitas harian.
                """

            else:

                risk_label = "🔴 Near-Burnout"

                risk_desc = """
                Risiko kelelahan mental Anda sangat tinggi dan mendekati kondisi burnout.
                """

            # =================================================
            # HASIL DETEKSI
            # =================================================

            st.subheader("🌿 Kondisi Digital Wellness Anda")
            
            st.success(
            "✨ Sistem berhasil menganalisis kondisi digital wellness Anda"
    )

            st.progress(fatigue_percent)

            st.metric(
                "Tingkat Risiko Kelelahan Mental",
                f"{fatigue_percent}%"
            )
            
            st.info(f"""
            {risk_label}

            {risk_desc}
            """)
            # =================================================
            # GAUGE CHART AI
            # =================================================

            gauge = go.Figure(go.Indicator(

                mode="gauge+number",

                value=fatigue_percent,

                title={
                    'text':
                    "Estimasi Kondisi Mental"
                },

                gauge={

                    'axis': {
                        'range': [0, 100]
                    },

                    'bar': {
                        'color': "#00CC96"
                    },

                    'steps': [

                        {
                            'range': [0, 40],
                            'color': "#10B981"
                        },

                        {
                            'range': [40, 70],
                            'color': "#F59E0B"
                        },

                        {
                            'range': [70, 100],
                            'color': "#EF4444"
                        }
                    ]
                }
            ))

            gauge.update_layout(

                paper_bgcolor="#0E1117",

                font={
                    'color': "white"
                },

                height=300
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )
            
            # =================================================
            # HASIL PREDIKSI MACHINE LEARNING
            # =================================================

            if prediction == "Refreshed":

                category = "🟢 Refreshed"

                explanation = """
                Kondisi mental Anda masih stabil, fokus masih terjaga,
                dan aktivitas digital belum memberikan tekanan kognitif berlebihan.
                """

            elif prediction == "Strained":

                category = "🟡 Strained"

                explanation = """
                Anda mulai mengalami tekanan mental
                dan kelelahan mental ringan akibat
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

            st.info(f"""
            📊 Semakin tinggi persentase,
            maka semakin tinggi risiko kelelahan mental
            akibat aktivitas digital dan stres harian.

            Tingkat risiko Anda saat ini:
            {fatigue_percent}%
            """)

            # =================================================
            # REKOMENDASI CERDAS
            # =================================================

            st.header("💡 Rekomendasi Aktivitas Recovery")

            recommendations = []

            # =================================================
            # SCREEN TIME TINGGI
            # =================================================

            if screen_time > 8:

                recommendations.extend([

                    "📚 Membaca buku fisik selama 20–30 menit.",

                    "🚶 Jalan santai sore tanpa membawa gadget.",

                    "🌳 Duduk santai di area terbuka untuk mengurangi overstimulasi digital.",

                    "☕ Luangkan waktu istirahat tanpa membuka media sosial."
                ])

            # =================================================
            # STRES TINGGI
            # =================================================

            if stress_level > 7:

                recommendations.extend([

                    "🧘 Melakukan meditasi atau latihan pernapasan mindfulness.",

                    "🎵 Mendengarkan musik relaksasi tanpa scrolling media sosial.",

                    "🌿 Luangkan waktu untuk relaksasi dan menenangkan pikiran.",

                    "✍️ Menulis jurnal harian untuk mengurangi tekanan mental."
                ])

            # =================================================
            # DURASI TIDUR RENDAH
            # =================================================

            if sleep_hours < 6:

                recommendations.extend([

                    "😴 Tidur lebih awal dan hindari gadget sebelum tidur.",

                    "📖 Membaca buku sebelum tidur untuk membantu relaksasi.",

                    "🛌 Ciptakan suasana kamar yang nyaman dan minim distraksi.",

                    "🌙 Kurangi konsumsi konten digital pada malam hari."
                ])

            # =================================================
            # AKTIVITAS FISIK RENDAH
            # =================================================

            if exercise < 20:

                recommendations.extend([

                    "🏃 Jogging ringan selama 15–20 menit.",

                    "🚴 Bersepeda santai di pagi atau sore hari.",

                    "🤸 Stretching atau olahraga ringan di rumah.",

                    "🚶 Tingkatkan aktivitas berjalan kaki harian."
                ])

            # =================================================
            # PRODUKTIVITAS MENURUN
            # =================================================

            if productivity < 60:

                recommendations.extend([

                    "📝 Membuat jadwal aktivitas harian secara teratur.",

                    "🎯 Gunakan teknik fokus seperti Pomodoro.",

                    "📵 Kurangi distraksi digital saat bekerja atau belajar.",

                    "☀️ Sisihkan waktu istirahat singkat agar otak tidak kelelahan."
                ])

            # =================================================
            # KONDISI MASIH STABIL
            # =================================================

            if len(recommendations) == 0:

                st.success("""
                🌿 Kondisi digital wellness Anda masih cukup baik.

                Pertahankan keseimbangan antara aktivitas digital,
                aktivitas fisik, dan waktu istirahat agar kesehatan
                mental tetap terjaga.
                """)

            # =================================================
            # TAMPILKAN REKOMENDASI
            # =================================================

            else:

                unique_recommendations = list(
                    dict.fromkeys(recommendations)
                )

                for rec in unique_recommendations:

                    st.success(rec)

            # =================================================
            # BRAIN RECOVERY SYSTEM
            # =================================================

            st.markdown("---")

            st.header("🌱 Kondisi Recovery Anda")

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
# TAB 4 - RECOVERY & WELLNESS AI
# =========================================================

with tab4:

    st.header("🌱 Recovery Center")

    st.markdown("""
    Recovery Center membantu Anda memahami kondisi keseimbangan digital,
    mengurangi overstimulasi akibat penggunaan gadget berlebihan,
    serta memberikan rekomendasi recovery harian untuk menjaga fokus, kesehatan mental, dan kualitas istirahat.
    """)

    st.markdown("---")

    # =====================================================
    # AI WELLNESS SUMMARY
    # =====================================================

    st.header("🧠 AI Wellness Summary")

    data = st.session_state.wellness_result

    if data is None:

        st.warning("""
        Silakan lakukan Wellness Check terlebih dahulu.
        """)

    else:

        fatigue = data['fatigue_percent']
        screen = data['screen_time']
        sleep = data['sleep_hours']
        stress = data['stress_level']

        if fatigue <= 35:

            summary = f"""
            Kondisi mental Anda masih cukup stabil.

            Penggunaan gadget sekitar {screen} jam/hari
            masih dalam batas aman
            dan belum memberikan dampak signifikan
            terhadap fokus maupun keseimbangan mental.
            """

        elif fatigue <= 65:

            summary = f"""
            Aktivitas digital harian mulai mempengaruhi
            fokus dan energi mental Anda.

            Penggunaan gadget {screen} jam/hari,
            tidur {sleep} jam,
            serta level stres sedang
            menunjukkan tanda-tanda kelelahan mental ringan pada diri Anda.
            """

        else:

            summary = f"""
            Sistem mendeteksi risiko kelelahan mental tinggi.
            Penggunaan gadget yang tinggi,
            kurang tidur, dan level stres tinggi
            mulai mempengaruhi keseimbangan mental Anda.
            Disarankan untuk melakukan digital recovery
            dan mengurangi overstimulasi digital sementara waktu.
            """

        st.info(summary)

    # =====================================================
    # DOPAMINE OVERLOAD METER
    # =====================================================

    st.markdown("---")

    st.header("⚡ Dopamine Overload Meter")

    data = st.session_state.wellness_result

    if data is None:

        st.warning("""
        Silakan lakukan Wellness Check terlebih dahulu.
        """)

    else:

        fatigue_percent = data['fatigue_percent']

        prediction = data['prediction']

        screen_time = data['screen_time']

        social_media = data['social_media']

        # =====================================================
        # SINKRON DENGAN WELLNESS CHECK
        # =====================================================

        if prediction == "Refreshed":

            dopamine_percent = max(
                15,
                fatigue_percent - 10
            )

            dopamine_status = "🟢 Rendah"

            dopamine_desc = f"""
            Aktivitas digital Anda masih cukup sehat dan belum menunjukkan overstimulasi berlebihan. Dengan kesimpulan:
            - Penggunaan gadget sekitar {screen_time} jam/hari
            
            * masih dalam batas yang cukup aman untuk fokus dan keseimbangan mental.
            """

        elif prediction == "Strained":

            dopamine_percent = fatigue_percent

            dopamine_status = "🟡 Sedang"

            dopamine_desc = f"""
            Sistem mendeteksi tanda-tanda overstimulasi digital ringan. Dengan kesimpulan:
            - Penggunaan gadget {screen_time} jam/hari
            - dan penggunaan media sosial {social_media} jam/hari
            
            * mulai mempengaruhi fokus, konsentrasi, dan energi mental Anda.
            """

        else:

            dopamine_percent = min(
                fatigue_percent + 5,
                95
            )

            dopamine_status = "🔴 Tinggi"

            dopamine_desc = f"""
            Sistem mendeteksi overstimulasi digital tinggi yang berkaitan dengan risiko brainrot dan kelelahan mental berat.
            Dengan kesimpulan:
            - Aktivitas digital yang berlebihan,
            - scrolling media sosial berlebihan,
            - serta kurangnya durasi waktu tidur.
            
            * mulai mempengaruhi fokus dan keseimbangan mental Anda secara signifikan.
            """

        # =====================================================
        # OUTPUT
        # =====================================================

        st.metric(
            "Tingkat Dopamine Overload",
            f"{dopamine_percent}%"
        )

        st.progress(dopamine_percent)

        st.warning(f"""
        {dopamine_status}

        {dopamine_desc}
        """)

        # =====================================================
        # INSIGHT TAMBAHAN
        # =====================================================

        st.info(f"""
        📊 Tingkat Dopamine Overload disesuaikan
        dengan hasil Wellness Check dan Machine Learning Prediction.

        Semakin tinggi nilainya,
        maka semakin tinggi risiko overstimulasi digital
        akibat penggunaan gadget berlebihan,
        media sosial,
        dan konsumsi konten instan berlebihan.
        """)

        # =====================================================
        # VISUALISASI AI
        # =====================================================

        gauge_dopamine = go.Figure(go.Indicator(

            mode="gauge+number",

            value=dopamine_percent,

            title={
                'text':
                "Overstimulasi Digital"
            },

            gauge={

                'axis': {
                    'range': [0, 100]
                },

                'bar': {
                    'color': "#6366F1"
                },

                'steps': [

                    {
                        'range': [0, 35],
                        'color': "#10B981"
                    },

                    {
                        'range': [35, 70],
                        'color': "#F59E0B"
                    },

                    {
                        'range': [70, 100],
                        'color': "#EF4444"
                    }
                ]
            }
        ))

        gauge_dopamine.update_layout(

            paper_bgcolor="#0E1117",

            font={
                'color': "white"
            },

            height=320
        )

        st.plotly_chart(
            gauge_dopamine,
            use_container_width=True
        )

        # =====================================================
        # DAILY RECOVERY CHALLENGE
        # =====================================================

        st.markdown("---")

        st.header("🎯 Daily Recovery Challenge")

        data = st.session_state.wellness_result

        if data is None:

            st.warning("""
            Silakan lakukan Wellness Check terlebih dahulu.
            """)

        else:

            fatigue_percent = data['fatigue_percent']

            prediction = data['prediction']

            screen_time = data['screen_time']

            sleep_hours = data['sleep_hours']

            stress_level = data['stress_level']

            social_media = data['social_media']

            exercise = data['exercise']

        # =====================================================
        # AI RECOVERY INTRO
        # =====================================================

        if prediction == "Refreshed":

            st.success("""
            🌿 Kondisi mental Anda masih cukup stabil.
            Berikut beberapa challenge ringan yang dapat Anda lakukan
            untuk menjaga keseimbangan digital:
            """)

        elif prediction == "Strained":

            st.warning("""
            ⚠️ Sistem mendeteksi bahwa Anda mengalami gejala awal kelelahan mental ringan.
            Berikut Challenge yang dapat membantu Anda untuk mengurangi overstimulasi digital berlebihan:
            """)

        else:

            st.error("""
            🚨 Sistem mendeteksi risiko kelelahan mental tinggi.
            Berikut Recovery challenge yang disarankan
            untuk Anda dalam membantu pemulihan mental dan mengurangi brainrot:
            """)

        # =====================================================
        # LIST CHALLENGE
        # =====================================================

        challenges = []

        # =====================================================
        # SCREEN TIME TINGGI
        # =====================================================

        if screen_time > 8:

            challenges.append(
                "📵 Kurangi screen time 1–2 jam lebih sedikit dari biasanya hari ini."
            )

        elif screen_time > 5:

            challenges.append(
                "⏳ Coba lakukan 30 menit tanpa gadget sebelum tidur."
            )

        # =====================================================
        # SOCIAL MEDIA TINGGI
        # =====================================================

        if social_media > 6:

            challenges.append(
                "📱 Hindari scrolling media sosial selama 1 jam penuh."
            )

        elif social_media > 3:

            challenges.append(
                "🎯 Batasi konsumsi short-form content hari ini."
            )

        # =====================================================
        # TIDUR RENDAH
        # =====================================================

        if sleep_hours < 4:

            challenges.append(
                "😴 Tidur lebih awal dan targetkan minimal 7 jam tidur malam ini."
            )

        elif sleep_hours < 6:

            challenges.append(
                "🌙 Hindari gadget 30 menit sebelum tidur."
            )

        # =====================================================
        # STRES TINGGI
        # =====================================================

        if stress_level >= 8:

            challenges.append(
                "🧘 Lakukan meditasi atau relaksasi selama 15–20 menit."
            )

        elif stress_level >= 6:

            challenges.append(
                "🎵 Dengarkan musik relaksasi tanpa membuka media sosial."
            )

        # =====================================================
        # AKTIVITAS FISIK RENDAH
        # =====================================================

        if exercise < 15:

            challenges.append(
                "🚶 Jalan santai atau olahraga ringan minimal 20 menit."
            )

        elif exercise < 30:

            challenges.append(
                "🤸 Lakukan stretching ringan untuk membantu recovery tubuh."
            )

        # =====================================================
        # FATIGUE TINGGI
        # =====================================================

        if fatigue_percent >= 75:

            challenges.append(
                "📚 Lakukan aktivitas non-digital seperti membaca buku fisik."
            )

            challenges.append(
                "🌳 Luangkan waktu di area terbuka tanpa gadget."
            )

        # =====================================================
        # KONDISI MASIH STABIL
        # =====================================================

        if len(challenges) == 0:

            st.success("""
            🌿 Kondisi digital wellness Anda masih cukup baik.

            Pertahankan pola hidup sehat,
            kualitas tidur,
            dan keseimbangan aktivitas digital.
            """)

        # =====================================================
        # TAMPILKAN CHALLENGE
        # =====================================================

        else:

            unique_challenges = list(
                dict.fromkeys(challenges)
            )

            for challenge in unique_challenges:

                st.success(challenge)

        # =====================================================
        # RECOVERY SCORE
        # =====================================================

        recovery_score = max(
            100 - fatigue_percent,
            5
        )

        st.markdown("---")

        st.metric(
            "Recovery Readiness Score",
            f"{recovery_score}%"
        )

        if recovery_score >= 70:

            st.success("""
            🌿 Kondisi recovery Anda cukup baik.
            """)

        elif recovery_score >= 40:

            st.warning("""
            ⚠️ Recovery mental Anda perlu ditingkatkan.
            """)

        else:

            st.error("""
            🚨 Kondisi mental Anda membutuhkan recovery lebih serius.
            """)

# =========================================================
# TAB 5 - PROGRESS TRACKER
# =========================================================

with tab5:

    st.header("📈 Progress Tracker")
    
    history = st.session_state.progress_history

    if len(history) == 0:

        st.warning("""
        Belum ada data progress.

        Silakan lakukan Wellness Check terlebih dahulu.
        """)

    else:

        history_df = pd.DataFrame(history)

        history_df['Check'] = range(
            1,
            len(history_df) + 1
        )

        fig_progress = px.line(

            history_df,

            x='Check',
            y='Fatigue Risk',

            markers=True,

            title='Perkembangan Risiko Kelelahan Mental'
        )

        fig_progress.update_layout(

            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font_color="white",

            height=450
        )

        st.plotly_chart(
            fig_progress,
            use_container_width=True
        )

        latest = history_df.iloc[-1]['Fatigue Risk']

        first = history_df.iloc[0]['Fatigue Risk']

        # =====================================================
        # INTERPRETASI PERKEMBANGAN
        # =====================================================

        if latest < first:

            st.success("""
            📉 Kondisi mental Anda menunjukkan perkembangan positif.

            Risiko kelelahan mental mulai menurun
            dibandingkan pemeriksaan sebelumnya.
            """)

        elif latest > first:

            st.error("""
            📈 Risiko kelelahan mental Anda meningkat.

            Aktivitas digital dan stres harian
            mulai memberikan dampak yang lebih besar.
            """)

        else:

            st.info("""
            📊 Kondisi mental Anda relatif stabil.
            """)

        # =====================================================
        # TABEL HISTORY
        # =====================================================

        st.markdown("---")

        st.subheader("🗂 Riwayat Pemeriksaan")

        st.dataframe(
            history_df,
            use_container_width=True
        )
    
    # =====================================================
    # LOAD HISTORY DATAFRAME
    # =====================================================

    history_df = pd.DataFrame(
        st.session_state.progress_history
    )
    # =====================================================
    # RECOVERY TIMELINE
    # =====================================================

    st.markdown("---")

    st.subheader("📅 Recovery Timeline")

    if len(history_df) > 0:

        timeline_fig = px.line(

            history_df,

            x='Date',

            y='Fatigue Risk',

            markers=True,

            title="Perkembangan Risiko Mental Harian"
        )

        timeline_fig.update_layout(

            paper_bgcolor="#0E1117",

            plot_bgcolor="#0E1117",

            font=dict(color="white")
        )

        st.plotly_chart(
            timeline_fig,
            use_container_width=True
        )

    else:

        st.info("""
        Belum ada histori pemeriksaan.
        """)
    
    # =====================================================
    # RECOVERY STREAK
    # =====================================================

    st.markdown("---")

    st.subheader("🔥 Recovery Streak")

    if len(history_df) >= 2:

        streak = 0

        fatigue_values = history_df[
            'Fatigue Risk'
        ].tolist()

        for i in range(1, len(fatigue_values)):

            if fatigue_values[i] < fatigue_values[i - 1]:

                streak += 1

        st.metric(
            "Recovery Improvement Streak",
            f"{streak} sesi"
        )

        if streak >= 3:

            st.success("""
            🌿 Kondisi mental Anda menunjukkan perkembangan positif.
            """)

        else:

            st.warning("""
            Recovery Anda masih belum stabil.
            """)

    else:

        st.info("""
        Minimal diperlukan 2 histori pemeriksaan.
        """)
        
    # =====================================================
    # TREND MENTAL
    # =====================================================

    st.markdown("---")

    st.subheader("📈 Trend Mental")

    if len(history_df) >= 2:

        latest = history_df[
            'Fatigue Risk'
        ].iloc[-1]

        previous = history_df[
            'Fatigue Risk'
        ].iloc[-2]

        if latest < previous:

            st.success("""
            📉 Tingkat kelelahan mental Anda mulai berkurang.
            Kondisi digital wellness menunjukkan perkembangan positif.
            """)

        elif latest > previous:

            st.error("""
            📈 Risiko mental Anda meningkat.
            Disarankan meningkatkan recovery.
            """)

        else:

            st.info("""
            ➖ Kondisi mental Anda relatif stabil.
            """)

    else:

        st.info("""
        Belum cukup data untuk melihat trend.
        """)
    
    # =====================================================
    # MOOD JOURNAL
    # =====================================================

    st.markdown("---")

    st.subheader("😊 Mood Journal")

    mood_note = st.text_area(
        "Bagaimana kondisi Anda hari ini?"
    )

    if st.button("💾 Simpan Catatan"):

        with open(
            "mood_journal.txt",
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                f"{datetime.now()} - {mood_note}\n"
            )

        st.success("""
        Catatan berhasil disimpan.
        """)


# =========================================================
# TAB 6
# =========================================================

with tab6:

    st.header("📘 Panduan & Insight")

    st.markdown("""
    Panduan & Insight membantu Anda memahami dampak aktivitas digital,
    menjaga keseimbangan mental, serta memberikan edukasi sederhana
    tentang recovery dan digital wellness sehari-hari.
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
    berdasarkan kebiasaan baru. Artinya: brainrot atau kelelahan akibat overstimulasi digital bukan kondisi permanen.

    Berikut beberapa kebiasaan sehat yang dapat membantu memulihkan fokus, konsentrasi, dan kesehatan mental secara bertahap. seperti:

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