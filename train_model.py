import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv(
    "data/cleaned_cognitive_fatigue_dataset.csv"
)
print(df.columns)
# =====================================================
# FEATURE ENGINEERING
# =====================================================

df['fatigue_score'] = (
    (df['screen_time'] * 0.35) +
    ((10 - df['sleep_hours']) * 0.30) +
    (df['stress_level'] * 0.35)
)

# =====================================================
# LABEL
# =====================================================

df['fatigue_label'] = np.where(
    df['fatigue_score'] < 4,
    'Refreshed',

    np.where(
        df['fatigue_score'] < 7,
        'Strained',
        'Near-Burnout'
    )
)

# =====================================================
# FEATURE & TARGET
# =====================================================

X = df[[
    'screen_time',
    'sleep_hours',
    'stress_level',
    'digital_balance',
    'physical_activity',
    'work_hours'
]]

y = df['fatigue_label']

# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# =====================================================
# TRAINING
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# PREDIKSI
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# AKURASI
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Akurasi Model: {accuracy:.2f}")

# =====================================================
# SIMPAN MODEL
# =====================================================

joblib.dump(
    model,
    "model/fatigue_model.pkl"
)

print("Model berhasil disimpan!")