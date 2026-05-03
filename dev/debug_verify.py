"""
Verifikasi final: konfirmasi root cause dan cek apakah hanya memakai 6 fitur pertama
model bisa sensitif terhadap Lokasi_Target.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from services.model_loader import models
models.load()

model_features = list(models.model_low.feature_names_)
print(f"Model hanya mengenal {len(model_features)} fitur: {model_features}")

# Buat DataFrame hanya dengan 6 fitur yang dikenal model
from services.feature_engineer import engineer_regression_features

print("\n=== Test lokasi dengan HANYA 6 fitur yang model kenal ===")
from category_encoders import TargetEncoder

for loc in ['Cinere', 'Sawangan', 'Margonda', 'Tapos', 'Citayam', 'Pangkalan Jati']:
    # Encode lokasi
    lok_enc = float(
        models.target_encoder.transform(
            pd.DataFrame({"Lokasi": [loc]})
        )["Lokasi"].iloc[0]
    )
    # Buat DataFrame dengan persis 6 fitur
    X6 = pd.DataFrame([{
        "Kamar Tidur":   3,
        "Kamar Mandi":   2,
        "Garasi":        1,
        "Luas Tanah":    120.0,
        "Luas Bangunan": 90.0,
        "Lokasi_Target": lok_enc,
    }])
    log_pred = float(models.model_low.predict(X6)[0])
    pred = float(np.expm1(log_pred))
    print(f"  {loc:<20s} Lokasi_Target={lok_enc:.4f}  harga=Rp {pred:>15,.0f}")
