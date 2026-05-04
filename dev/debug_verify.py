"""
Final verification: confirm the root cause and check whether the model is only using the first 6 features.
The model may be sensitive to Location_Target.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from services.model_loader import models
models.load()

model_features = list(models.model_low.feature_names_)
print(f"Model hanya mengenal {len(model_features)} fitur: {model_features}")

# Create a DataFrame using only the 6 features recognized by the model
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
    # Create a DataFrame with exactly the 6 features
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
