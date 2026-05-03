"""Diagnosa kritis: cek apakah urutan/nama kolom DataFrame cocok dengan model training."""
import sys
sys.path.insert(0, '.')
import numpy as np

from services.model_loader import models
models.load()

from services.feature_engineer import engineer_regression_features

# Cek feature names di model vs DataFrame yang dikirim
model_features = list(models.model_low.feature_names_)
metadata_features = list(models.meta_regresi["fitur"])

print("=== Feature names model_low (dari model) ===")
for i, f in enumerate(model_features):
    print(f"  [{i:2d}] {f}")

print("\n=== Feature names dari metadata_regresi.json ===")
for i, f in enumerate(metadata_features):
    print(f"  [{i:2d}] {f}")

print("\n=== Apakah urutan sama? ===")
if model_features == metadata_features:
    print("  SAMA persis")
else:
    print("  BERBEDA!")
    for i, (mf, df) in enumerate(zip(model_features, metadata_features)):
        mark = "OK" if mf == df else "<<< BEDA"
        print(f"  [{i:2d}] model={mf!r:30s} meta={df!r}  {mark}")

# Cek DataFrame yang dihasilkan engineer
X = engineer_regression_features(3, 2, 1, 120.0, 90.0, 'Cinere')
print("\n=== Kolom DataFrame hasil engineer (actual) ===")
for i, c in enumerate(X.columns):
    print(f"  [{i:2d}] {c}")

print("\n=== Apakah urutan DataFrame == model? ===")
df_cols = list(X.columns)
if df_cols == model_features:
    print("  SAMA persis")
else:
    print("  BERBEDA!")
    for i, (mc, dc) in enumerate(zip(model_features, df_cols)):
        mark = "OK" if mc == dc else "<<< BEDA"
        print(f"  [{i:2d}] model={mc!r:30s} df={dc!r}  {mark}")

# Bandingkan prediksi dengan urutan kolom asli vs reorder
print("\n=== Prediksi dengan X langsung ===")
pred1 = float(np.expm1(float(models.model_low.predict(X)[0])))
print(f"  Harga (X langsung):  Rp {pred1:,.0f}")

# Coba reorder sesuai model_features jika berbeda
if df_cols != model_features:
    X_reordered = X[model_features]
    pred2 = float(np.expm1(float(models.model_low.predict(X_reordered)[0])))
    print(f"  Harga (X reordered): Rp {pred2:,.0f}")

# Cek nilai aktual di DataFrame
print("\n=== Nilai baris X untuk Cinere ===")
print(X.T.to_string())
