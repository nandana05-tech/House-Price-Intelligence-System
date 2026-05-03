"""Diagnosa lanjut: cek apakah model_low benar-benar sensitif terhadap fitur."""
import sys
sys.path.insert(0, '.')
import numpy as np

from services.model_loader import models
models.load()

from services.feature_engineer import engineer_regression_features

# Test 1: variasi lokasi dengan properti sama
print("\n=== Test 1: Variasi Lokasi (3KT 2KM 1G LT=120 LB=90) ===")
for loc in ['Cinere', 'Sawangan', 'Margonda', 'Tapos', 'Citayam']:
    X = engineer_regression_features(3, 2, 1, 120.0, 90.0, loc)
    log_pred = float(models.model_low.predict(X)[0])
    pred = float(np.expm1(log_pred))
    lokasi_target = X['Lokasi_Target'].iloc[0]
    print(f"  {loc:<18s} LT_enc={lokasi_target:.2f}  log_pred={log_pred:.6f}  harga={pred:,.0f}")

# Test 2: variasi luas tanah drastis
print("\n=== Test 2: Variasi Luas Tanah (Cinere, 3KT 2KM 1G) ===")
for lt in [50, 100, 200, 500, 1000]:
    X = engineer_regression_features(3, 2, 1, float(lt), 90.0, 'Cinere')
    log_pred = float(models.model_low.predict(X)[0])
    pred = float(np.expm1(log_pred))
    print(f"  LT={lt:<6}  log_pred={log_pred:.6f}  harga={pred:,.0f}")

# Test 3: cek apakah fitur X memang berbeda
print("\n=== Test 3: Cek DataFrame fitur untuk Cinere vs Tapos ===")
X1 = engineer_regression_features(3, 2, 1, 120.0, 90.0, 'Cinere')
X2 = engineer_regression_features(3, 2, 1, 120.0, 90.0, 'Tapos')
diff = X1.iloc[0] - X2.iloc[0]
print("Selisih Cinere - Tapos:")
for col in diff.index:
    if abs(diff[col]) > 0.0001:
        print(f"  {col:<25s}: {diff[col]:+.6f}")

# Test 4: cek jumlah tree predictions
print("\n=== Test 4: Cek tree count model_low ===")
print(f"  tree_count = {models.model_low.tree_count_}")
print(f"  feature_names = {models.model_low.feature_names_[:5]}...")
