"""Script diagnosa: cek apakah lokasi berbeda menghasilkan harga berbeda."""
import sys
sys.path.insert(0, '.')

from services.model_loader import models
models.load()

from services.predictor import predict_price

locs = ['Cinere', 'Sawangan', 'Margonda', 'Tapos', 'Citayam', 'Pangkalan Jati', 'Beji', 'Depok I']

print("\n=== Test Prediksi (3KT 2KM 1G, LT=120 LB=90) ===")
for loc in locs:
    r = predict_price(3, 2, 1, 120.0, 90.0, loc)
    harga = r["harga_estimasi"]
    model = r["model_digunakan"]
    print(f"  {loc:<22s} -> Rp {harga:>15,.0f}  [{model}]")
