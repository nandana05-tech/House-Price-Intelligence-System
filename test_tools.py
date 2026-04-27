from services.model_loader import models
models.load()
from services.predictor import predict_price, classify_segment, cluster_property

# Test 1: Predict price
r = predict_price(3, 2, 1, 90, 70, "Cinere")
print("[predict_price]", r["harga_estimasi_format"], "| model:", r["model_digunakan"], "| MAPE:", r["mape_persen"])

# Test 2: Classify segment (auto-estimate price)
c = classify_segment(3, 2, 1, 90, 70, "Cinere")
print("[classify_segment]", c["kelas_label"], "| harga_sumber:", c["harga_sumber"])

# Test 3: Cluster property (auto-estimate price)
cl = cluster_property(90, 70, 3, 2, "Cinere")
print("[cluster_property]", cl["cluster_label"], "| harga_sumber:", cl["harga_sumber"])

print("SEMUA OK!")
