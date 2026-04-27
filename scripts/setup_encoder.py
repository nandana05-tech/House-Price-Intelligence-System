"""
Setup script — rebuilds the target_encoder.pkl from data_with_clusters.csv.
The Lokasi_enc column is log(mean_harga) per lokasi, precomputed during training.
This script builds a lookup dict and saves it as target_encoder.pkl.

Run once:
    python setup_encoder.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent

print("Loading data...")
df = pd.read_csv(BASE_DIR / "data_with_clusters.csv")
print("  Rows:", len(df))

# Build per-lokasi mean of Lokasi_enc (should be stable per group)
lokasi_enc_map = df.groupby("Lokasi")["Lokasi_enc"].mean().to_dict()

# Global fallback: mean of all Lokasi_enc values
fallback = float(df["Lokasi_enc"].mean())

encoder_artifact = {
    "type": "lokasi_enc_lookup",
    "map": lokasi_enc_map,
    "fallback": fallback,
}

out_path = BASE_DIR / "target_encoder.pkl"
with open(out_path, "wb") as f:
    pickle.dump(encoder_artifact, f)
print("Saved rebuilt target_encoder.pkl")

# Verification
for loc in ["Cinere", "Sawangan", "Beji", "UNKNOWN"]:
    val = lokasi_enc_map.get(loc, fallback)
    print(f"  {loc:25s} -> {val:.4f}")

print("Done. You can now run: python server.py")
