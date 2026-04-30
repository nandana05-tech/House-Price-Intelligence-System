"""
Setup script — rebuilds target_encoder.pkl using category_encoders.TargetEncoder
to match the training pipeline behavior.

Run once:
    python scripts/setup_encoder.py
"""
import os
import pickle
from pathlib import Path

import category_encoders as ce
import numpy as np
import pandas as pd
import psycopg2

BASE_DIR = Path(__file__).parent.parent
OUT_PATH = BASE_DIR / "models" / "target_encoder.pkl"

print("Loading data from CleanPropertyRegression...")
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
df = pd.read_sql('SELECT * FROM "CleanPropertyRegression"', conn)
df = df.rename(columns={"harga": "Harga", "lokasi": "Lokasi"})
conn.close()
print("  Rows:", len(df))

# Train target encoder on log target to match regression training
y = np.log1p(df["Harga"])
te = ce.TargetEncoder(cols=["Lokasi"], smoothing=10)
te.fit(df[["Lokasi"]], y)

with open(OUT_PATH, "wb") as f:
    pickle.dump(te, f)

print(f"Saved TargetEncoder object to: {OUT_PATH}")

# Verification
test_lokasi = pd.DataFrame({"Lokasi": ["Cinere", "Sawangan", "Beji", "UNKNOWN"]})
encoded = te.transform(test_lokasi)["Lokasi"].tolist()
for loc, val in zip(test_lokasi["Lokasi"], encoded, strict=False):
    print(f"  {loc:25s} -> {float(val):.6f}")

print("Done. You can now run: python server.py")
