"""Cek feature importance model_low untuk memastikan root cause."""
import sys
sys.path.insert(0, '.')

from services.model_loader import models
models.load()

import pandas as pd

# Feature importance model_low
fi = models.model_low.get_feature_importance()
fn = models.model_low.feature_names_
fi_df = pd.DataFrame({'feature': fn, 'importance': fi}).sort_values('importance', ascending=False)

print("\n=== Feature Importance model_low ===")
print(fi_df.to_string(index=False))

print("\n=== Lokasi-related features ===")
lokasi_feats = fi_df[fi_df['feature'].str.contains('lokasi|Lokasi', case=False)]
print(lokasi_feats.to_string(index=False))

print(f"\nTotal lokasi importance: {lokasi_feats['importance'].sum():.4f}%")
print(f"Total all importance:    {fi_df['importance'].sum():.4f}%")
