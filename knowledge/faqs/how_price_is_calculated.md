# How Property Prices Are Estimated

## Model Methodology
Our system uses **CatBoost** gradient boosting machines trained on 40,000+ Depok property transactions.

### Training Features
1. **Core Specs**: Bedrooms, bathrooms, garage capacity, land/building area
2. **Location Encoding**: Kecamatan one-hot + cluster embeddings
3. **Size Normalization**: LT/LB ratios and log transforms

### Prediction Workflow
```
User Input → Feature Engineering → CatBoost Ensemble → Price Estimate
                         ↓
                 MAPE Confidence Score + Segment Label
```

### Model Variants
- **Low/Medium/High** price specialists (3 models)
- **Ensemble** combines all for final prediction
- Continuously retrained via user feedback loop

**Accuracy**: ~12-15% MAPE on held-out test set
