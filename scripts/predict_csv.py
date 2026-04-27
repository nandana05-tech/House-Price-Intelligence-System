#!/usr/bin/env python3
"""
Batch prediction on CSV data using HPI models.
Usage: python predict_csv.py your_data.csv
Requires .env w/ DATABASE_URL etc if needed.
"""

import argparse
import pandas as pd
import json
from dotenv import load_dotenv
from services.model_loader import load_all_models
from services.feature_engineer import engineer_features
from services.predictor import predict_price, classify_segment, cluster_property
from mlflow_utils.tracker import PredictionTracker

load_dotenv()

def main(csv_file):
    # Load models
    models = load_all_models()
    
    # Read CSV (assume columns: kamar_tidur, kamar_mandi, garasi, luas_tanah, luas_bangunan, lokasi)
    df = pd.read_csv(csv_file)
    
    results = []
    tracker = PredictionTracker()
    
    for idx, row in df.iterrows():
        features = engineer_features(row)
        
        # Predict
        price = predict_price(**features)
        segment = classify_segment(**features, harga=price['harga_estimasi'])
        cluster = cluster_property(**features, harga=price['harga_estimasi'])
        
        result = {
            'original_index': idx,
            **row.to_dict(),
            'predicted_price': price['harga_estimasi'],
            'mape': price['mape'],
            'model_used': price['model_digunakan'],
            'segment': segment['kelas_label'],
            'cluster': cluster['cluster_label']
        }
        
        # Track to MLflow
        tracker.log_prediction(result)
        
        results.append(result)
        print(json.dumps(result, indent=2))
    
    # Save results
    output_file = csv_file.replace('.csv', '_predictions.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predict on CSV")
    parser.add_argument("csv_file", help="Input CSV file")
    args = parser.parse_args()
    main(args.csv_file)
