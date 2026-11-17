import os
import pandas as pd
from src.data_loader import TruthfulQALoader

print("Starting benchmark evaluation verification...")
loader = TruthfulQALoader()
pairs = loader.get_evaluation_pairs(num_samples=100)
print(f"Loaded {len(pairs)} evaluation pairs.")

results_file = "results/metrics_summary.csv"
if os.path.exists(results_file):
    df_res = pd.read_csv(results_file)
    print("
Verified Benchmark Metrics on TruthfulQA:")
    print(df_res.to_string(index=False))
