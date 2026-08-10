# from scipy.stats import ks_2samp
# import pandas as pd
# import os

# # PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# data_3 = pd.read_csv(os.path.join("new_dataset_3.csv"))
# data_4 = pd.read_csv(os.path.join("new_dataset_4.csv"))

# feature_cols = [col for col in data_3.columns if col not in ['concentration', 'is_augmented', 'source_id', 'filename']]

# results = []
# for col in feature_cols:
#     train_vals = data_3[col].dropna()
#     test_vals = data_4[col].dropna()
#     stat, pval = ks_2samp(train_vals, test_vals)
#     results.append({
#         'feature': col,
#         'train_mean': train_vals.mean(), 'test_mean': test_vals.mean(),
#         'train_std': train_vals.std(), 'test_std': test_vals.std(),
#         'ks_stat': stat, 'ks_pval': pval
#     })

# shift_df = pd.DataFrame(results).sort_values('ks_stat', ascending=False)
# print(shift_df.head(10))






from src.features import process_dataset_records
from src.pipeline import load_dataset_records

# dataset_3 = load_dataset_records("/Users/vineshuniyal/Downloads/chlorine-study/data/processed/new_dataset_3.json")
# dataset_4 = load_dataset_records("/Users/vineshuniyal/Downloads/chlorine-study/data/processed/new_dataset_4.json")
dataset_5 = load_dataset_records("/Users/vineshuniyal/Downloads/chlorine-study/data/processed/new_dataset_5.json")
process_dataset_records(
    data=dataset_5,
    output_csv = "new_dataset_5.csv", namespace = "dataset_5.json"
)


