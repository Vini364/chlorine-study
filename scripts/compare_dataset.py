import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import process_dataset_records

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def compare_datasets(dataset1_path, dataset2_path):
    """Compare two datasets and print differences in image counts and concentrations"""
    with open (dataset1_path, "r") as f:
        dataset1 = json.load(f)
    with open (dataset2_path, "r") as f:
        dataset2 = json.load(f)
    df_a = process_dataset_records(dataset1)
    df_b = process_dataset_records(dataset2)

        
    high_a = df_a[df_a["concentration"] >= 1.5]
    high_b = df_b[df_b["concentration"] >= 1.5]

    # for feature in ["mean_A", "std_A", "mean_S", "mean_R"]:
    #     print(f"\n{feature}")
    #     print(
    #         f"Aiman/Sanskriti: {high_a[feature].mean():.2f}"
    #     )
    #     print(
    #         f"Vinesh/Juliania: {high_b[feature].mean():.2f}"
    #     )
    #     print(
    #         f"Difference: {high_a[feature].mean() - high_b[feature].mean():.2f}"
    #     )
  

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        df_a["mean_A"],
        df_a["mean_S"],
        c=df_a["concentration"],
        cmap="Blues",
        alpha=0.7,
        label="Aiman/Sanskriti"
    )

    plt.scatter(
        df_b["mean_A"],
        df_b["mean_S"],
        c=df_b["concentration"],
        cmap="Oranges",
        alpha=0.7,
        label="Vinesh/Juliania"
    )

    plt.xlabel("mean_A")
    plt.ylabel("mean_S")
    plt.title("Feature Space: mean_A vs mean_S")

    plt.legend()
    plt.colorbar(scatter, label="Concentration")

    plt.show()
        
    # for conc in sorted(df_a["concentration"].unique()):
    #     a_mean = df_a[df_a["concentration"] == conc]["mean_A"].mean()
    #     b_mean = df_b[df_b["concentration"] == conc]["mean_A"].mean()

    #     print(
    #         f"{conc}: "
    #         f"Aiman/Sanskriti={a_mean:.2f}, "
    #         f"Vinesh/Juliania={b_mean:.2f}, "
    #         f"Diff={abs(a_mean-b_mean):.2f}"
    #     )
if __name__ == "__main__":
    if len(sys.argv) > 2:
        dataset1_path = DATA_DIR / sys.argv[1]
        dataset2_path = DATA_DIR / sys.argv[2]
        compare_datasets(str(dataset1_path), str(dataset2_path))
    else:
        print("Usage: python scripts/compare_dataset.py <dataset1.json> <dataset2.json>")

    