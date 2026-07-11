import pandas as pd
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import decode_image, yolo_model
from src.pipeline import load_dataset_records, choose_input_dataset
import cv2
import numpy as np

def test_resolution_impact(sample_records, test_dims=[None, 1500, 1200, 800, 600]):
    results = []
    for sample in sample_records:
        rgb_full = decode_image(sample["image"])
        for dim in test_dims:
            if dim is None:
                rgb_test = rgb_full
                label = "full_res"
            else:
                h, w = rgb_full.shape[:2]
                scale = dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                rgb_test = cv2.resize(rgb_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
                label = f"{dim}px"

            yolo_result = yolo_model(rgb_test, verbose=False)[0]
            mask_found = yolo_result.masks is not None
            mask_area_frac = None
            if mask_found:
                mask = yolo_result.masks.data[0].cpu().numpy()
                mask_area_frac = float(np.mean(mask > 0.5))

            results.append({
                "source_id": sample["source_id"],
                "concentration": sample["concentration"],
                "resolution": label,
                "yolo_detected": mask_found,
                "mask_area_frac": mask_area_frac,
            })
    return pd.DataFrame(results)


if __name__ == "__main__":
    input_file = choose_input_dataset()
    records = load_dataset_records(input_file)

    # pick a handful spanning low/mid/high conc
    records_sorted = sorted(records, key=lambda r: r["concentration"])
    sample_records = (
        records_sorted[:3] + records_sorted[len(records_sorted)//2-1:len(records_sorted)//2+2] + records_sorted[-3:]
    )

    df = test_resolution_impact(sample_records)
    print(df.to_string())
    
    # Save to project root
    output_path = Path(__file__).parent.parent / "resolution_test_results.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")