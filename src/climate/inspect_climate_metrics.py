from pathlib import Path
import pandas as pd

folder = Path("results/climate")

files = [
    "fft_tdmd_frame_metrics.csv",
    "haar_tdmd_frame_metrics.csv",
    "cosine_tdmd_frame_metrics.csv",
    "unfolding_dmd_frame_metrics.csv",
    "tt_dmd_frame_metrics.csv",
]

for filename in files:
    path = folder / filename
    print(f"\n--- {filename} ---")

    if not path.exists():
        print("FILE MISSING")
        continue

    df = pd.read_csv(path)
    print("Columns:", list(df.columns))
    print("Number of rows:", len(df))
    print(df.head(3).to_string(index=False))
    print("Last row:")
    print(df.tail(1).to_string(index=False))
