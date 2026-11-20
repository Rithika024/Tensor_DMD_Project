from pathlib import Path
import pandas as pd

folder = Path("results/climate")

files = {
    "FFT-TDMD": "fft_tdmd_frame_metrics.csv",
    "Haar-TDMD": "haar_tdmd_frame_metrics.csv",
    "Cosine-TDMD": "cosine_tdmd_frame_metrics.csv",
    "Unfolding DMD": "unfolding_dmd_frame_metrics.csv",
    "TT-DMD": "tt_dmd_frame_metrics.csv",
}

required_metrics = [
    "relative_error_percent",
    "mae",
    "rmse",
]

normalized = {}
all_frames_match = True

for method, filename in files.items():
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Normalize the target-frame column.
    frame_candidates = [
        "target_original_frame",
        "original_frame",
        "frame",
    ]
    frame_col = next(
        (c for c in frame_candidates if c in df.columns),
        None,
    )

    if frame_col is None:
        raise ValueError(
            f"{filename}: cannot identify target frame column. "
            f"Columns: {list(df.columns)}"
        )

    df = df.rename(columns={frame_col: "target_frame"})

    missing = [c for c in required_metrics if c not in df.columns]
    if missing:
        raise ValueError(
            f"{filename}: missing metrics {missing}. "
            f"Columns: {list(df.columns)}"
        )

    df["target_frame"] = pd.to_numeric(
        df["target_frame"], errors="raise"
    ).astype(int)

    if df["target_frame"].duplicated().any():
        raise ValueError(f"{filename}: duplicate target frames found.")

    df = df.sort_values("target_frame").reset_index(drop=True)
    normalized[method] = df

    print(
        f"{method}: frames {df['target_frame'].min()}-"
        f"{df['target_frame'].max()}, rows={len(df)}"
    )

reference_frames = normalized["FFT-TDMD"]["target_frame"].tolist()

for method, df in normalized.items():
    frames = df["target_frame"].tolist()
    matches = frames == reference_frames
    print(f"Frame alignment - {method}: {'PASS' if matches else 'FAIL'}")

    if not matches:
        all_frames_match = False

if not all_frames_match:
    raise ValueError(
        "Target frames differ across methods. "
        "Do not compare the aggregate metrics until this is resolved."
    )

rows = []

for method, df in normalized.items():
    rows.append({
        "Method": method,
        "Frames evaluated": len(df),
        "First target frame": int(df["target_frame"].min()),
        "Last target frame": int(df["target_frame"].max()),
        "Mean MAE": df["mae"].mean(),
        "Mean RMSE": df["rmse"].mean(),
        "Mean frame relative error (%)":
            df["relative_error_percent"].mean(),
        "Worst frame relative error (%)":
            df["relative_error_percent"].max(),
    })

summary = pd.DataFrame(rows).sort_values(
    "Mean frame relative error (%)"
)

output_csv = folder / "climate_method_comparison_verified.csv"
summary.to_csv(output_csv, index=False, float_format="%.10g")

print("\nVERIFIED CLIMATE METHOD COMPARISON")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.8f}"))
print(f"\nSaved: {output_csv}")
print("\nFrame alignment: PASS for all five methods.")
