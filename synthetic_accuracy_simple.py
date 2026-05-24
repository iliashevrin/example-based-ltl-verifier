import argparse
import pandas as pd
from collections import defaultdict


def compute_source_ratios(data_csv_path, results_csv_path):
    # Read CSV files
    data_df = pd.read_csv(data_csv_path, sep=";")
    results_df = pd.read_csv(results_csv_path, sep=",")

    # Validate required columns
    required_data_cols = {"original LTL", "Spot LTL", "source"}
    required_results_cols = {"Ground Truth"}

    missing_data = required_data_cols - set(data_df.columns)
    missing_results = required_results_cols - set(results_df.columns)

    if missing_data:
        raise ValueError(f"Missing columns in data CSV: {missing_data}")

    if missing_results:
        raise ValueError(f"Missing columns in results CSV: {missing_results}")

    # Count total rows per source
    total_per_source = (
        data_df.groupby("source")
        .size()
        .to_dict()
    )

    # Build lookup: both original LTL and Spot LTL -> source
    ltl_to_source = {}

    for _, row in data_df.iterrows():
        source = row["source"]

        original_ltl = str(row["original LTL"]).strip()
        spot_ltl = str(row["Spot LTL"]).strip()

        if original_ltl and original_ltl not in ltl_to_source:
            ltl_to_source[original_ltl] = source

        if spot_ltl and spot_ltl not in ltl_to_source:
            ltl_to_source[spot_ltl] = source

    # Count found rows per source
    found_per_source = defaultdict(int)

    overall_found = 0
    overall_total = len(data_df)

    for _, row in results_df.iterrows():
        ground_truth = str(row["Ground Truth"]).strip()

        if ground_truth in ltl_to_source:
            source = ltl_to_source[ground_truth]

            found_per_source[source] += 1
            overall_found += 1

    # Print per-source complement ratios
    print("Per-source complement ratios")
    print("=" * 50)

    for source, total_count in total_per_source.items():
        found_count = found_per_source.get(source, 0)

        # Complement ratio: missing / total
        ratio = 1 - (found_count / total_count) if total_count > 0 else 0.0

        print(
            f"Source: {source}\n"
            f"  Found: {found_count}\n"
            f"  Total: {total_count}\n"
            f"  Complement Ratio: {ratio:.4f}\n"
        )

    # Overall found ratio
    overall_ratio = 1 - (
        overall_found / overall_total
        if overall_total > 0 else 0.0
    )

    print("=" * 50)
    print("Overall")
    print(
        f"  Found: {overall_found}\n"
        f"  Total: {overall_total}\n"
        f"  Found Ratio: {overall_ratio:.4f}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-source complement ratios from a single results CSV."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to semicolon-delimited data CSV file"
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Path to comma-delimited results CSV file"
    )

    args = parser.parse_args()

    compute_source_ratios(
        data_csv_path=args.data,
        results_csv_path=args.results
    )