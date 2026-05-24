import argparse
import pandas as pd
from collections import defaultdict


def compute_accumulated_source_ratios(data_csv_path, batch_results_pairs):

    overall_found = 0
    overall_total = 0

    # Read main data CSV (semicolon-delimited)
    data_df = pd.read_csv(data_csv_path, sep=";")

    # Validate required columns
    required_data_cols = {"original LTL", "source", "batch_id"}

    missing_data = required_data_cols - set(data_df.columns)

    if missing_data:
        raise ValueError(f"Missing columns in data CSV: {missing_data}")

    # Accumulators across all batches/results
    total_per_source = defaultdict(int)
    found_per_source = defaultdict(int)

    for batch_id, results_csv_path in batch_results_pairs:
        print(f"Processing batch {batch_id} with results file {results_csv_path}")

        # Read results CSV (comma-delimited)
        results_df = pd.read_csv(results_csv_path, sep=",")

        if "Ground Truth" not in results_df.columns:
            raise ValueError(
                f'Missing "Ground Truth" column in results CSV: {results_csv_path}'
            )

        # Filter data for current batch
        batch_data = data_df[data_df["batch_id"] == batch_id]

        # Accumulate total counts per source
        batch_totals = (
            batch_data.groupby("source")
            .size()
            .to_dict()
        )

        overall_total += len(batch_data)

        for source, count in batch_totals.items():
            total_per_source[source] += count

        # Build lookup: original LTL -> source
        ltl_to_source = {}

        for _, row in batch_data.iterrows():
            ltl = str(row["original LTL"]).strip()
            source = row["source"]

            if ltl not in ltl_to_source:
                ltl_to_source[ltl] = source

        # Count matches from results CSV
        for _, row in results_df.iterrows():
            ground_truth = str(row["Ground Truth"]).strip()

            if ground_truth in ltl_to_source:
                source = ltl_to_source[ground_truth]
                found_per_source[source] += 1

                overall_found += 1

    # Final output
    print("\nFinal accumulated ratios")
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
        description="Compute accumulated complement ratios across batches/results."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to semicolon-delimited data CSV file"
    )

    parser.add_argument(
        "--batch_results",
        nargs="+",
        required=True,
        metavar=("BATCH_ID", "RESULTS_CSV"),
        help=(
            "Pairs of batch_id and results CSV path. "
            "Example: --batch_results 1 results1.csv 2 results2.csv"
        )
    )

    args = parser.parse_args()

    # Validate pair structure
    if len(args.batch_results) % 2 != 0:
        raise ValueError(
            "batch_results arguments must appear in pairs: "
            "batch_id results_csv"
        )

    batch_results_pairs = []

    for i in range(0, len(args.batch_results), 2):
        batch_id = int(args.batch_results[i])
        results_csv = args.batch_results[i + 1]

        batch_results_pairs.append((batch_id, results_csv))

    compute_accumulated_source_ratios(
        data_csv_path=args.data,
        batch_results_pairs=batch_results_pairs
    )