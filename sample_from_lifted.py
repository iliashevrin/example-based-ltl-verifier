import csv
import random
import argparse


def sample_csv(input_file, output_file, sample_size, delimiter=";"):
    """
    Randomly sample a fixed number of rows from a CSV file.
    """

    # Read all rows
    with open(input_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)

        rows = list(reader)

    # Check sample size
    if sample_size > len(rows):
        raise ValueError(
            f"Sample size ({sample_size}) is larger than "
            f"number of rows ({len(rows)})"
        )

    # Randomly choose rows
    sampled_rows = random.sample(rows, sample_size)

    # Write sampled rows
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(sampled_rows)

    print(f"Saved {sample_size} random rows to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample rows from a CSV file."
    )

    parser.add_argument(
        "input_file",
        help="Path to input CSV file"
    )

    parser.add_argument(
        "output_file",
        help="Path to output CSV file"
    )

    parser.add_argument(
        "sample_size",
        type=int,
        help="Number of random rows to select"
    )

    parser.add_argument(
        "--delimiter",
        default=";",
        help="CSV delimiter (default: ;)"
    )

    args = parser.parse_args()

    sample_csv(
        args.input_file,
        args.output_file,
        args.sample_size,
        args.delimiter
    )


if __name__ == "__main__":
    main()