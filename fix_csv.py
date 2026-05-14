import csv
import re
from pathlib import Path
import argparse


# Original broken delimiter
OLD_DELIMITER = ";|||;"
# Desired delimiter
NEW_DELIMITER = ";"

# Regex:
# Matches all of these variants:
#
# "text1(text2)"
# "text1(text2
# text1(text2)"
# "text1(text2,text3)"
#
# Quotes on either side are optional.
#
PATTERN_QUOTES = re.compile(r'"?([^"\(\)]+)\(([^"\(\)]+)\)"?')

# Regex:
# In column 2 only:
# Add a space after G/F/U/X if followed by lowercase letter
#
PATTERN_SPACING = re.compile(r'([GFUX])([a-z])')


# Regex:
# Replace hyphen with underscore
# EXCEPT when hyphen is part of ->
#
# Matches "-" not preceded by "-" and not followed by ">"
#
PATTERN_HYPHEN = re.compile(r'-(?!>)')


def transform_quotes(value: str) -> str:
    """
    Replace:
        "text1(text2)"
        "text1(text2,text3)"
        "text1(text2,text3,...,textN)"

    with:
        text1_text2
        text1_text2_text3
        text1_text2_..._textN
    """

    def replacer(match):

        text1 = match.group(1).strip()
        inner = match.group(2).strip()

        # Split by commas and remove extra spaces
        parts = [p.strip() for p in inner.split(",")]

        # Combine everything with underscores
        return "_".join([text1] + parts)

    return PATTERN_QUOTES.sub(replacer, value)


def add_spacing(value: str) -> str:
    """
    Add space after G/F/U/X when followed by lowercase letter.
    """

    return PATTERN_SPACING.sub(r'\1 \2', value)


def replace_hyphens(value: str) -> str:
    """
    Replace hyphens with underscores,
    except when part of the implies symbol '->'
    """

    return PATTERN_HYPHEN.sub("_", value)


def process_csv(input_file: str, output_file: str):

    # Read raw content first because delimiter is malformed
    raw_text = Path(input_file).read_text(encoding="utf-8")

    # Fix delimiter before CSV parsing
    fixed_text = raw_text.replace(OLD_DELIMITER, NEW_DELIMITER)

    # Split into lines for CSV reader
    lines = fixed_text.splitlines()

    processed_rows = []

    reader = csv.reader(lines, delimiter=NEW_DELIMITER)

    for row in reader:

        # Process column 2 (index 1)
        if len(row) > 1:
            row[1] = transform_quotes(row[1])
            row[1] = add_spacing(row[1])
            row[1] = replace_hyphens(row[1])

        # Process column 3 (index 2)
        if len(row) > 2:
            row[2] = transform_quotes(row[2])
            row[2] = replace_hyphens(row[2])


        processed_rows.append(row)

    # Write cleaned CSV
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=NEW_DELIMITER)
        writer.writerows(processed_rows)

    print(f"Processed file written to: {output_file}")



def main():
    parser = argparse.ArgumentParser(
        description="Fix delimiters and process CSV content."
    )

    parser.add_argument(
        "input_file",
        help="Path to input CSV file"
    )

    parser.add_argument(
        "output_file",
        help="Path to output CSV file"
    )

    args = parser.parse_args()

    process_csv(args.input_file, args.output_file)


if __name__ == "__main__":
    main()