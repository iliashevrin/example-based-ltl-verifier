#!/usr/bin/env python3
import argparse
import csv
import os
import sys

import pandas as pd

from openai import OpenAI

sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()

import config

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY


PROMPT = """Translate the following natural language requirement into Linear Temporal Logic (LTL).

Use STRICT programming-style notation:
- F for "finally"
- G for "globally"
- X for "next"
- U for "until"
- ! for negation
- & for AND
- | for OR
- -> for implication
- <-> for equivalence (iff)

Do NOT use LaTeX or any mathematical symbols such as ◇, □, ¬, ∧, ∨, etc.
Do NOT use the equality sign.
Do NOT include explanations, comments, or multiple formulas.

Return EXACTLY one LTL formula as a single line.
The output must be directly parseable as an LTL formula.

Requirement:
{requirement}
"""


def normalize_formula(text: str) -> str:
    """Strip common formatting so the result is parser-friendly."""
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Keep only the first non-empty line if the model accidentally adds extra text.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def ask_chatgpt(client: OpenAI, model: str, requirement: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(requirement=requirement),
            }
        ],
    )

    return normalize_formula(response.choices[0].message.content or "")


def semantically_equivalent(formula_a: str, formula_b: str) -> bool:
    """
    Checks LTL semantic equivalence using Spot.

    Two formulas are equivalent iff their XOR is unsatisfiable:
        !(a <-> b)
    """
    try:
        f_a = spot.formula(formula_a)
        f_b = spot.formula(formula_b)

        xor_formula = spot.formula.Not(spot.formula.Equiv(f_a, f_b))
        return spot.translate(xor_formula).is_empty()

    except Exception as exc:
        print(
            f"Warning: could not compare formulas:\n"
            f"  Ground Truth: {formula_a}\n"
            f"  Response:     {formula_b}\n"
            f"  Error:        {exc}",
            file=sys.stderr,
        )
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate natural language requirements to LTL and compare with ground truth."
    )
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("output_csv", help="Output CSV file")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model to use, default: gpt-4.1-mini",
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    df = pd.read_csv(args.input_csv)

    required_columns = {"Natural Language", "Ground Truth"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    client = OpenAI()

    rows = []
    correct = 0
    total = 0

    for _, row in df.iterrows():
        requirement = str(row["Natural Language"])
        ground_truth = str(row["Ground Truth"]).strip()

        model_response = ask_chatgpt(client, args.model, requirement)

        equivalent = semantically_equivalent(ground_truth, model_response)

        rows.append(
            {
                "Ground Truth": ground_truth,
                "Response": model_response,
                "Equivalent": equivalent,
            }
        )

        total += 1
        correct += int(equivalent)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Ground Truth", "Response", "Equivalent"],
        )
        writer.writeheader()
        writer.writerows(rows)

    accuracy = correct / total if total else 0.0
    print(f"Total accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()