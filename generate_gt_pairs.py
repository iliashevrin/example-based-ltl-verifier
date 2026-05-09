#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import re

from openpyxl import load_workbook
import random

import pandas as pd

from openai import OpenAI

sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()

import config
import json

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

from utils import collect_aps


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

Use the following atomic proposition mapping:
{atomic_proposition}

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


def ask_chatgpt(client: OpenAI, model: str, requirement: str, atomic_proposition: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    requirement=requirement,
                    atomic_proposition=atomic_proposition,
                ),
            }
        ],
    )

    return normalize_formula(response.choices[0].message.content or "")


def semantically_equivalent(formula_a: str, formula_b: str):
    """
    Returns:
        True  -> semantically equivalent
        False -> valid syntax but not equivalent
        None  -> syntax error, exclude from accuracy
    """
    try:
        f_a = spot.formula(formula_a)
        f_b = spot.formula(formula_b)

        xor_formula = spot.formula.Not(spot.formula.Equiv(f_a, f_b))
        return spot.translate(xor_formula).is_empty()

    except Exception as exc:
        msg = str(exc)

        if "syntax error" in msg.lower():
            print(
                f"Syntax error; excluding from accuracy:\n"
                f"  Error:        {exc}",
                file=sys.stderr,
            )
            return None

        print(
            f"Warning: could not compare formulas:\n"
            f"  Error:        {exc}",
            file=sys.stderr,
        )
        return None


def extract_ap_mapping(ltl_formula: str) -> str:
    """
    Extract AP names from the GT formula using Spot.
    Return them as a set of strings (stringified for the prompt).
    """
    f = spot.formula(ltl_formula)
    aps = sorted(str(ap) for ap in spot.atomic_prop_collect(f))

    return "{" + ", ".join(aps) + "}"



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate natural language requirements to LTL and compare with ground truth."
    )
    parser.add_argument("input", help="Input file")
    parser.add_argument("output_csv", help="Output CSV file")
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="OpenAI model to use, default: gpt-5.4-mini",
    )

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    dataset = []
    parse_errors = 0


    if args.input.endswith("xlsx"):


        wb = load_workbook(args.input, data_only=True)
        ws = wb.active

        pairs = []
        malformed_count = 0

        for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):

            try:
                nl_text = row[1].value  # Column B

                # ------------------------------------------------------------
                # STEP 1-3 : decision2 / bool_expr2
                # ------------------------------------------------------------
                ltl_prefix = None
                used_bool_expr2 = False
                bool_expr2 = None

                h_val = row[7].value  # Column H

                if h_val:
                    h_json = json.loads(h_val)
                    decision2 = h_json[0]["decision2"]

                    if "upon bool_exp2" in decision2[0]:
                        i_val = row[8].value  # Column I

                        if not i_val:
                            raise ValueError("Missing column I")

                        i_json = json.loads(i_val)

                        bool_expr2 = i_json[0]["bool_exp2"][0]
                        used_bool_expr2 = True

                # ------------------------------------------------------------
                # STEP 4-6 : decision1 / bool_expr1
                # ------------------------------------------------------------
                j_val = row[9].value  # Column J

                if not j_val:
                    raise ValueError("Missing column J")

                j_json = json.loads(j_val)
                decision1 = j_json[0]["decision1"]

                bool_expr1 = None

                if "while bool_exp1" in decision1[0] or "whenever bool_exp1" in decision1[0]:
                    k_val = row[10].value  # Column K

                    if not k_val:
                        raise ValueError("Missing column K")

                    k_json = json.loads(k_val)
                    bool_expr1 = k_json[0]["bool_exp1"][0]

                    if used_bool_expr2:
                        ltl_prefix = (
                            f"G(({bool_expr1} & !{bool_expr2} & X({bool_expr2})) -> "
                        )
                    else:
                        ltl_prefix = f"G({bool_expr1} -> "

                else:

                    if "upon bool_exp1" in decision1[0]:

                        k_val = row[10].value  # Column K

                        if not k_val:
                            raise ValueError("Missing column K")

                        k_json = json.loads(k_val)
                        bool_expr1 = k_json[0]["bool_exp1"][0]

                        ltl_prefix = (
                            f"G((!{bool_expr1} & X({bool_expr1})) -> "
                        )

                    elif used_bool_expr2:
                        ltl_prefix = (
                            f"G((!{bool_expr2} & X({bool_expr2})) -> "
                        )
                    else:

                        ltl_prefix = "("

                        # raise ValueError("No usable decision1/decision2 logic")

                # ------------------------------------------------------------
                # STEP 7-8 : decision3 random selection
                # ------------------------------------------------------------
                f_val = row[5].value  # Column F

                if not f_val:
                    raise ValueError("Missing column F")

                f_json = json.loads(f_val)
                decision3 = f_json[0]["decision3"]

                valid_choices = [
                    s for s in decision3
                    if "N_DURATION" not in s
                ]

                if not valid_choices:
                    raise ValueError("No valid decision3 entries")

                chosen = random.choice(valid_choices)

                # ------------------------------------------------------------
                # STEP 9 : bool_expr3 / bool_expr4
                # ------------------------------------------------------------
                g_val = row[6].value  # Column G

                if not g_val:
                    raise ValueError("Missing column G")

                g_json = json.loads(g_val)
                g_first = g_json[0]

                bool_expr3 = g_first["bool_exp3"][0]

                # ------------------------------------------------------------
                # STEP 10-11 : finish formula
                # ------------------------------------------------------------
                suffix = None

                if chosen == "eventually satisfy bool_exp3":
                    suffix = f"F({bool_expr3})"

                elif chosen == "always satisfy bool_exp3":
                    suffix = f"G({bool_expr3})"

                elif chosen == "at the next timepoint satisfy bool_exp3":
                    suffix = f"X({bool_expr3})"

                elif chosen == "immediately satisfy bool_exp3":
                    suffix = f"({bool_expr3})"

                elif chosen == "until bool_exp4, satisfy bool_exp3":
                    bool_expr4 = g_first["bool_exp4"][0]
                    suffix = f"({bool_expr3} U {bool_expr4})"

                else:
                    raise ValueError(f"Unsupported decision3 option: {chosen}")

                ltl_formula = ltl_prefix + suffix + ")"

                try:
                    ap_set = extract_ap_mapping(ltl_formula)
                except Exception as exc:
                    print(
                        f"Skipping item {idx} due to Spot parse error:\n"
                        f"  Formula: {ground_truth}\n"
                        f"  Error:   {exc}",
                        file=sys.stderr,
                    )
                    parse_errors += 1
                    break

                dataset.append((nl_text, ltl_formula, ap_set))

            except Exception as e:
                malformed_count += 1
                print(f"Malformed row {row_idx}: {e}")



    if args.input.endswith("csv"):

        df = pd.read_csv(args.input, sep=',')

        required_columns = {"Natural Language", "Ground Truth", "Atomic Proposition"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for _, row in df.iterrows():
            requirement = str(row["Natural Language"])
            ground_truth = str(row["Ground Truth"]).strip()
            atomic_proposition = str(row["Atomic Proposition"]).strip()

            dataset.append((requirement, ground_truth, atomic_proposition))


    elif args.input == "spacewire.json":

        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Input JSON must contain an array of objects.")


        for idx, item in enumerate(data):
            requirement = str(item.get("text", "")).strip()

            logic_entries = item.get("logics", [])

            if not isinstance(logic_entries, list):
                continue

            found_ltl = False

            for logic in logic_entries:
                if logic.get("type") != "LTL":
                    continue

                found_ltl = True

                f_code = logic.get("f_code", "")

                if not f_code or not str(f_code).strip():
                    break

                ground_truth = str(f_code).strip()

                ground_truth = ground_truth.replace("-->", "->")
                ground_truth = re.sub(r"\bnot\b", "!", ground_truth)
                ground_truth = re.sub(r"\band\b", "&", ground_truth)
                ground_truth = re.sub(r"\bor\b", "&", ground_truth)

                try:
                    ap_set = extract_ap_mapping(ground_truth)
                except Exception as exc:
                    print(
                        f"Skipping item {idx} due to Spot parse error:\n"
                        f"  Formula: {ground_truth}\n"
                        f"  Error:   {exc}",
                        file=sys.stderr,
                    )
                    parse_errors += 1
                    break

                dataset.append((requirement, ground_truth, ap_set))

                # only take the first valid LTL formula
                break

            if not found_ltl:
                continue
                                  

    elif args.input.endswith("json"):

        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Input JSON must contain an array of objects.")


        for idx, item in enumerate(data):
            if "nlTask" not in item or "ltlequ" not in item:
                raise ValueError(f"Element {idx} is missing 'nlTask' or 'ltlequ'.")

            requirement = str(item["nlTask"]).strip()
            ground_truth = str(item["ltlequ"][0]).strip()[:-1]

            # dataset cleanup before Spot parsing
            ground_truth = ground_truth.replace("U", " U ")

            try:
                atomic_proposition = extract_ap_mapping(ground_truth)
            except Exception as exc:
                print(
                    f"Ground-truth parse error at item {idx}; excluding:\n"
                    f"  Ground Truth: {ground_truth}\n"
                    f"  Error:        {exc}",
                    file=sys.stderr,
                )
                parse_errors += 1
                continue

            dataset.append((requirement, ground_truth, atomic_proposition))
                       

    client = OpenAI()

    rows = []
    correct = 0
    total = 0
    syntax_errors = 0


    for requirement, ground_truth, atomic_proposition in dataset:

        model_response = ask_chatgpt(client, args.model, requirement, atomic_proposition)

        equivalent = semantically_equivalent(ground_truth, model_response)

        print(
            f"  Requirement: {requirement}\n"
            f"  Ground Truth: {ground_truth}\n"
            f"  Response:     {model_response}\n"
            f"  Equivalent:     {equivalent}\n",
            file=sys.stderr,
        )

        if equivalent is None:
            syntax_errors += 1
        else:
            total += 1
            correct += int(equivalent)

        if not equivalent:
            rows.append(
                {
                    "Ground Truth": ground_truth,
                    "Response": model_response,
                }
            )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Ground Truth", "Response"],
        )
        writer.writeheader()
        writer.writerows(rows)

    accuracy = correct / total if total else 0.0
    print(f"Total accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Syntax errors excluded: {syntax_errors}")


if __name__ == "__main__":
    main()