#!/usr/bin/env python3
"""
Sample an LLM's LTL translations of natural-language requirements repeatedly
at a diversity-inducing temperature, cluster the samples by SEMANTIC
equivalence (via Spot), and report the normalized entropy of that cluster
distribution per requirement.

Unlike generate_gt_pairs.py, this script never looks at any ground-truth
formula in the input dataset -- only the NL requirement text is used to
prompt the LLM. Dataset parsing is reused from generate_gt_pairs.load_dataset.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import spot
spot.setup()

import config
from generate_gt_pairs import load_dataset, normalize_formula, semantically_equivalent

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ["GEMINI_API_KEY"] = config.GEMINI_API_KEY
os.environ["ANTHROPIC_API_KEY"] = config.ANTHROPIC_API_KEY


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

Choose concise, descriptive atomic proposition names based on the requirement text.

Return EXACTLY one LTL formula as a single line.
The output must be directly parseable as an LTL formula.

Requirement:
{requirement}
"""


def ask_claude(client: anthropic.Anthropic, model: str, requirement: str, temperature: float) -> str:
    # Current-generation models dropped `temperature` as a typed SDK kwarg
    # (sampling controls are removed/400 on those models anyway); pass it
    # via extra_body, which still reaches the API for models that honor it
    # (e.g. claude-sonnet-4-6, the default here).
    response = client.messages.create(
        model=model,
        max_tokens=128,
        extra_body={"temperature": temperature},
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(requirement=requirement),
            }
        ],
    )

    return normalize_formula(response.content[0].text)


def ask_gemini(client: "genai.Client", model: str, requirement: str, temperature: float) -> str:
    from google import genai

    response = client.models.generate_content(
        model=model,
        contents=PROMPT.format(requirement=requirement),
        config=genai.types.GenerateContentConfig(temperature=temperature),
    )

    return normalize_formula(response.text or "")


def ask_chatgpt(client: "OpenAI", model: str, requirement: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(requirement=requirement),
            }
        ],
    )

    return normalize_formula(response.choices[0].message.content or "")


DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-flash",
    "gpt": "gpt-5.4-mini",
}

ASK_FUNCTIONS = {
    "claude": ask_claude,
    "gemini": ask_gemini,
    "gpt": ask_chatgpt,
}


def build_client(llm: str):
    if llm == "claude":
        import anthropic

        return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if llm == "gemini":
        from google import genai

        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if llm == "gpt":
        from openai import OpenAI

        return OpenAI()
    raise ValueError(f"Unknown LLM: {llm}")


def cluster_by_semantic_equivalence(formulas: list[str]) -> list[list[str]]:
    """
    Greedily group formulas into clusters of semantic equivalence.

    A formula that fails to parse is placed in its own singleton cluster,
    grouped only with other formulas that are byte-for-byte identical
    (since semantic equivalence can't be checked without a parse).
    """
    clusters: list[list[str]] = []
    representatives: list[str] = []

    for formula in formulas:
        placed = False

        try:
            spot.formula(formula)
            parseable = True
        except Exception:
            parseable = False

        for cluster, representative in zip(clusters, representatives):
            if not parseable:
                if formula == representative:
                    cluster.append(formula)
                    placed = True
                    break
                continue

            if semantically_equivalent(representative, formula):
                cluster.append(formula)
                placed = True
                break

        if not placed:
            clusters.append([formula])
            representatives.append(formula)

    return clusters


def normalized_entropy(clusters: list[list[str]], n: int) -> float:
    """Shannon entropy (nats) over the cluster-size distribution, squashed to
    [0, 1) via 1 - e^-H. H = 0 (single cluster) maps to 0; H grows unboundedly
    as clusters flatten out, approaching 1."""
    entropy = 0.0
    for cluster in clusters:
        p = len(cluster) / n
        entropy -= p * math.log(p)

    return 1 - math.exp(-entropy)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample an LLM's LTL translation of each NL requirement n times at a "
            "diversity-inducing temperature, and report the normalized semantic "
            "entropy of the responses."
        )
    )
    parser.add_argument("input", help="Input dataset file (.txt/.xlsx/.csv/.json)")
    parser.add_argument("output_csv", help="Output CSV file (columns: req, normalized_entropy)")
    parser.add_argument(
        "--llm",
        choices=sorted(ASK_FUNCTIONS.keys()),
        default="claude",
        help="Which LLM provider to query, default: claude",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default model name for the chosen --llm",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to draw per requirement, default: 10",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature, default: 0.7",
    )

    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")

    model = args.model or DEFAULT_MODELS[args.llm]
    ask = ASK_FUNCTIONS[args.llm]
    client = build_client(args.llm)

    dataset, parse_errors = load_dataset(args.input)
    if parse_errors:
        print(f"Dataset parse errors (ignored, entries skipped): {parse_errors}", file=sys.stderr)

    # Ground truth / atomic propositions are intentionally discarded here:
    # only the NL requirement text drives the prompt. Dedup requirements so
    # each distinct text is sampled (and reported) exactly once.
    requirements = list(dict.fromkeys(requirement for requirement, _, _ in dataset))

    rows = []

    for requirement in requirements:
        samples = [
            ask(client, model, requirement, args.temperature)
            for _ in range(args.num_samples)
        ]

        clusters = cluster_by_semantic_equivalence(samples)
        entropy = normalized_entropy(clusters, args.num_samples)

        print(
            f"  Requirement: {requirement}\n"
            f"  Samples:     {samples}\n"
            f"  Clusters:    {len(clusters)}\n"
            f"  Norm. entropy: {entropy:.4f}\n",
            file=sys.stderr,
        )

        rows.append({"req": requirement, "normalized_entropy": entropy})

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["req", "normalized_entropy"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} requirement(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
