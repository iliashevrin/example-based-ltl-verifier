#!/usr/bin/env python3
import re
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
from utils import get_words_from_conditions, check_acceptance


MUTATIONS = [
    (r"\bF\b", "G"),
    (r"\bG\b", "F"),
    (r"\bU\b", "W"),
    (r"\bW\b", "U"),
    (r"\bR\b", "M"),
    (r"\bM\b", "R"),
    (r"\bG\b", "X"),
    (r"\bX\b", "G"),
    (r"\bF\b", "X"),
    (r"\bX\b", "F"),
    (r"\bX\b\s*", ""),
    (r"\bF\b\s*", ""),
    (r"\bG\b\s*", ""),
    (r"&&|\band\b|&", "|"),
    (r"\|\||\bor\b|\|", "&"),
    (r"->", "<->"),
    (r"<->", "->"),
]


def parse_formula(s):
    try:
        return spot.formula(s)
    except RuntimeError:
        return None


def canonical(s):
    f = parse_formula(s)
    return str(f) if f else None


def regex_mutations(formula):
    mutants = []

    for pattern, replacement in MUTATIONS:
        for m in re.finditer(pattern, formula):
            candidate = formula[:m.start()] + replacement + formula[m.end():]
            mutants.append(candidate)

    return mutants


def literal_negation_mutations(formula):
    """
    Toggle every atomic proposition:
      a  -> !a
      !a -> a

    This assumes atomic propositions are simple identifiers:
      a, b, req, grant, p1, etc.
    """
    mutants = []

    token_pattern = re.compile(r"!?\b[a-zA-Z_][a-zA-Z0-9_]*\b")

    reserved = {
        "F", "G", "X", "U", "W", "R", "M",
        "true", "false", "tt", "ff",
        "and", "or",
    }

    for m in token_pattern.finditer(formula):
        token = m.group()

        bare = token[1:] if token.startswith("!") else token

        if bare in reserved:
            continue

        if token.startswith("!"):
            replacement = bare
        else:
            replacement = f"!{token}"

        candidate = formula[:m.start()] + replacement + formula[m.end():]
        mutants.append(candidate)

    return mutants


def until_like_swap_mutations(formula):
    """
    Swap operands in simple binary temporal expressions:
      a U b -> b U a
      a W b -> b W a
      a R b -> b R a
      a M b -> b M a

    This handles simple operands such as:
      a U b
      !a U b
      X a U F b
      (a && b) U c
      a U (b || c)

    Complex nested cases are still validated by Spot afterward.
    """
    mutants = []

    ops = ["U", "W", "R", "M"]

    operand = r"(?:!?[a-zA-Z_][a-zA-Z0-9_]*|[FGX]\s+!?[a-zA-Z_][a-zA-Z0-9_]*|\([^()]+\))"

    for op in ops:
        pattern = re.compile(rf"({operand})\s+\b{op}\b\s+({operand})")

        for m in pattern.finditer(formula):
            left = m.group(1)
            right = m.group(2)

            swapped = f"{right} {op} {left}"
            candidate = formula[:m.start()] + swapped + formula[m.end():]
            mutants.append(candidate)

    return mutants


def mutate_formula(formula):
    original = canonical(formula)
    if original is None:
        raise ValueError(f"Invalid LTL formula: {formula}")

    candidates = []
    candidates.extend(regex_mutations(formula))
    candidates.extend(literal_negation_mutations(formula))
    candidates.extend(until_like_swap_mutations(formula))

    mutants = []
    seen = {original}

    for candidate in candidates:
        cand_canon = canonical(candidate)

        if cand_canon and cand_canon not in seen:
            seen.add(cand_canon)
            mutants.append(cand_canon)

    return mutants


def accepting_traces(formula):
    aut = spot.translate(formula)
    run = aut.accepting_run()
    if run is None:
        return []


    conditions = []

    # Prefix part
    for edge in run.prefix:
        conditions.append(
            spot.bdd_format_formula(aut.get_dict(), edge.label)
        )

    cycle_start = len(conditions)

    # Cycle part
    for edge in run.cycle:
        conditions.append(
            spot.bdd_format_formula(aut.get_dict(), edge.label)
        )

    return get_words_from_conditions(conditions, cycle_start)



def rejecting_traces(formula):
    return accepting_traces(f"!({formula})")




def generate_traces_by_mutation(formula):
    mutants = mutate_formula(formula)

    positives = []
    negatives = []
    used_traces = set()

    original_aut = spot.translate(formula)

    for mutant in mutants:
        # Generate candidate traces from mutant and negated mutant
        candidates = []
        candidates.extend(accepting_traces(mutant))
        candidates.extend(rejecting_traces(mutant))


        for trace in candidates:
            if trace is None or trace in used_traces:
                continue

            is_positive = check_acceptance(original_aut, trace)

            used_traces.add(trace)

            if is_positive == True:
                positives.append((mutant, trace))
            elif is_positive == False:
                negatives.append((mutant, trace))

    return positives, negatives
