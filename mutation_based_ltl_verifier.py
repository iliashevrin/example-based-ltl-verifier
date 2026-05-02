#!/usr/bin/env python3
import re
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
from utils import get_words_from_conditions, check_acceptance





def children_count(f):
    return f.size()


def replace_child(f, target_idx, new_child):
    idx = -1

    def mapper(child):
        nonlocal idx
        idx += 1
        return new_child if idx == target_idx else child

    return f.map(mapper)


def is_negated_literal(f):
    return f._is(spot.op_Not) and children_count(f) == 1 and f[0]._is(spot.op_ap)


def is_plain_literal(f):
    return f._is(spot.op_ap)


def current_node_mutations(f):
    muts = []

    # 1. Swap F with G and X
    if f._is(spot.op_F):
        muts.append(spot.formula.G(f[0]))
        muts.append(spot.formula.X(f[0]))

    elif f._is(spot.op_G):
        muts.append(spot.formula.F(f[0]))
        muts.append(spot.formula.X(f[0]))

    elif f._is(spot.op_X):
        # 5. Remove X if it is in front of a literal
        if is_plain_literal(f[0]) or is_negated_literal(f[0]):
            muts.append(f[0])

        # 1. Swap X with F and G
        muts.append(spot.formula.F(f[0]))
        muts.append(spot.formula.G(f[0]))

    # 2. Swap U with W
    if f._is(spot.op_U):
        muts.append(spot.formula.W(f[0], f[1]))

        # 3. Swap operands in U
        muts.append(spot.formula.U(f[1], f[0]))

    elif f._is(spot.op_W):
        muts.append(spot.formula.U(f[0], f[1]))

        # 3. Swap operands in W
        muts.append(spot.formula.W(f[1], f[0]))

    # 4. Negate literals and un-negate negated literals
    if is_plain_literal(f):
        muts.append(spot.formula.Not(f))

        # 5. Add X in front of literals
        muts.append(spot.formula.X(f))

    elif is_negated_literal(f):
        muts.append(f[0])

        # 5. Add X in front of negated literals
        muts.append(spot.formula.X(f))

    # 6. Swap & with |
    if f._is(spot.op_And):
        muts.append(spot.formula.Or([f[i] for i in range(children_count(f))]))

    elif f._is(spot.op_Or):
        muts.append(spot.formula.And([f[i] for i in range(children_count(f))]))

    # 7. Swap -> with <->
    if f._is(spot.op_Implies):
        muts.append(spot.formula.Equiv(f[0], f[1]))

    elif f._is(spot.op_Equiv):
        muts.append(spot.formula.Implies(f[0], f[1]))

    return muts


def generate_mutants_at_all_nodes(f):
    """
    Generate formulas where exactly one mutation is applied somewhere in the AST.
    """
    # Mutations at this node
    for m in current_node_mutations(f):
        yield m

    # Mutations inside children
    for i in range(children_count(f)):
        child = f[i]
        for mutated_child in generate_mutants_at_all_nodes(child):
            yield replace_child(f, i, mutated_child)


def mutate_ltl_formula(formula_str):
    """
    Args:
        formula_str: LTL formula string

    Returns:
        list[str]: unified deduplicated list of mutated formulas
    """
    original = spot.formula(formula_str)

    mutants = []
    seen = set()

    for mutant in generate_mutants_at_all_nodes(original):
        mutant_str = str(mutant)

        if mutant_str not in seen and mutant_str != str(original):
            seen.add(mutant_str)
            mutants.append(mutant_str)

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
    mutants = mutate_ltl_formula(formula)

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
