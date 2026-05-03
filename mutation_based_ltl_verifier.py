#!/usr/bin/env python3
import re
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
from utils import get_words_from_conditions, check_acceptance

from enum import Enum



class Mutation(str, Enum):
    SWAP_F_WITH_G = "SWAP_F_WITH_G"
    SWAP_F_WITH_X = "SWAP_F_WITH_X"
    SWAP_G_WITH_F = "SWAP_G_WITH_F"
    SWAP_G_WITH_X = "SWAP_G_WITH_X"
    SWAP_X_WITH_F = "SWAP_X_WITH_F"
    SWAP_X_WITH_G = "SWAP_X_WITH_G"

    SWAP_U_WITH_W = "SWAP_U_WITH_W"
    SWAP_W_WITH_U = "SWAP_W_WITH_U"

    SWAP_U_OPERANDS = "SWAP_U_OPERANDS"
    SWAP_W_OPERANDS = "SWAP_W_OPERANDS"

    NEGATE_LITERAL = "NEGATE_LITERAL"
    UNNEGATE_LITERAL = "UNNEGATE_LITERAL"

    ADD_X_TO_LITERAL = "ADD_X_TO_LITERAL"
    REMOVE_X_FROM_LITERAL = "REMOVE_X_FROM_LITERAL"

    SWAP_AND_WITH_OR = "SWAP_AND_WITH_OR"
    SWAP_OR_WITH_AND = "SWAP_OR_WITH_AND"

    SWAP_IMPLIES_WITH_EQUIV = "SWAP_IMPLIES_WITH_EQUIV"
    SWAP_EQUIV_WITH_IMPLIES = "SWAP_EQUIV_WITH_IMPLIES"



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
    """
    Returns:
        list[tuple[spot.formula, Mutation]]
    """
    muts = []

    # 1. Swap F with G and X
    if f._is(spot.op_F):
        muts.append((spot.formula.G(f[0]), Mutation.SWAP_F_WITH_G))
        muts.append((spot.formula.X(f[0]), Mutation.SWAP_F_WITH_X))

    elif f._is(spot.op_G):
        muts.append((spot.formula.F(f[0]), Mutation.SWAP_G_WITH_F))
        muts.append((spot.formula.X(f[0]), Mutation.SWAP_G_WITH_X))

    elif f._is(spot.op_X):
        muts.append((spot.formula.F(f[0]), Mutation.SWAP_X_WITH_F))
        muts.append((spot.formula.G(f[0]), Mutation.SWAP_X_WITH_G))

        # 5. Remove X if it is in front of a literal
        if is_plain_literal(f[0]) or is_negated_literal(f[0]):
            muts.append((f[0], Mutation.REMOVE_X_FROM_LITERAL))

    # 2. Swap U with W
    if f._is(spot.op_U):
        muts.append((spot.formula.W(f[0], f[1]), Mutation.SWAP_U_WITH_W))

        # 3. Swap operands in U
        muts.append((spot.formula.U(f[1], f[0]), Mutation.SWAP_U_OPERANDS))

    elif f._is(spot.op_W):
        muts.append((spot.formula.U(f[0], f[1]), Mutation.SWAP_W_WITH_U))

        # 3. Swap operands in W
        muts.append((spot.formula.W(f[1], f[0]), Mutation.SWAP_W_OPERANDS))

    # 4. Negate literals and un-negate negated literals
    if is_plain_literal(f):
        muts.append((spot.formula.Not(f), Mutation.NEGATE_LITERAL))

        # 5. Add X in front of literals
        muts.append((spot.formula.X(f), Mutation.ADD_X_TO_LITERAL))

    elif is_negated_literal(f):
        muts.append((f[0], Mutation.UNNEGATE_LITERAL))

        # 5. Add X in front of negated literals
        muts.append((spot.formula.X(f), Mutation.ADD_X_TO_LITERAL))

    # 6. Swap & with |
    if f._is(spot.op_And):
        muts.append(
            (
                spot.formula.Or([f[i] for i in range(children_count(f))]),
                Mutation.SWAP_AND_WITH_OR,
            )
        )

    elif f._is(spot.op_Or):
        muts.append(
            (
                spot.formula.And([f[i] for i in range(children_count(f))]),
                Mutation.SWAP_OR_WITH_AND,
            )
        )

    # 7. Swap -> with <->
    if f._is(spot.op_Implies):
        muts.append((spot.formula.Equiv(f[0], f[1]), Mutation.SWAP_IMPLIES_WITH_EQUIV))

    elif f._is(spot.op_Equiv):
        muts.append((spot.formula.Implies(f[0], f[1]), Mutation.SWAP_EQUIV_WITH_IMPLIES))

    return muts


def generate_mutants_at_all_nodes(f):
    """
    Generate formulas where exactly one mutation is applied somewhere in the AST.

    Yields:
        tuple[spot.formula, Mutation]
    """
    for mutated, mutation_type in current_node_mutations(f):
        yield mutated, mutation_type

    for i in range(children_count(f)):
        child = f[i]

        for mutated_child, mutation_type in generate_mutants_at_all_nodes(child):
            yield replace_child(f, i, mutated_child), mutation_type


def mutate_ltl_formula(formula_str):
    """
    Args:
        formula_str: LTL formula string

    Returns:
        list[tuple[str, Mutation]]:
            Each item is (mutated_formula_string, mutation_type).
    """
    original = spot.formula(formula_str)
    original_str = str(original)

    mutants = []
    seen = set()

    for mutant, mutation_type in generate_mutants_at_all_nodes(original):
        mutant_str = str(mutant)
        key = (mutant_str, mutation_type)

        if mutant_str != original_str and key not in seen:
            seen.add(key)
            mutants.append((mutant_str, mutation_type))

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

    for mutant, mutation_type in mutants:
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
                positives.append((trace, mutation_type))
            elif is_positive == False:
                negatives.append((trace, mutation_type))

    return positives, negatives
