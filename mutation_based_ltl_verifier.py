#!/usr/bin/env python3
import re
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
from utils import get_words_from_conditions, check_acceptance

from enum import Enum
import random




class Mutation(str, Enum):
    SWAP_F_WITH_G = "SWAP_F_WITH_G"
    SWAP_F_WITH_X = "SWAP_F_WITH_X"
    SWAP_G_WITH_F = "SWAP_G_WITH_F"
    SWAP_G_WITH_X = "SWAP_G_WITH_X"
    SWAP_X_WITH_F = "SWAP_X_WITH_F"
    SWAP_X_WITH_G = "SWAP_X_WITH_G"

    REMOVE_F = "REMOVE_F"
    REMOVE_G = "REMOVE_G"

    SWAP_U_WITH_W = "SWAP_U_WITH_W"
    SWAP_W_WITH_U = "SWAP_W_WITH_U"
    SWAP_U_OPERANDS = "SWAP_U_OPERANDS"
    SWAP_W_OPERANDS = "SWAP_W_OPERANDS"

    ADD_NEGATION = "ADD_NEGATION"
    REMOVE_NEGATION = "REMOVE_NEGATION"

    ADD_X = "ADD_X"
    REMOVE_X = "REMOVE_X"

    ADD_F = "ADD_F"
    ADD_G = "ADD_G"

    SWAP_AND_WITH_OR = "SWAP_AND_WITH_OR"
    SWAP_OR_WITH_AND = "SWAP_OR_WITH_AND"

    SWAP_IMPLIES_WITH_EQUIV = "SWAP_IMPLIES_WITH_EQUIV"
    SWAP_EQUIV_WITH_IMPLIES = "SWAP_EQUIV_WITH_IMPLIES"


BEST_ORDER = [
    Mutation.SWAP_G_WITH_X,
    Mutation.REMOVE_F,
    Mutation.SWAP_AND_WITH_OR,
    Mutation.SWAP_IMPLIES_WITH_EQUIV,
    Mutation.SWAP_G_WITH_F,
    Mutation.REMOVE_NEGATION,
    Mutation.ADD_X,
    Mutation.ADD_F,
    Mutation.ADD_G,
    Mutation.ADD_NEGATION,
]



def children_count(f):
    return f.size()


def replace_child(f, target_idx, new_child):
    idx = -1

    def mapper(child):
        nonlocal idx
        idx += 1
        return new_child if idx == target_idx else child

    return f.map(mapper)


def current_node_mutations(f):
    muts = []

    # Add negation to any subformula
    if not f._is(spot.op_Not):
        muts.append((spot.formula.Not(f), Mutation.ADD_NEGATION))

    # Remove negation from any negated subformula
    if f._is(spot.op_Not) and f.size() == 1:
        muts.append((f[0], Mutation.REMOVE_NEGATION))

    # Add X to any subformula
    muts.append((spot.formula.X(f), Mutation.ADD_X))

    # Add F to any subformula, except if it already starts with F
    if not f._is(spot.op_F):
        muts.append((spot.formula.F(f), Mutation.ADD_F))

    # Add G to any subformula, except if it already starts with G
    if not f._is(spot.op_G):
        muts.append((spot.formula.G(f), Mutation.ADD_G))

    # Remove X from any X-subformula
    if f._is(spot.op_X) and f.size() == 1:
        muts.append((f[0], Mutation.REMOVE_X))

    # Swap F/G/X
    if f._is(spot.op_F):
        muts.append((spot.formula.G(f[0]), Mutation.SWAP_F_WITH_G))
        muts.append((spot.formula.X(f[0]), Mutation.SWAP_F_WITH_X))
        muts.append((f[0], Mutation.REMOVE_F))

    elif f._is(spot.op_G):
        muts.append((spot.formula.F(f[0]), Mutation.SWAP_G_WITH_F))
        muts.append((spot.formula.X(f[0]), Mutation.SWAP_G_WITH_X))
        muts.append((f[0], Mutation.REMOVE_G))

    elif f._is(spot.op_X):
        muts.append((spot.formula.F(f[0]), Mutation.SWAP_X_WITH_F))
        muts.append((spot.formula.G(f[0]), Mutation.SWAP_X_WITH_G))

    # U/W mutations
    if f._is(spot.op_U):
        muts.append((spot.formula.W(f[0], f[1]), Mutation.SWAP_U_WITH_W))
        muts.append((spot.formula.U(f[1], f[0]), Mutation.SWAP_U_OPERANDS))

    elif f._is(spot.op_W):
        muts.append((spot.formula.U(f[0], f[1]), Mutation.SWAP_W_WITH_U))
        muts.append((spot.formula.W(f[1], f[0]), Mutation.SWAP_W_OPERANDS))

    # Boolean mutations
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

    # Implication / equivalence mutations
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


def mutation_expert(candidate):

    traces = mutation_gradual(candidate)
    rank = {str(value): i for i, value in enumerate(EXPERT_ORDER)}
    traces.sort(key=lambda trace: rank.get(trace[2], -1), reverse=True)

    return traces


def mutation_random(formula):
    traces = mutation_gradual(formula)
    random.shuffle(traces)
    return traces

def mutation_gradual(formula):
    mutants = mutate_ltl_formula(formula)

    traces = []
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

            used_traces.add(trace)

            is_positive = check_acceptance(original_aut, trace)
            if is_positive == True:
                traces.append((trace, True, mutation_type))
            elif is_positive == False:
                traces.append((trace, False, mutation_type))


    return traces
