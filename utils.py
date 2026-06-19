#!/usr/bin/env python3
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
import itertools
import re

from enum import Enum


class Mutation(str, Enum):

    SWAP_F_WITH_G = "SWAP_F_WITH_G"
    SWAP_F_WITH_X = "SWAP_F_WITH_X"
    SWAP_G_WITH_F = "SWAP_G_WITH_F"
    SWAP_G_WITH_X = "SWAP_G_WITH_X"
    SWAP_X_WITH_F = "SWAP_X_WITH_F"
    SWAP_X_WITH_G = "SWAP_X_WITH_G"

    REMOVE_F = "REMOVE_F"
    REMOVE_G = "REMOVE_G"
    REMOVE_X = "REMOVE_X"
    REMOVE_NEGATION = "REMOVE_NEGATION"

    SWAP_U_WITH_W = "SWAP_U_WITH_W"
    SWAP_W_WITH_U = "SWAP_W_WITH_U"

    SWAP_U_OPERANDS = "SWAP_U_OPERANDS"
    SWAP_W_OPERANDS = "SWAP_W_OPERANDS"

    REMOVE_LEFT_SUBFORMULA = "REMOVE_LEFT_SUBFORMULA"
    REMOVE_RIGHT_SUBFORMULA = "REMOVE_RIGHT_SUBFORMULA"

    SWAP_AND_WITH_IMPLIES = "SWAP_AND_WITH_IMPLIES"
    SWAP_AND_WITH_EQUIV = "SWAP_AND_WITH_EQUIV"
    SWAP_OR_WITH_IMPLIES = "SWAP_OR_WITH_IMPLIES"
    SWAP_OR_WITH_EQUIV = "SWAP_OR_WITH_EQUIV"
    SWAP_IMPLIES_WITH_AND = "SWAP_IMPLIES_WITH_AND"
    SWAP_IMPLIES_WITH_OR = "SWAP_IMPLIES_WITH_OR"
    SWAP_EQUIV_WITH_AND = "SWAP_EQUIV_WITH_AND"
    SWAP_EQUIV_WITH_OR = "SWAP_EQUIV_WITH_OR"

    SWAP_AND_WITH_OR = "SWAP_AND_WITH_OR"
    SWAP_OR_WITH_AND = "SWAP_OR_WITH_AND"

    SWAP_IMPLIES_WITH_EQUIV = "SWAP_IMPLIES_WITH_EQUIV"
    SWAP_EQUIV_WITH_IMPLIES = "SWAP_EQUIV_WITH_IMPLIES"

    SWAP_APS = "SWAP_APS"

    ADD_F = "ADD_F"
    ADD_G = "ADD_G"
    ADD_X = "ADD_X"
    ADD_NEGATION = "ADD_NEGATION"


TOP_CONTEXTS = [

(Mutation.ADD_X,1,None,2),
(Mutation.SWAP_G_WITH_F,1,None,2),
(Mutation.ADD_G,4,True,2),
(Mutation.ADD_X,4,True,1),
(Mutation.ADD_F,1,False,1),
(Mutation.ADD_NEGATION,3,None,2),
(Mutation.ADD_X,3,False,3),
(Mutation.ADD_X,1,False,4),
(Mutation.ADD_F,2,False,2),
(Mutation.SWAP_IMPLIES_WITH_EQUIV,2,True,1),
(Mutation.ADD_F,1,False,2),
(Mutation.ADD_F,3,False,2),
(Mutation.REMOVE_LEFT_SUBFORMULA,2,True,1),
(Mutation.ADD_X,3,None,2),
(Mutation.ADD_X,3,None,3),
(Mutation.ADD_G,3,True,2),
(Mutation.ADD_NEGATION,1,False,2),
(Mutation.ADD_NEGATION,1,True,1),
(Mutation.ADD_F,1,True,2),
(Mutation.ADD_X,1,False,3),

]



def check_acceptance(aut, trace):

    d = aut.get_dict()

    word = spot.parse_word(trace, d)
    
    if not aut.intersects(word):
        return False
    
    trace_aut = word.as_automaton()
    
    if spot.contains(aut, trace_aut):
        return True
    
    return None


def collect_aps(f):
    return sorted(str(ap) for ap in spot.atomic_prop_collect(f))


def rec_separate(cond):
    
    if len(cond) < 2:
        return [cond]
    
    separate = [rec_separate(c) for c in cond]
    
    if cond._is(spot.op_Or):
        return [elem for c in separate for elem in c]
    
    if cond._is(spot.op_And):
        return [spot.formula.And(single) for single in itertools.product(*separate)]


def separate(cond):
    return [str(formula) for formula in rec_separate(spot.formula(cond))]


def build_word(conditions, index):
    prefix = '; '.join(conditions[:index])
    cycle = "cycle{{ {} }}".format('; '.join(conditions[index:]))
    return cycle if not prefix else "{}; {}".format(prefix, cycle)



def count_literals(trace: str) -> int:
    lits = re.findall(r'!?[A-Za-z_][A-Za-z0-9_]*', trace)
    return sum(1 for lit in lits if lit.lstrip('!') != 'cycle')



def simulate_user(ground_truth, trace, candidate_acceptance):

    gt_acceptance = check_acceptance(spot.translate(ground_truth), trace)

    # User rejects candidate based on trace
    return gt_acceptance != candidate_acceptance



def get_words_from_conditions(conditions, index):

    words = []
    
    for single_cond in itertools.product(*[separate(cond) for cond in conditions]):

        word = build_word(single_cond, index)
        # word_ptr = spot.parse_word(word)
        # word_ptr.simplify()
        # words.append(str(word_ptr))
        words.append(word)

    return words

def trace_len(trace):
    return str(trace).count(";") + 1


def get_formula_features(formula_str: str):
    f = spot.formula(formula_str)

    ops = [
        ("G", spot.op_G),
        ("F", spot.op_F),
        ("U", spot.op_U),
        ("X", spot.op_X),
        ("Not", spot.op_Not),
        ("And", spot.op_And),
        ("Or", spot.op_Or),
        ("Implies", spot.op_Implies),
        ("Equiv", spot.op_Equiv),
    ]

    counts = {name: 0 for name, _ in ops}
    max_depths = {name: 0 for name, _ in ops}

    def depth(node):
        if node.size() == 0:
            return 0
        return 1 + max(depth(node[i]) for i in range(node.size()))

    def visit(node):
        node_depth = depth(node)

        for name, op in ops:
            if node._is(op):
                counts[name] += 1
                max_depths[name] = max(max_depths[name], node_depth)

        for i in range(node.size()):
            visit(node[i])

    total_depth = depth(f)
    visit(f)

    return [
        total_depth,
        counts["G"],
        counts["F"],
        counts["U"],
        counts["X"],
        counts["Not"],
        counts["And"],
        counts["Or"],
        counts["Implies"],
        counts["Equiv"],
        max_depths["G"],
        max_depths["F"],
        max_depths["U"],
        max_depths["X"],
        max_depths["Not"],
        max_depths["And"],
        max_depths["Or"],
        max_depths["Implies"],
        max_depths["Equiv"],
    ]


def all_contexts(acceptance, mut, length):
    return True

def top1_context(acceptance, mut, length):
    return (mut[0], mut[1], acceptance, length) in TOP_CONTEXTS[-1:]

def top5_context(acceptance, mut, length):
    return (mut[0], mut[1], acceptance, length) in TOP_CONTEXTS[-5:]

def top10_context(acceptance, mut, length):
    return (mut[0], mut[1], acceptance, length) in TOP_CONTEXTS[-10:]

def top20_context(acceptance, mut, length):
    return (mut[0], mut[1], acceptance, length) in TOP_CONTEXTS[-20:]

