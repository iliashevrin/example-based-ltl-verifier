#!/usr/bin/env python3
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
import itertools

def check_acceptance(aut, trace):

    word = spot.parse_word(trace)
    
    if not aut.intersects(word):
        return False
    
    trace_aut = word.as_automaton()
    
    if spot.contains(aut, trace_aut):
        return True
    
    return None



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



def get_words_from_conditions(conditions, index):

    words = []
    
    for single_cond in itertools.product(*[separate(cond) for cond in conditions]):

        word = build_word(single_cond, index)
        word_ptr = spot.parse_word(word)
        word_ptr.simplify()
        words.append(str(word_ptr))

    return words


def ltl_structure_vector(formula_str: str):
    f = spot.formula(formula_str)

    ops = [
        ("G", spot.op_G),
        ("F", spot.op_F),
        ("U", spot.op_U),
        ("W", spot.op_W),
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
        counts["W"],
        counts["X"],
        counts["Not"],
        counts["And"],
        counts["Or"],
        counts["Implies"],
        counts["Equiv"],
        max_depths["G"],
        max_depths["F"],
        max_depths["U"],
        max_depths["W"],
        max_depths["X"],
        max_depths["Not"],
        max_depths["And"],
        max_depths["Or"],
        max_depths["Implies"],
        max_depths["Equiv"],
    ]