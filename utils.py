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