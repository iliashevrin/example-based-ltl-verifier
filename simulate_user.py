#!/usr/bin/env python3
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
import itertools
from mutation_based_ltl_verifier import generate_traces_by_mutation 
from traversal_based_ltl_verifier import generate_traces_by_traversal 
from utils import check_acceptance



def mutation_based(candidate):

    positive, negative = generate_traces_by_mutation(candidate)
    positive = set([p[1] for p in positive])
    negative = set([p[1] for p in negative])

    return positive, negative

def automaton_traversal(candidate):

    aut = spot.translate(candidate, 'parity', 'sbacc', 'state-based', 'complete', 'colored', 'deterministic')
    positive1, negative1 = generate_traces_by_traversal(aut, max_visits=1)
    positive2, negative2 = generate_traces_by_traversal(aut, max_visits=2)

    positive = set(positive1 + positive2)
    negative = set(negative1 + negative2)

    return positive, negative



def simulate_user(candidate, ground_truth, get_traces):

    positive, negative = get_traces(candidate)

    print("all positive")
    for pos in positive:
        print(pos)

    print("")

    print("all negative")
    for neg in negative:
        print(neg)

    print("")

    for pos in positive:
        intersects = check_acceptance(spot.translate(ground_truth), pos)
        if intersects is None:
            print(f'{pos} inconclusive, the candidate is not the desired one')
            return
        elif intersects == False:
            print(f'{pos} rejected, the candidate is not the desired one')
            return
        else:
            print(f'{pos} accepted, keep going')

    for neg in negative:
        intersects = check_acceptance(spot.translate(ground_truth), neg)
        if intersects is None:
            print(f'{pos} inconclusive, the candidate is not the desired one')
            return
        elif intersects == True:
            print(f'{neg} accepted, the candidate is not the desired one')
            return
        else:
            print(f'{neg} rejected, keep going')



if __name__ == "__main__":


    # sys.argv[1] = candidate output of an LLM
    # sys.argv[2] = what we really want, i.e., ground truth
    # sys.argv[3] = get examples function [AUTOMATON, MUTATION]
    if sys.argv[3] == "AUTOMATON":
        func = automaton_traversal
    elif sys.argv[3] == "MUTATION":
        func = mutation_based
    else:
        raise("Incorrect example generation function")
    simulate_user(sys.argv[1], sys.argv[2], func)