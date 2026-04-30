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
    positive = [p[1] for p in positive]
    negative = [p[1] for p in negative]

    return positive, negative

def automaton_traversal(candidate):

    aut = spot.translate(candidate, 'parity', 'sbacc', 'state-based', 'complete', 'colored', 'deterministic')
    positive1, negative1 = generate_traces_by_traversal(aut, max_visits=1)
    positive2, negative2 = generate_traces_by_traversal(aut, max_visits=2)

    positive = positive1 + positive2
    negative = negative1 + negative2

    return positive, negative



def simulate_user(candidate, ground_truth, get_traces):

    positive, negative = get_traces(candidate)

    # print("all positive")
    # for pos in positive:
    #     print(pos)

    # print("")

    # print("all negative")
    # for neg in negative:
    #     print(neg)

    # print("")

    traces_seen = 0

    for trace in set(positive + negative):
        traces_seen += 1

        user_accepts = check_acceptance(spot.translate(ground_truth), trace)

        if user_accepts is None:
            print(f'{trace} inconclusive, the candidate is not the desired one')
            return traces_seen, True

        elif user_accepts == False and trace in positive:
            print(f'{trace} rejected, the candidate is not the desired one')
            return traces_seen, True

        elif user_accepts == True and trace in negative:
            print(f'{trace} accepted, the candidate is not the desired one')
            return traces_seen, True

        elif user_accepts == False and trace in negative:
            print(f'{trace} rejected, keep going')

        elif user_accepts == True and trace in positive:
            print(f'{trace} accepted, keep going')

    return traces_seen, False


test_list = [

    ('(!a) U (b | G!a)', '!a U b'),
    ('G(a -> F b)', 'G(a -> X b)'),
    ('G(a -> Xa)', 'a -> Xa'),
    ('F(a & XFa)', 'Fa')


]



if __name__ == "__main__":


    if sys.argv[1] == "TRAVERSAL":
        func = automaton_traversal
    elif sys.argv[1] == "MUTATION":
        func = mutation_based
    else:
        raise("Incorrect example generation function")

    seen = 0
    success = 0

    for candidate, ground_truth in test_list:
        traces_seen, rejected = simulate_user(candidate, ground_truth, func)

        seen += traces_seen
        if rejected:
            success += 1


    print(seen / len(test_list))
    print(success / len(test_list))

