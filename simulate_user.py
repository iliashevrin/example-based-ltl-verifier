#!/usr/bin/env python3
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
import itertools
from mutation_based_ltl_verifier import generate_traces_by_mutation 
from traversal_based_ltl_verifier import generate_traces_by_traversal 
from utils import check_acceptance

import csv







def simulate_user(candidate, ground_truth, get_traces):

    positive, negative = get_traces(candidate)

    print(len(positive + negative))

    # print("all positive")
    # for pos in positive:
    #     print(pos)

    # print("")

    # print("all negative")
    # for neg in negative:
    #     print(neg)

    # print("")

    traces_seen = 0

    for trace, props in set(positive + negative):
        traces_seen += 1

        user_accepts = check_acceptance(spot.translate(ground_truth), trace)

        if user_accepts is None:
            # print(f'{trace} inconclusive, the candidate is not the desired one')
            return traces_seen, props

        elif user_accepts == False and trace in positive:
            # print(f'{trace} rejected, the candidate is not the desired one')
            return traces_seen, props

        elif user_accepts == True and trace in negative:
            # print(f'{trace} accepted, the candidate is not the desired one')
            return traces_seen, props

        elif user_accepts == False and trace in negative:
            # print(f'{trace} rejected, keep going')
            continue

        elif user_accepts == True and trace in positive:
            # print(f'{trace} accepted, keep going')
            continue

    return traces_seen, None


test_list = [

    # ('(!a) U (b | G!a)', '!a U b'),
    # ('G(a -> F b)', 'G(a -> X b)'),
    # ('G(a -> Xa)', 'a -> Xa'),
    # ('F(a & XFa)', 'Fa'),
    ('G (x1 -> (F (x2 & F x3)))', 'G(x1->F(x3 & F(x2)))'),

]



if __name__ == "__main__":


    if sys.argv[1] == "TRAVERSAL":
        func = generate_traces_by_traversal
    elif sys.argv[1] == "MUTATION":
        func = generate_traces_by_mutation
    else:
        raise("Incorrect example generation function")

    seen = 0
    success = 0
    total = 0

    props_map = {}


    with open(sys.argv[2], newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ground_truth = row["Ground Truth"]
            candidate = row["Response"]
            total += 1

    # for candidate, ground_truth in test_list:

            print(ground_truth, candidate)

            traces_seen, props = simulate_user(candidate, ground_truth, func)

            seen += traces_seen
            # Incorrect candidate was successfully rejected, look at the props of the discriminating trace
            if props is not None:
                success += 1

                if props not in props_map:
                    props_map[props] = 1
                else:
                    props_map[props] += 1


    print(seen / total)
    print(success / total)

    for prop in props_map:
        print(f'{str(prop)}:{props_map[prop]}')

