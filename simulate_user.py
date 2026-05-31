#!/usr/bin/env python3
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
import itertools
from mutation_based import mutation_random, mutation_gradual, mutation_expert, mutation_by_length, Mutation
from traversal_based import traversal_random, traversal_gradual, traversal_expert, traversal_by_length
from utils import check_acceptance, get_formula_features, collect_aps, trace_len
from train_ordering import trace_ranking

import csv
import random
import statistics
import joblib
import numpy as np


DATASIZE = {
    
    "textbook":370,
    "ARTEMIS":125,
    "spacewire":35,
    "Dwyer":100,

    "ALL_RL":630,

    "ConformalLTL":678,
    "Synthetic":1000,

    "heldout": 2548 # Approximation based on the full dataset
}


def unified_by_length(formula):

    traces = mutation_gradual(formula)
    traces.extend(traversal_gradual(formula))

    traces.sort(key=lambda trace: str(trace).count(";"))

    return traces


def unified_random(formula):

    traces = mutation_gradual(formula)
    traces.extend(traversal_gradual(formula))
    random.shuffle(traces)
    return traces




def simulate_user_iteration(candidate, ground_truth, method, fltr):

    traces = method(candidate, fltr)

    traces_seen = 0
    trace_length = 0

    # for trace, is_positive, props in traces:
    #     print(is_positive, props)
    # print('-----')

    if not traces:
        return 0, 0, None


    distinguish_props = None

    for trace, candidate_acceptance, props in traces:

        traces_seen += 1
        trace_length += (trace.count(";") + 1)

        gt_acceptance = check_acceptance(spot.translate(ground_truth), trace)

        if gt_acceptance != candidate_acceptance:
            distinguish_props = props
            break


    return traces_seen, trace_length/traces_seen, distinguish_props


test_list = [

    # ('(!a) U (b | G!a)', '!a U b'),
    # ('G(a -> F b)', 'G(a -> X b)'),
    # ('G(a -> Xa)', 'a -> Xa'),
    # ('F(a & XFa)', 'Fa'),
    ('G (x1 -> (F (x2 & F x3)))', 'G(x1->F(x3 & F(x2)))'),

]


def stats(data):

    if len(data) <= 1:
        return f"N/A"

    avg_val = statistics.mean(data)      # Average (Arithmetic Mean)
    med_val = statistics.median(data)    # Median
    std_val = statistics.stdev(data)     # Standard Deviation (sample)
    max_val = max(data)                  # Built-in max function
    min_val = min(data)                  # Built-in min function

    # Percentiles
    p75 = np.percentile(data, 75)
    p90 = np.percentile(data, 90)

    return (
        f"Avg: {avg_val:.3f}, "
        f"Median: {med_val:.2f}, "
        f"Std: {std_val:.2f}, "
        f"P75: {p75:.2f}, "
        f"P90: {p90:.2f}, "
        f"Max: {max_val:.2f}, "
        f"Min: {min_val:.2f}"
    )



def confidence(undetected_under_n, total, dataset):

    mismatch = total / DATASIZE[dataset]
    match = (1 - mismatch)

    return match / (match + (mismatch * (undetected_under_n / total)))


if __name__ == "__main__":


    if sys.argv[1] == "RANDOM":
        generation_method = mutation_random
    elif sys.argv[1] == "BY_LENGTH":
        generation_method = mutation_by_length
    elif sys.argv[1] == "DYNAMIC":
        generation_method = trace_ranking

    else:
        raise ValueError("Incorrect example generation function")

    if len(sys.argv) >= 4:
        fltr = sys.argv[3]
    else:
        fltr = "no_filter"

    seen_in_success = []
    seen_in_failure = []
    trace_lengths = []
    total = 0

    props_map = {}
    rows = []
    success_map = {}


    dataset = None
    for key in DATASIZE.keys():
        if key in sys.argv[2]:
            dataset = key

    if dataset is None:
        raise ValueError("Incorrect dataset")

    with open(sys.argv[2], newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            ground_truth = row["Ground Truth"]
            candidate = row["Response"]

            repeats = 1
            if sys.argv[1] == "RANDOM":
                repeats = 30

            for _ in range(0,repeats):

                total += 1
                traces_seen, trace_length, props = simulate_user_iteration(candidate, ground_truth, generation_method, fltr)

                trace_lengths.append(trace_length)

                # Incorrect candidate was successfully rejected, look at the props of the discriminating trace
                if props is not None:
                    # success += 1
                    seen_in_success.append(traces_seen)

                    if traces_seen in success_map:
                        success_map[traces_seen] += 1
                    else:
                        success_map[traces_seen] = 1

                    if props not in props_map:
                        props_map[props] = 1
                    else:
                        props_map[props] += 1

                else:
                    seen_in_failure.append(traces_seen)

                    # Missed detections
                    rows.append(
                        {
                            "Ground Truth": ground_truth,
                            "Candidate": candidate,
                        }
                    )



    det_more_than_i = []
    undet_under_i = []
    confidence = []    
    undet_inf = len(seen_in_failure) / total
    accuracy = 1 - ((total / repeats) / DATASIZE[dataset])

    print(f'Accuracy: {accuracy:.3f}')
    print(f'Total Formulas: {total / repeats}')
    print(f'Detected Ratio: {(1-undet_inf):.3f}')

    for i in range(0,16):

        det_more_than_i.append(len([n for n in seen_in_success if n > i]) / total)
        undet_under_i.append(undet_inf + det_more_than_i[i])
        confidence.append(accuracy / (accuracy + ((1 - accuracy) * undet_under_i[i])))

        print(f'Undetected <={i} Ratio: {undet_under_i[i]:.3f} (Undetected Ratio: {undet_inf:.3f} + Detected >{i} Ratio: {det_more_than_i[i]:.3f})')
        print(f'Confidence score for n={i}: {confidence[i]:.3f}')

    print(f'Trace Numbers: {stats(seen_in_success)}')
    # print(f'Inspected Traces When No Detection: {stats(seen_in_failure)}')

    print(f'Trace Lengths: {stats(trace_lengths)}')

    props_map = dict(sorted(props_map.items(), key=lambda item: item[1]))
    for prop in props_map:
        print(f'{str(prop)}:{props_map[prop]}')

    # cummulative = 0
    # for i in range(1,max(success_map.keys())):
    #     if i not in success_map:
    #         continue
    #     cummulative += success_map[i]
    #     print(f'{str(i)}:{(cummulative / success):.3f}')

