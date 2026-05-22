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

    "heldout": 399 # Approximation based on the full dataset
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




def simulate_user_iteration(candidate, ground_truth, method):

    traces = method(candidate)

    traces_seen = 0
    trace_length = 0

    # for trace, is_positive, props in traces:
    #     print(is_positive, props)
    # print('-----')


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


    if sys.argv[1] == "TRAVERSAL_RANDOM":
        generation_method = traversal_random
    elif sys.argv[1] == "TRAVERSAL_BY_LENGTH":
        generation_method = traversal_by_length
    # elif sys.argv[1] == "TRAVERSAL_EXPERT":
    #     generation_method = traversal_expert
    elif sys.argv[1] == "RANDOM":
        generation_method = mutation_random
    elif sys.argv[1] == "BY_LENGTH":
        generation_method = mutation_by_length
    # elif sys.argv[1] == "EXPERT":
    #     generation_method = mutation_expert
    # elif sys.argv[1] == "MUTATION_INTERLEAVED":
    #     generation_method = mutation_interleaved
    # elif sys.argv[1] == "UNIFIED_RANDOM":
    #     generation_method = unified_random
    # elif sys.argv[1] == "UNIFIED_BY_LENGTH":
    #     generation_method = unified_by_length
    # elif sys.argv[1] == "UNIFIED_EXPERT":
    #     generation_method = unified_expert
    elif sys.argv[1] == "DYNAMIC":
        generation_method = trace_ranking

    else:
        raise ValueError("Incorrect example generation function")

    seen_in_success = []
    seen_in_failure = []
    trace_lengths = []
    success = 0
    total = 0

    props_map = {}

    rows = []

    success_map = {}

    undetected_5_or_under = 0
    undetected_10_or_under = 0

    detected_over_5 = 0
    detected_over_10 = 0

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
                repeats = 10

            for _ in range(0,repeats):

                total += 1
                traces_seen, trace_length, props = simulate_user_iteration(candidate, ground_truth, generation_method)

                trace_lengths.append(trace_length)

                # Incorrect candidate was successfully rejected, look at the props of the discriminating trace
                if props is not None:
                    success += 1
                    seen_in_success.append(traces_seen)

                    if traces_seen in success_map:
                        success_map[traces_seen] += 1
                    else:
                        success_map[traces_seen] = 1

                    if props not in props_map:
                        props_map[props] = 1
                    else:
                        props_map[props] += 1

                    if traces_seen > 5:
                        undetected_5_or_under += 1
                        detected_over_5 += 1
                    if traces_seen > 10:
                        undetected_10_or_under += 1
                        detected_over_10 += 1

                else:
                    seen_in_failure.append(traces_seen)
                    # print(f'Ground Truth: {ground_truth}; Candidate: {candidate}')

                    # Missed detections
                    rows.append(
                        {
                            "Ground Truth": ground_truth,
                            "Candidate": candidate,
                        }
                    )

                    undetected_5_or_under += 1
                    undetected_10_or_under += 1


    print(f'Total Formulas: {total / repeats}')
    print(f'Total Detected: {success / repeats}')
    print(f'Detected Ratio: {(success / total):.3f}')

    print(f'Undetected <=5 Ratio: {(undetected_5_or_under / total):.3f} (Undetected Ratio: {1 - (success / total):.3f} + Detected >5 Ratio: {(detected_over_5 / total):.3f})')
    print(f'Undetected <=10 Ratio: {(undetected_10_or_under / total):.3f} (Undetected Ratio: {1 - (success / total):.3f} + Detected >10 Ratio: {(detected_over_10 / total):.3f})')

    print(f'Confidence score for n=5: {confidence(undetected_5_or_under / repeats, total / repeats, dataset):.3f}')
    print(f'Confidence score for n=10: {confidence(undetected_10_or_under / repeats, total / repeats, dataset):.3f}')

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



    if len(sys.argv) >= 4:
        with open(sys.argv[3], "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["Ground Truth", "Candidate"],
            )
            writer.writeheader()
            writer.writerows(rows)

