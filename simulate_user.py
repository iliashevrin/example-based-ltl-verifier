#!/usr/bin/env python3
import sys
sys.path.insert(0,'/usr/local/lib/python3.10/site-packages/')
import spot
spot.setup()
import itertools
from mutation_based import mutation_random, mutation_by_length
from utils import check_acceptance, get_formula_features, collect_aps, trace_len
from train_ordering import trace_ranking_0, trace_ranking_1

import csv
import random
import statistics
import joblib
import numpy as np

import time


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




def simulate_user(cand, gt, method, restriction):

    start_time = time.perf_counter()
    traces = method(cand, restriction)
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    seen = 0
    length = 0
    acc_values = {"True": 0, "False": 0, "None": 0}

    if not traces:
        return 0, 0, None, execution_time, acc_values


    good_mc = None

    for trace, cand_acc, mc in traces:

        seen += 1
        length += (trace.count(";") + 1)

        acc = check_acceptance(spot.translate(gt), trace)

        acc_values[str(acc)] += 1

        if acc != cand_acc:
            good_mc = mc
            break

    return seen, length/seen, good_mc, execution_time, acc_values




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



def main():

    if sys.argv[1] == "RANDOM":
        method = mutation_random
    elif sys.argv[1] == "BY_LENGTH":
        method = mutation_by_length
    elif sys.argv[1] == "LTLTRUST_0":
        method = trace_ranking_0
    elif sys.argv[1] == "LTLTRUST_1":
        method = trace_ranking_1

    else:
        raise ValueError("Incorrect example generation function")

    restriction = sys.argv[3]

    seen_dt = []
    seen_ndt = []
    lengths = []
    total = 0

    mc_map = {}

    times = []

    acc_values = {"True": 0, "False": 0, "None": 0}


    dataset = None
    for key in DATASIZE.keys():
        if key in sys.argv[2]:
            dataset = key

    if dataset is None:
        raise ValueError("Incorrect dataset")

    with open(sys.argv[2], newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            gt = row["Ground Truth"]
            cand = row["Response"]

            repeats = 1
            if sys.argv[1] == "RANDOM":
                repeats = 30

            curr_seen_dt = []
            curr_seen_ndt = []
            curr_length = []
            total += 1

            for _ in range(0, repeats):

                seen, length, mc, exec_time, values = simulate_user(cand, gt, method, restriction)

                for val in values:
                    acc_values[val] += values[val]

                curr_length.append(length)
                times.append(exec_time)

                # Mismatch was successfully detected, look at the mc of the discriminating trace
                if mc is not None:
                    curr_seen_dt.append(seen)

                    if mc not in mc_map:
                        mc_map[mc] = 1
                    else:
                        mc_map[mc] += 1

                else:
                    curr_seen_ndt.append(seen)


            if len(curr_seen_dt) > 0:
                seen_dt.append(statistics.mean(curr_seen_dt))

            lengths.append(statistics.mean(curr_length))

            if len(curr_seen_ndt) > 0:
                seen_ndt.append(curr_seen_ndt[0])



    det_more_than_i = []
    undet_under_i = []
    confidence = []    
    undet_inf = len(seen_ndt) / total
    accuracy = 1 - (total / DATASIZE[dataset])

    output_file = f"results_{dataset}_{sys.argv[1]}_{restriction}.txt"

    with open(output_file, "w", encoding="utf-8") as out:
        print(f'Average Time: {statistics.mean(times):.3f}', file=out)
        print(f'Accepting traces for GT: {(acc_values["True"] / repeats):.3f}', file=out)
        print(f'Rejecting traces for GT: {(acc_values["False"] / repeats):.3f}', file=out)
        print(f'Inconclusive traces for GT: {(acc_values["None"] / repeats):.3f}', file=out)

        print(f'\nAccuracy: {accuracy:.3f}', file=out)
        print(f'Total Formulas: {total}', file=out)
        print(f'Detected Ratio: {(1-undet_inf):.3f}', file=out)

        for i in range(0, 16):
            det_more_than_i.append(len([n for n in seen_dt if n > i]) / total)
            undet_under_i.append(undet_inf + det_more_than_i[i])
            confidence.append(
                accuracy / (accuracy + ((1 - accuracy) * undet_under_i[i]))
            )

            print(
                f'\nUndetected <={i} Ratio: {undet_under_i[i]:.3f} '
                f'(Undetected Ratio: {undet_inf:.3f} + '
                f'Detected >{i} Ratio: {det_more_than_i[i]:.3f})',
                file=out,
            )
            print(
                f'Confidence score for n={i}: {confidence[i]:.3f}',
                file=out,
            )

        print(f'\nTraces to Detection: {stats(seen_dt)}', file=out)
        # print(f'Inspected Traces When No Detection: {stats(seen_ndt)}', file=out)

        print(f'\nTrace Lengths: {stats(lengths)}', file=out)

        print(f'\nMutation Contexts by Usefulness', file=out)

        mc_map = dict(sorted(mc_map.items(), key=lambda item: item[1]))
        for mc in mc_map:
            print(f'{mc}:{mc_map[mc]}', file=out)

        print(f'\nTraces to Detection List: {[f"{seen:.2f}" for seen in seen_dt]}', file=out)



if __name__ == "__main__":
    main()