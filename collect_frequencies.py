from mutation_based import generate_traces
import sys
import csv
import json
from evaluate import DATASIZE


def collect_frequencies(
    formulas,
    strategy,
):
    by_mutation = {}

    for formula_id, formula in enumerate(formulas):

        # formula = (ground truth, candidate)

        # trace = (trace, accept/reject, mutation type)

        traces = generate_traces(formula[1], strategy)

        for trace in traces:
            if trace[2] not in by_mutation:
                by_mutation[trace[2]] = 0
            by_mutation[trace[2]] += 1

    return by_mutation


dataset = None
for key in DATASIZE.keys():
    if key in sys.argv[1]:
        dataset = key

if dataset is None:
    raise ValueError("Incorrect dataset")


with open(sys.argv[1], "r", encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    formulas = [tuple(row) for row in reader]

freq = collect_frequencies(formulas, sys.argv[2])

freq = dict(sorted(freq.items(), key=lambda item: -item[1]))


with open(f"freq_{dataset}_{sys.argv[2]}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Mutation Context", "Frequency"])  # Optional: write a header
    
    for key, value in freq.items():
        writer.writerow([key, value])