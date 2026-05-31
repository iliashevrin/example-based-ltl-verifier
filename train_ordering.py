from collections import defaultdict
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

from utils import get_formula_features, simulate_user, trace_len, Mutation
from mutation_based import mutation_gradual

from utils import top1_context, top1_mut, top5_mut, only_acc, only_rej, acc_and_rej, only_shallow, only_deep, no_filter

import joblib
import argparse
import csv


def generate_traces(formula, fltr):

    traces = mutation_gradual(formula, fltr)
    traces = [(trace, acceptance, f'{str(mut[0])}_{mut[1]}_{acceptance}') for trace, acceptance, mut in traces]
    return traces


# ==================================================
# BAYESIAN-SMOOTHED USEFULNESS RATIO
# ==================================================

def smoothed_ratio(
    num_helpful,
    total,
    alpha_prior=1.0,
    beta_prior=1.0,
):
    """
    Beta-binomial smoothing.

    Example:
        alpha=1, beta=1
        -> Laplace smoothing
    """

    return (
        num_helpful + alpha_prior
    ) / (
        total + alpha_prior + beta_prior
    )


# ==================================================
# Reciprocal mutation utility
# ==================================================

def reciprocal_utility(labels, alpha=1.0):
    """
    labels[i] is 1 if the i-th trace is useful, else 0.
    Traces are assumed to already be sorted shortest-first.

    Utility:
        sum_i y_i / i^alpha

    With alpha=1:
        y1 / 1 + y2 / 2 + y3 / 3 + ...
    """
    utility = 0.0

    for i, y in enumerate(labels, start=1):
        utility += y / (i ** alpha)

    return utility



# ==================================================
# BUILD TRAINING TABLE
# ==================================================

def build_training_data(
    formulas,
    fltr,
    alpha_prior=1.0,
    beta_prior=1.0,
):
    rows = []

    for formula_id, formula in enumerate(formulas):

        # formula = (ground truth, candidate)

        # trace = (trace, accept/reject, mutation type)

        try:
            # --------------------------------------
            # Formula-level processing
            # --------------------------------------

            formula_features = get_formula_features(formula[1])

            traces = generate_traces(formula[1], fltr)

        except Exception as e:
            print(
                f"[WARNING] Skipping formula #{formula_id} {formula} "
                f"due to preprocessing error:\n{e}"
            )
            continue

        by_mutation = defaultdict(list)

        for trace in traces:
            by_mutation[trace[2]].append(trace)

        for mutation, mutation_traces in by_mutation.items():

            # --------------------------------------
            # shortest traces first INSIDE mutation
            # --------------------------------------

            mutation_traces.sort(
                key=lambda t: trace_len(t[0])
            )


            labels = [
                int(simulate_user(formula[0], trace[0], trace[1]))
                for trace in mutation_traces
            ]

            helpful_count = sum(labels)
            total_count = len(labels)

            usefulness = smoothed_ratio(
                helpful_count,
                total_count,
                alpha_prior=alpha_prior,
                beta_prior=beta_prior,
            )

            # usefulness = reciprocal_utility(labels, alpha=1.0)

            row = {
                "formula_id": formula_id,
                "mutation": mutation,
                "num_traces": total_count,
                "helpful_count": helpful_count,
                "target": usefulness,
            }

            for i, value in enumerate(formula_features):
                row[f"f{i}"] = value

            rows.append(row)

    return pd.DataFrame(rows)


# ==================================================
# TRAIN MODEL
# ==================================================

def train_model(
    formulas,
    fltr,
    alpha_prior=1.0,
    beta_prior=1.0,
    test_size=0.2,
    random_state=42,
):
    df = build_training_data(
        formulas,
        fltr,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
    )

    # ----------------------------------------------
    # one-hot mutation enum
    # ----------------------------------------------

    df_model = pd.get_dummies(
        df,
        columns=["mutation"],
        dummy_na=False,
    )

    y = df_model["target"]

    drop_cols = [
        "target",
        "formula_id",
        "helpful_count",
    ]

    X = df_model.drop(columns=drop_cols)

    groups = df_model["formula_id"]

    # ----------------------------------------------
    # split BY FORMULA
    # ----------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, valid_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_valid = X.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    # ----------------------------------------------
    # regression model
    # ----------------------------------------------

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        max_depth=-1,
        random_state=random_state,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
    )

    preds = model.predict(X_valid)

    rmse = np.sqrt(
        mean_squared_error(
            y_valid,
            preds,
        )
    )

    print(f"Validation RMSE: {rmse:.4f}")

    feature_columns = X.columns.tolist()


    # --------------------------------------------------
    # SAVE HELD-OUT TEST FORMULAS
    # --------------------------------------------------

    valid_formula_ids = sorted(
        set(df.iloc[valid_idx]["formula_id"])
    )

    test_formulas_df = pd.DataFrame({
        "Ground Truth": [
            formulas[i][0]
            for i in valid_formula_ids
        ],
        "Response": [
            formulas[i][1]
            for i in valid_formula_ids
        ]
    })

    test_formulas_df.to_csv(
        f"heldout_test_formulas_{fltr}.csv",
        index=False,
    )

    print(
        f"Saved {len(test_formulas_df)} held-out formulas "
        f"to heldout_test_formulas_{fltr}.csv"
    )

    return model, feature_columns, df


# ==================================================
# PREDICTION HELPERS
# ==================================================

def make_prediction_rows(
    formula,
    by_mutation,
):
    formula_features = get_formula_features(formula)

    rows = []
    mutations = []

    for mutation, traces in by_mutation.items():

        row = {
            "mutation": mutation,
            "num_traces": len(traces),
        }

        for i, value in enumerate(formula_features):
            row[f"f{i}"] = value

        rows.append(row)
        mutations.append(mutation)

    return rows, mutations


def align_features(
    rows,
    feature_columns,
):
    X = pd.DataFrame(rows)

    X = pd.get_dummies(
        X,
        columns=["mutation"],
        dummy_na=False,
    )

    X = X.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    return X


# ==================================================
# DIVERSIFIED SCHEDULING
# ==================================================

def diversified_trace_ranking(
    formula,
    fltr,
    model,
    feature_columns,
    diversification_alpha=1,
):
    """
    diversification_alpha controls
    diversification strength.

    alpha = 0:
        no diversification

    alpha = 1:
        reciprocal decay

    Suggested:
        0.2 - 0.5
    """

    traces = generate_traces(formula, fltr)

    if not traces:
        return traces, None

    by_mutation = defaultdict(list)

    for trace in traces:
        by_mutation[trace[2]].append(trace)

    # ----------------------------------------------
    # shortest traces first INSIDE mutation
    # ----------------------------------------------

    for mutation in by_mutation:
        by_mutation[mutation].sort(
            key=lambda t: trace_len(t[0])
        )

    # ----------------------------------------------
    # predict base usefulness per mutation
    # ----------------------------------------------

    rows, mutations = make_prediction_rows(
        formula,
        by_mutation,
    )

    X_pred = align_features(
        rows,
        feature_columns,
    )

    scores = model.predict(X_pred)

    base_scores = dict(zip(mutations, scores))

    # ----------------------------------------------
    # dynamic diversification state
    # ----------------------------------------------

    shown_count = {
        mutation: 0
        for mutation in mutations
    }

    ranked_traces = []

    total_traces = sum(
        len(v)
        for v in by_mutation.values()
    )

    # ----------------------------------------------
    # greedy diversified scheduling
    # ----------------------------------------------

    for _ in range(total_traces):

        best_mutation = None
        best_effective_score = -float("inf")

        for mutation in mutations:

            k = shown_count[mutation]

            # no remaining traces
            if k >= len(by_mutation[mutation]):
                continue

            # --------------------------------------
            # SOFT diversification
            # --------------------------------------

            effective_score = (
                base_scores[mutation]
                / ((k + 1) ** diversification_alpha)
            )

            if effective_score > best_effective_score:
                best_effective_score = effective_score
                best_mutation = mutation

        if best_mutation is None:
            break

        k = shown_count[best_mutation]

        next_trace = by_mutation[best_mutation][k]

        ranked_traces.append(next_trace)

        shown_count[best_mutation] += 1

    return ranked_traces, base_scores




def trace_ranking(formula, fltr):

    model = joblib.load(f"mutation_ranker_{fltr}.pkl")
    feature_columns = joblib.load(f"feature_columns_{fltr}.pkl")

    ranked_traces, mutation_scores = diversified_trace_ranking(
        formula=formula,
        fltr=fltr,
        model=model,
        feature_columns=feature_columns,
        diversification_alpha=1,
    )

    return ranked_traces


def main():
    parser = argparse.ArgumentParser(
        description="Train model and provide mutation ranking."
    )

    parser.add_argument(
        "data_file",
        help="Path to data CSV file"
    )

    parser.add_argument("filter", default="no_filter")

    args = parser.parse_args()


    # Read all rows
    with open(args.data_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        train_formulas = [tuple(row) for row in reader]


    model, feature_columns, _ = train_model(
        formulas=train_formulas,
        fltr=args.filter
    )

    joblib.dump(model, f"mutation_ranker_{args.filter}.pkl")
    joblib.dump(feature_columns, f"feature_columns_{args.filter}.pkl")


if __name__ == "__main__":
    main() 