from collections import defaultdict
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

from utils import get_formula_features, simulate_user, trace_len, Mutation
from mutation_based import generate_traces

import joblib
import argparse
import csv



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
    strategy,
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

            traces = generate_traces(formula[1], strategy)

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

            # mutation_traces.sort(
            #     key=lambda t: trace_len(t[0])
            # )


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
    strategy,
    alpha_prior=1.0,
    beta_prior=1.0,
    test_size=0.2,
    random_state=42,
):
    df = build_training_data(
        formulas,
        strategy,
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
        f"heldout_test_formulas_{strategy}.csv",
        index=False,
    )

    print(
        f"Saved {len(test_formulas_df)} held-out formulas "
        f"to heldout_test_formulas_{strategy}.csv"
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
    traces,
    model,
    feature_columns,
    diversification_alpha,
):
    """
    diversification_alpha controls
    diversification strength.

    alpha = 0:
        no diversification

    alpha = 1:
        reciprocal decay
    """

    by_mutation = defaultdict(list)

    for trace in traces:
        by_mutation[trace[2]].append(trace)

    # ----------------------------------------------
    # fixed lexicographic sort INSIDE mutation
    # ----------------------------------------------

    # Lexicographic sort to fix some ordering within the same mutation context when all traces are of the same length
    for mutation in by_mutation:
        by_mutation[mutation].sort()
            # key=lambda t: trace_len(t[0]))


    # if formula == "G(((x1 & X(x2)) -> (x3 & X(x4))) & ((x3 & X(x4)) -> (x1 & X(x2))))":
    #     for mutation in by_mutation:
    #         for trace in by_mutation[mutation]:
    #             print(f"mutation={mutation}; trace={trace}")

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

        for mutation in sorted(mutations):

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

        # if formula == "(F(x2 & x1)) -> (x1 U (x2 | X(x2)))":
        #     print(f"###{len(ranked_traces)} k={k}; best_effective_score={best_effective_score}; best_mutation={best_mutation}; next_trace={next_trace}")

        ranked_traces.append(next_trace)

        shown_count[best_mutation] += 1


    return ranked_traces, base_scores




def trace_ranking(formula, traces, strategy, alpha):

    model = joblib.load(f"mutation_ranker_{strategy}.pkl")
    feature_columns = joblib.load(f"feature_columns_{strategy}.pkl")

    ranked_traces, mutation_scores = diversified_trace_ranking(
        formula=formula,
        traces=traces,
        model=model,
        feature_columns=feature_columns,
        diversification_alpha=alpha,
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

    parser.add_argument("strategy", default="no_strategy")

    args = parser.parse_args()


    # Read all rows
    with open(args.data_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        train_formulas = [tuple(row) for row in reader]


    model, feature_columns, _ = train_model(
        formulas=train_formulas,
        strategy=args.strategy
    )

    joblib.dump(model, f"mutation_ranker_{args.strategy}.pkl")
    joblib.dump(feature_columns, f"feature_columns_{args.strategy}.pkl")


if __name__ == "__main__":
    main() 