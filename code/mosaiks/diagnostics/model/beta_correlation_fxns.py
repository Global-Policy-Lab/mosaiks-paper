import mosaiks.solve.solve_functions as solver
import numpy as np
from mosaiks import config as c
from mosaiks import transforms
from mosaiks.solve.interpret_results import interpret_kfold_results
from sklearn.model_selection import KFold


def kfold_solve_no_overlap(
    X,
    y,
    solve_function=solver.ridge_regression,
    num_folds=5,
    return_preds=True,
    return_model=False,
    **kwargs_solve
):

    assert num_folds > 1

    y = solver.y_to_matrix(y)
    n_outcomes = y.shape[1]
    # keep track of all runs over several iterations
    kfold_metrics_test = []
    kfold_metrics_train = []
    kfold_preds_test = []
    kfold_preds_train = []
    kfold_y_train = []
    kfold_y_test = []
    kfold_models = []
    hp_warnings = []

    print("on fold (of {0}): ".format(num_folds), end="")

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    split = kf.split(X)

    kf_split_idxs = []
    for i, (_, val_idxs_i) in enumerate(split):
        kf_split_idxs.append(val_idxs_i)
    val_idxs = kf_split_idxs[0]
    train_splits = kf_split_idxs[1:]

    i = 0
    for train_idxs in train_splits:
        i += 1
        print("{0} ".format(i), end="")

        X_train, X_val = X[train_idxs], X[val_idxs]
        y_train, y_val = y[train_idxs], y[val_idxs]

        # record train/test obs for this split
        this_y_train = np.empty(n_outcomes, dtype=np.ndarray)
        this_y_val = np.empty(n_outcomes, dtype=np.ndarray)
        for o in range(n_outcomes):
            this_y_train[o] = y_train[:, o]
            this_y_val[o] = y_val[:, o]
        kfold_y_train.append(this_y_train)
        kfold_y_test.append(this_y_val)

        # call solve func
        solve_results = solve_function(
            X_train,
            X_val,
            y_train,
            y_val,
            return_preds=return_preds,
            return_model=return_model,
            **kwargs_solve
        )

        # record performance metrics
        kfold_metrics_test.append(solve_results["metrics_test"])
        kfold_metrics_train.append(solve_results["metrics_train"])

        # record optional preds and model parameters
        if return_preds:
            kfold_preds_test.append(solve_results["y_pred_test"])
            kfold_preds_train.append(solve_results["y_pred_train"])
        if return_model:
            kfold_models.append(solve_results["models"])

        # recpord np warnings
        hp_warnings.append(solve_results["hp_warning"])

    # Return results
    rets = {
        "metrics_test": np.array(kfold_metrics_test),
        "metrics_train": np.array(kfold_metrics_train),
        "y_true_test": np.array(kfold_y_test),
        "y_true_train": np.array(kfold_y_train),
        "hp_warning": np.array(hp_warnings),
        "cv": kf,
    }

    if return_preds:
        rets["y_pred_test"] = np.array(kfold_preds_test)
        rets["y_pred_train"] = np.array(kfold_preds_train)

    if return_model:
        rets["models"] = np.array(kfold_models)

    return rets


def rets_to_weights(rets_this):
    best_idxs, metrics_best_idx, y_pred_best_idx = interpret_kfold_results(
        rets_this, "r2_score"
    )
    weights = rets_this["models"]
    best_idx = best_idxs[0][0]
    weights_this = weights[:, 0, best_idx]
    return weights_this, metrics_best_idx


def compute_beta_correlations(
    task_name_a,
    task_name_b,
    task_idx_a,
    task_idx_b,
    y,
    X,
    d=1024,
):
    kwargs_solve = {"lambdas": np.logspace(-4, 8, base=10, num=5)}

    # drop any invalid values
    _, valid_a = transforms.dropna_Y(y[:, task_idx_a], task_name_a)
    _, valid_b = transforms.dropna_Y(y[:, task_idx_b], task_name_b)
    valid = valid_a & valid_b
    X, y = X[valid], y[valid]

    rets_a = kfold_solve_no_overlap(
        X[:, :d],
        y[:, task_idx_a],
        num_folds=c.ml_model["n_folds"],
        return_preds=True,
        return_model=True,
        **kwargs_solve
    )
    weights_a, metrics_a = rets_to_weights(rets_a)

    rets_b = kfold_solve_no_overlap(
        X[:, :d],
        y[:, task_idx_b],
        num_folds=c.ml_model["n_folds"],
        return_preds=True,
        return_model=True,
        **kwargs_solve
    )
    weights_b, metrics_b = rets_to_weights(rets_b)

    return weights_a, weights_b
