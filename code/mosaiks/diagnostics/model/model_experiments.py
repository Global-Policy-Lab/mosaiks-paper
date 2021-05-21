import warnings

import numpy as np
from mosaiks.solve import interpret_results
from mosaiks.solve import solve_functions as solve
from sklearn.model_selection import KFold


def performance_by_num_features(
    X,
    y,
    num_features,
    num_folds=5,
    solve_function=solve.ridge_regression,
    crit="r2_score",
    **solve_kwargs
):
    """
    Slices features into smaller subsets of featurization (by index), and reports
    performance of 5 folds on different feature dimensions d_i < d = X.shape[1]. If you want it
    done randomly, shuffle columns of X before inputing to the function.
    args:
        X: n x d array-like, feature representation
        y: n x 1 array-like, labels
        num_features: list of ints, num_features over which to collect performance results
        num_folds: int, default 5, number of cross validation folds
        solve_function: which function to use for the solve, default ridge regression.
        crit (str): citeria for which to optimize hps
        **solve_kwargs (**dict): dictionary of params for solve fxn
    returns:
        kfold_l_idxs_by_num_feats:list of ints, the best-agreed (across k-folds) lambda index swept over, by feature size d_i
        kfold_test_metrics_by_num_feats: 2d array of dicts, axis=0 corresponds to number of features, axis 1 to fold.
        \fold_test_predictions_by_num_feats: list of list of arrays, test set predictions results from
        each of k k-fold models, where lambda is  set according to l_idxs_by_num_feat for each train set size,
        uniformly across folds.
    """
    solve_kwargs["return_preds"] = True

    assert np.max(num_features) <= X.shape[1], "not enough features to satisfy"
    results_by_num_feat = []

    kfold_test_metrics_by_num_feats = []
    kfold_l_idxs_by_num_feats = []
    kfold_test_predictions_by_num_feats = []

    for i, num_feats in enumerate(num_features):
        res = solve.kfold_solve(
            X[:, :num_feats],
            y,
            num_folds=num_folds,
            solve_function=solve_function,
            **solve_kwargs
        )
        results_by_num_feat.append(res)

        (
            best_idxs,
            metrics_best_idx,
            y_pred_best_idx,
        ) = interpret_results.interpret_kfold_results(res, crit)
        kfold_test_metrics_by_num_feats.append(metrics_best_idx)
        kfold_l_idxs_by_num_feats.append(best_idxs)
        kfold_test_predictions_by_num_feats.append(y_pred_best_idx)

    return (
        np.array(kfold_l_idxs_by_num_feats),
        np.array(kfold_test_metrics_by_num_feats),
        np.array(kfold_test_predictions_by_num_feats),
    )


def performance_by_num_train_samples(
    X,
    y,
    num_samples,
    num_folds=5,
    solve_function=solve.ridge_regression,
    crit="r2_score",
    **solve_kwargs
):
    """
    Slices features into smaller subsets of training set (randomization taken care of by Kfold), and reports
    performance of 5 folds on different train set sizes s_i < s = X.shape[0]*(num_folds-1)/num_folds.
    If you rows pulled randomly, shuffle rows of X before inputing to the function.
    args:
        X: n x d array-like, feature representation
        y: n x 1 array-like, labels
        num_samples: list of ints, train set sizes over which to collect performance results
        num_folds: int, default 5, number of cross validation folds
        solve_function: which function to use for the solve, default ridge regression.
        crit (str): citeria for which to optimize hps
        **solve_kwargs (**dict): dictionary of params for solve fxn
    returns:
        l_idxs_by_num_sample: list of ints, the best-agreed (across k-folds) lambda index swept over,
            by train set size
        fold_test_metrics_by_num_samples: list of dicts, results of each of k k-fold models, where lambda is
            set according to l_idxs_by_num_feat for each train set size, uniformly across folds.
            organized in order num_sample
        fold_test_predictions_by_num_samples: list of arrays, test set predictions results from
        each of k k-fold models, where lambda is  set according to l_idxs_by_num_feat for each train set size,
        uniformly across folds.
        num_samples_taken: the number of samples actually taken for each model.
    """

    solve_kwargs["return_preds"] = True

    if np.max(num_samples) > int(X.shape[0] * (num_folds - 1) / num_folds):
        warnings.warn(
            "not enough training points to satisfy {0} samples; ".format(
                np.max(num_samples)
            )
            + "we will use the maximum number available for the last ones which is {0}".format(
                int(X.shape[0] * (num_folds - 1) / num_folds)
            )
        )

    test_metrics_by_num_samples = []
    l_idxs_by_num_samples = []
    test_predictions_by_num_samples = []
    print(" on run (of {0}):".format(len(num_samples)), end=" ")

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

    num_samples_taken = []
    for i, num_samp in enumerate(num_samples):
        print(i + 1, end=" ")
        results = []
        # take out the val set before sub-indexing. Because we fixed the random state of KFold, we will
        # get the same val_idxs for each fold every time.
        for train_idxs, val_idxs in kf.split(X):
            X_train = X[train_idxs]
            y_train = y[train_idxs]
            X_val = X[val_idxs]
            y_val = y[val_idxs]
            # now do results by number of samples.
            results_by_fold = solve.single_solve(
                X_train[:num_samp, :],
                X_val,
                y_train[:num_samp],
                y_val,
                solve_function=solve_function,
                **solve_kwargs
            )
            results.append(results_by_fold)

        # record number of samples actually taken (only record for the last fold, should not differ between folds by
        # more than one).
        num_samples_taken.append(X_train[:num_samp, :].shape[0])

        # compile results as they should be
        results_compiled = {}
        for key in results[0].keys():
            # index everything by zero to avoid having an extra index when we send to interpret_results
            results_compiled[key] = np.array(
                [results[f][key][0] for f in range(num_folds)]
            )

        # results should be packed as if they were all just in a single fold
        (
            best_idxs,
            metrics_best_idx,
            y_pred_best_idx,
        ) = interpret_results.interpret_kfold_results(results_compiled, crit)
        test_metrics_by_num_samples.append(metrics_best_idx)
        l_idxs_by_num_samples.append(best_idxs)
        test_predictions_by_num_samples.append(y_pred_best_idx)

    return (
        np.array(l_idxs_by_num_samples),
        test_metrics_by_num_samples,
        test_predictions_by_num_samples,
        num_samples_taken,
    )
