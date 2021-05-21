import time
import warnings

import numpy as np
import sklearn.metrics as metrics
from mosaiks.solve import data_parser as parse
from mosaiks.utils.io import GPU
from scipy.linalg.misc import LinAlgWarning
from sklearn.linear_model._base import _preprocess_data
from sklearn.model_selection import KFold

DEBUG = False

if GPU:
    import cupy as xp
    from cupy import linalg

    linalg_solve_kwargs = {}
    asnumpy = xp.asnumpy
    mempool = xp.get_default_memory_pool()
    pinned_mempool = xp.get_default_pinned_memory_pool()
else:
    from scipy import linalg

    linalg_solve_kwargs = {"sym_pos": True}
    xp = np
    asnumpy = np.asarray


def ridge_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    svd_solve=False,
    lambdas=[1e2],
    return_preds=True,
    return_model=False,
    clip_bounds=None,
    intercept=False,
    allow_linalg_warning_instances=False,
):
    """Train ridge regression model for a series of regularization parameters.
    Optionally clip the predictions to bounds. Used as the default solve_function
    argument for single_solve() and kfold_solve() below.

    Parameters
    ----------
        X_{train,test} : :class:`numpy.ndarray`
            Features for training/test data (n_obs_{train,test} X n_ftrs 2darray).
        y_{train,test} : :class:`numpy.ndarray`
            Labels for training/test data (n_obs_{train,test} X n_outcomes 2darray).
        svd_solve : bool, optional
            If true, uses SVD to compute w^*, otherwise does matrix inverse for each
            lambda.
        lambdas : list of floats, optional
            Regularization values to sweep over.
        return_preds : bool, optional
            Whether to return predictions for training and test sets.
        return_model : bool, optional
            Whether to return the trained weights that define the ridge regression
            model.
        clip_bounds : array-like, optional
            If None, do not clip predictions. If not None, must be ann array of
            dimension ``n_outcomes X 2``. If any of the elements of the array are None,
            ignore that bound (e.g. if a row of the array is [None, 10], apply an upper
            bound of 10 but no lower bound).
        intercept : bool, optional
            Whether to add an unregulated intercept (or, equivalently, center the X and
            Y data).
        allow_linalg_warning_instances : bool, optional
            If False (default), track for which hyperparameters did ``scipy.linalg``
            raise an ill-conditioned matrix error, which could lead to poor performance.
            This is used to discard these models in a cross-validation context. If True,
            allow these models to be included in the hyperparameter grid search. Note
            that these errors will not occur when using ``cupy.linalg`` (i.e. if a GPU
            is detected), so the default setting may give differing results across
            platforms.

    Returns
    -------
    dict of :class:`numpy.ndarray`
        The results dictionary will always include the following key/value pairs:
            ``metrics_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is a dictionary of {Out-of,In}-sample model performance
                metrics for each lambda

        If ``return_preds``, the following arrays will be appended in order:
            ``y_pred_{test,train}`` : array of dimension n_outcomes X n_lambdas
                Each element is itself a 1darray of {Out-of,In}-sample predictions for
                each lambda. Each 1darray contains n_obs_{test,train} values

        if return_model, the following array will be appended:
            ``models`` : array of dimension n_outcomes X n_lambdas:
                Each element is itself a 1darray of model weights for each lambda. Each
                1darray contains n_ftrs values
    """

    # get dimensions needed to shape arrays
    n_ftrs, n_outcomes, n_obs_train, n_obs_test = get_dim_lengths(
        X_train, y_train, y_test
    )
    n_lambdas = len(lambdas)

    # center data if needed
    X_train, y_train, X_offset, y_offset, _ = _preprocess_data(
        X_train, y_train, intercept, normalize=False
    )

    # set up the data structures for reporting results
    results_dict = _initialize_results_arrays(
        (n_outcomes, n_lambdas), return_preds, return_model
    )

    t1 = time.time()

    # send to GPU if available
    X_train = xp.asarray(X_train)
    y_train = xp.asarray(y_train)

    if DEBUG:
        if GPU:
            print(f"Time to transfer X_train and y_train to GPU: {time.time() - t1}")
        t1 = time.time()

    # precomputing large matrices to avoid redundant computation
    if svd_solve:
        # precompute the SVD
        U, s, Vh = linalg.svd(X_train, full_matrices=False)
        V = Vh.T
        UT_dot_y_train = U.T.dot(y_train)
    else:
        XtX = X_train.T.dot(X_train)
        XtY = X_train.T.dot(y_train)

    if DEBUG:
        t2 = time.time()
        print("Time to create XtX matrix:", t2 - t1)

    # iterate over the lambda regularization values
    training_time = 0
    pred_time = 0
    for lx, lambdan in enumerate(lambdas):
        if DEBUG:
            t3 = time.time()

        # train model
        if svd_solve:
            s_lambda = s / (s ** 2 + lambdan * xp.ones_like(s))
            model = (V * s_lambda).dot(UT_dot_y_train)
            lambda_warning = None
        else:
            with warnings.catch_warnings(record=True) as w:
                # bind warnings to the value of w
                warnings.simplefilter("always")
                lambda_warning = False
                model = linalg.solve(
                    XtX + lambdan * xp.eye(n_ftrs, dtype=np.float64),
                    XtY,
                    **linalg_solve_kwargs,
                )

                # if there is a warning
                if len(w) > 1:
                    for this_w in w:
                        print(this_w.message)
                    # more than one warning is bad
                    raise Exception("warning/exception other than LinAlgWarning")
                if len(w) > 0:
                    # if it is a linalg warning
                    if w[0].category == LinAlgWarning:
                        print("linalg warning on lambda={0}: ".format(lambdan), end="")
                        # linalg warning
                        if not allow_linalg_warning_instances:
                            print("we will discard this model upon model selection")
                            lambda_warning = True
                        else:
                            lambda_warning = None
                            print("we will allow this model upon model selection")
                    else:
                        raise Exception("warning/exception other than LinAlgWarning")

        if DEBUG:
            t4 = time.time()
            training_time += t4 - t3
            print(f"Training time for lambda {lambdan}: {t4 - t3}")

        #####################
        # compute predictions
        #####################

        # send to gpu if available
        X_test = xp.asarray(X_test)
        y_test = xp.asarray(y_test)
        y_offset = xp.asarray(y_offset)
        X_offset = xp.asarray(X_offset)

        if DEBUG:
            t5 = time.time()

        # train
        pred_train = X_train.dot(model) + y_offset
        pred_train = y_to_matrix(pred_train)

        # test
        pred_test = X_test.dot(model) - X_offset.dot(model) + y_offset
        pred_test = y_to_matrix(pred_test)

        # clip if needed
        if clip_bounds is not None:
            for ix, i in enumerate(clip_bounds):
                # only apply if both bounds aren't None for this outcome
                if not (i == None).all():
                    pred_train[:, ix] = xp.clip(pred_train[:, ix], *i)
                    pred_test[:, ix] = xp.clip(pred_test[:, ix], *i)

        if DEBUG:
            t6 = time.time()
            pred_time += t6 - t5

        # bring back to cpu if needed
        pred_train, pred_test = asnumpy(pred_train), asnumpy(pred_test)
        y_train, y_test, model = (
            y_to_matrix(asnumpy(y_train)),
            y_to_matrix(asnumpy(y_test)),
            y_to_matrix(asnumpy(model)),
        )

        # create tuple of lambda index to match argument structure
        # of _fill_results_arrays function
        hp_tuple = (lx,)

        # Transpose model results so that n_outcomes is first dimension
        # so that _fill_results_array can handle it
        model = model.T

        # populate results dict with results from this lambda
        results_dict = _fill_results_arrays(
            y_train,
            y_test,
            pred_train,
            pred_test,
            model,
            hp_tuple,
            results_dict,
            hp_warning=lambda_warning,
        )
    if DEBUG:
        print("Training time:", training_time)
        print("Prediction time:", pred_time)
        print("Total time:", time.time() - t1)
    return results_dict


def kfold_solve(
    X,
    y,
    solve_function=ridge_regression,
    num_folds=5,
    return_preds=True,
    return_model=False,
    **kwargs_solve,
):
    """A general skeleton function for computing k-fold cross validation solves.

    Args:
        X (n_obs X n_ftrs 2darray): Feature matrix
        y (n_obs X n_outcomes 2darray): Attribute matrix
        solve_function (func): Which solve function in this module will you be using
        num_folds (int): How many folds to use for CV
        return_preds (bool): Return predictions for training and test sets?
        return_model (bool): Return the trained weights that define the ridge regression
            model?
        kwargs_solve (dict): Parameters to pass to the solve func

    Returns:
        Dict of ndarrays.
            The dict will always start with the following 4 key:value pairs. "..."
                refers to a number of dimensions equivalent to the number of
                hyperparameters, where each dimension has a length equal to the number
                of values being tested for that hyperparameter. The number of
                hyperparameters and order returned is defined in the definition of the
                particular solve function we have passed as the solve_function argument:
                    metrics_test: n_folds X n_outcomes X ... ndarray of dict:
                        Out-of-sample model performance metrics for each fold, for each
                        outcome, for each hyperparameter value
                    metrics_train: n_folds X n_outcomes X ... ndarray of dict: In-sample
                        model performance metrics
                    obs_test: n_folds X  n_outcomes  X ... array of ndarray of float64:
                        Out-of-sample observed values for each fold
                    obs_train: n_folds X  n_outcomes X ... array of ndarray of float64:
                        In-sample observed values
                    cv: :py:class:`sklearn.model_selection.KFold` : kfold
                        cross-validation splitting object used

            If return_preds, the following arrays will included:
                preds_test: n_folds X  n_outcomes X ... ndarray of ndarray of float64:
                    Out-of-sample predictions or each fold, for each outcome, for each
                    hyperparameter value
                preds_train: n_folds X n_outcomes X ... ndarray of ndarray of float64:
                    In-sample predictions

            if return_model, the following array will be included:
                models: n_folds X n_outcomes X ... ndarray of same type as model: Model
                    weights/parameters. xxx here is of arbitrary dimension specific to
                    solve_function
    """
    assert num_folds > 1

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

    # keep track of all runs over several iterations
    kfold_metrics_test = []
    kfold_metrics_train = []
    kfold_preds_test = []
    kfold_preds_train = []
    kfold_y_train = []
    kfold_y_test = []
    kfold_models = []
    hp_warnings = []
    i = 0
    print("on fold (of {0}): ".format(num_folds), end="")

    for train_idxs, val_idxs in kf.split(X):
        i += 1
        print("{0} ".format(i), end="")

        X_train, X_val = X[train_idxs], X[val_idxs]
        y_train, y_val = y[train_idxs], y[val_idxs]

        # record train/test obs for this split
        kfold_y_train.append(y_train)
        kfold_y_test.append(y_val)

        # call solve func
        solve_results = solve_function(
            X_train,
            X_val,
            y_train,
            y_val,
            return_preds=return_preds,
            return_model=return_model,
            **kwargs_solve,
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


def single_solve(
    X_train,
    X_val,
    y_train,
    y_val,
    solve_function=ridge_regression,
    return_preds=True,
    return_model=False,
    **kwargs_solve,
):
    """A general skeleton function for computing k-fold cross validation solves.

    Args:
        X_train, X_val (n_train_obs X n_ftrs 2darray), (n_test_obs X n_ftrs 2darray): Feature matrices
        y_train, y_val: y (n_train_obs X n_outcomes 2darray), (n_test_obs X n_outcomes 2darray) : Attribute matrices
        solve_function (func): Which solve function in this module will you be using
        num_folds (int): How many folds to use for CV
        return_preds (bool): Return predictions for training and test sets?
        return_model (bool): Return the trained weights that define the ridge regression model?
        kwargs_solve (dict): Parameters to pass to the solve func

    Returns:
        Dict of ndarrays.
            The dict will always start with the following 4 key:value pairs. "..." refers to a number
            of dimensions equivalent to the number of hyperparameters, where each dimension
            has a length equal to the number of values being tested for that hyperparameter.
            The number of hyperparameters and order returned is defined in the definition of
            the particular solve function we have passed as the solve_function argument:
                metrics_test:  n_outcomes X ... ndarray of dict: Out-of-sample model performance
                    metrics for each fold, for each outcome, for each hyperparameter value
                metrics_train: n_outcomes X ... ndarray of dict: In-sample model performance metrics
                obs_test: n_folds X  n_outcomes  X ... array of ndarray of float64: Out-of-sample observed values
                    for each fold
                obs_train:  n_outcomes X ... array of ndarray of float64: In-sample observed values
                cv: :py:class:`sklearn.model_selection.KFold` : kfold cross-validation splitting object used

            If return_preds, the following arrays will included:
                preds_test:  n_outcomes X ... ndarray of ndarray of float64: Out-of-sample predictions
                    for each fold, for each outcome, for each hyperparameter value
                preds_train: n_outcomes X ... ndarray of ndarray of float64: In-sample predictions

            if return_model, the following array will be included:
                models: n_outcomes X ... ndarray of same type as model: Model weights/parameters. xxx here is of
                    arbitrary dimension specific to solve_function
    """
    # call solve func
    solve_results = solve_function(
        X_train,
        X_val,
        y_train,
        y_val,
        return_preds=return_preds,
        return_model=return_model,
        **kwargs_solve,
    )

    # Return results wrapped to interface with interpret_results functoins
    rets = {
        "metrics_test": np.array([solve_results["metrics_test"]]),
        "metrics_train": np.array([solve_results["metrics_train"]]),
        "y_true_test": np.array(y_val),
        "y_true_train": np.array(y_train),
        "hp_warning": np.array([solve_results["hp_warning"]]),
    }

    if return_preds:
        rets["y_pred_test"] = np.array([solve_results["y_pred_test"]])
        rets["y_pred_train"] = np.array([solve_results["y_pred_train"]])

    if return_model:
        rets["models"] = np.array([solve_results["models"]])

    if GPU:
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    return rets


def compute_metrics(true, pred):
    """takes in a vector of true values, a vector of predicted values. To add more metrics,
    just add to the dictionary (possibly with a flag or when
    it is appropriate to add)"""
    res = dict()

    residuals = true - pred
    res["mse"] = np.sum(residuals ** 2) / residuals.shape[0]
    res["r2_score"] = metrics.r2_score(true, pred)

    return res


def _initialize_results_arrays(arr_shapes, return_preds, return_models):
    # these must be instantiated independently
    results_dict = {
        "metrics_test": np.empty(arr_shapes, dtype=dict),
        "metrics_train": np.empty(arr_shapes, dtype=dict),
    }
    if return_preds:
        results_dict["y_pred_test"] = np.empty(arr_shapes, dtype=np.ndarray)
        results_dict["y_pred_train"] = np.empty(arr_shapes, dtype=np.ndarray)
    if return_models:
        results_dict["models"] = np.empty(arr_shapes, dtype=np.ndarray)

    # for numerical precision tracking
    results_dict["hp_warning"] = np.empty(arr_shapes, dtype=object)
    results_dict["hp_warning"].fill(None)
    return results_dict


def _fill_results_arrays(
    y_train,
    y_test,
    pred_train,
    pred_test,
    model,
    hp_tuple,
    results_dict,
    hp_warning=None,
):
    """Fill a dictionary of results with the results for this particular
    set of hyperparameters.

    Args:
        y_{train,test} (n_obs_{train,test} X n_outcomes 2darray of float)
        pred_{train,test} (n_obs_{train,test} X n_outcomes 2darray of float)
        model (n_outcomes 1darray of arbitrary dtype)
        hp_tuple (tuple): tuple of hyperparameter values used in this model
        results_dict (dict): As created in solve functions, to be filled in.
    """

    n_outcomes = y_train.shape[1]
    for i in range(n_outcomes):

        # get index of arrays that we want to fill
        # first dimension is outcome, rest are hyperparams
        this_ix = (i,) + hp_tuple

        # compute and save metrics
        results_dict["metrics_train"][this_ix] = compute_metrics(
            y_train[:, i], pred_train[:, i]
        )
        results_dict["metrics_test"][this_ix] = compute_metrics(
            y_test[:, i], pred_test[:, i]
        )

        # save predictions if requested
        if "y_pred_test" in results_dict.keys():
            results_dict["y_pred_train"][this_ix] = pred_train[:, i]
            results_dict["y_pred_test"][this_ix] = pred_test[:, i]

        # save model results if requested
        if "models" in results_dict.keys():
            results_dict["models"][this_ix] = model[i]

        # save hp warnings if thats desired
        results_dict["hp_warning"][this_ix] = hp_warning

    return results_dict


def y_to_matrix(y):
    """ ensures that the y value is of non-empty dimesnion 1 """
    if type(y) == list:
        y = np.array(y)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    return y


def get_dim_lengths(X_train, Y_train, Y_test=None):
    """ packages data dimensions into one object"""
    if Y_train.ndim == 1:
        n_outcomes = 1
    else:
        n_outcomes = Y_train.shape[1]
    n_ftrs = X_train.shape[1]
    n_obs_trn = Y_train.shape[0]

    results = [n_ftrs, n_outcomes, n_obs_trn]
    if Y_test is not None:
        results.append(Y_test.shape[0])
    return results


def split_world_sample_solve(
    X,
    Y,
    latlonsdf,
    sample=0,
    subset_n=slice(None),
    subset_feat=slice(None),
    num_folds=5,
    solve_function=ridge_regression,
    globalclipping=False,
    **kwargs_solve,
):
    """
    runs a cross-validated solve on a subset of X, Y data defined by the sampling indicator contained in
    the latlonsdf object.

    input:
        X, Y are the features and labels matrices, respectively.
        latlonsdf is a pandas dataframe of lat-lon combinations, containing a column called 'samp' which
           contains an indicator of which sample each lat-lon falls into.
        sample is a scalar from 0 to 5 indicating which subregion of the world you want to solve for.
        subset_n and subset_feat can be used to subset observations (_n) and.or features (_feat)
        num_folds, solve_function are as described in kfold_solve
        globalclipping is logical; True implies clipping across the whole distribution in Y, False
            implies clipping within each sample passed into the function.

    returns:
        kfold_results object from the function solve.kfold_solve()
    """

    # limit to just your sample
    ids_samp = np.where(latlonsdf["samp"] == sample)
    X_samp = X.iloc[ids_samp]
    Y_samp = Y.iloc[ids_samp]
    latlonsdf_samp = latlonsdf.iloc[ids_samp]

    # latlons back to ndarray
    this_latlons_samp = latlonsdf_samp.values

    # clip: globally or locally
    mykwargs = kwargs_solve

    if not globalclipping:
        mykwargs["clip_bounds"] = Y_samp.describe().loc[["min", "max"], :].T.values
    else:
        mykwargs["clip_bounds"] = Y.describe().loc[["min", "max"], :].T.values

    # split sample data into train and test
    (
        X_train,
        X_test,
        Y_train,
        Y_test,
        idxs_train,
        idxs_test,
    ) = parse.split_data_train_test(
        X_samp.values, Y_samp.values, frac_test=0.2, return_idxs=True
    )
    latlons_train_samp = this_latlons_samp[idxs_train]

    # solve
    kfold_results_samp = kfold_solve(
        X_train[subset_n, subset_feat],
        Y_train[subset_n],
        solve_function=solve_function,
        num_folds=num_folds,
        return_model=True,
        **mykwargs,
    )

    # return the kfold_results object
    return kfold_results_samp, latlons_train_samp, idxs_train, idxs_test, mykwargs
