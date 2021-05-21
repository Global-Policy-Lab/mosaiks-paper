import numpy as np
from mosaiks.solve import interpret_results
from mosaiks.solve import solve_functions as solve


def checkered_predictions_by_radius(
    X,
    y,
    latlons,
    radii,
    extent,
    num_jitter_positions_sqrt=1,
    min_points=0,
    return_hp_idxs=False,
    return_models=False,
    crit="r2_score",
    solve_function=solve.ridge_regression,
    **solve_kwargs,
):
    """
    Consider each grid cell as its own test set, while training on all other cells.
    args:
        X: n x d array of floats, feature matrix
        y: n x 1 array of floats, labels
        latlons: n x 2 array of floats, locations
        radii: list of floats, radii defining the grid at successive trials
        extent: 4x1 list/array of floats, total extent on which to define the grid, e.g. the U.S. is captured
            by extent = [25,48,-126,-65]
        num_jitter_positions_sqrt: int, how many jitter positions to use in each dimension
        min_points: int, the minimum number of points at which to define a set.
        return_hp_idxs: boolean, whether to return the optimal hyperparameter indicies
        return_models: boolean, whether to return the models
        crit: which criteria to optimize for (if not r2_score,
                you'll also have to set a flag in interpret_results for minimization)
        solve_function: fxn, which solve_function to use
        **solve_kwargs: dict of keyqord arguments that you want to pass to the solve function

    returns:
        rets_by_delta: a list of dictionary of results, where each dictionary in the list corresponds to results
            for one radius value. The structure of each dictionary depends on the arguments given in the option
            return arguments.
    """

    # The object to return
    rets_by_radius = []
    # For each radius
    for i, radius in enumerate(radii):
        print(f"Radius: {i + 1}/{len(radii)}: Offset: ", end="")

        # If we're not jittering, just do it once:
        if num_jitter_positions_sqrt == 1:
            print("1/1...")
            idxs_a, idxs_b = put_in_checkers(latlons, extent, radius * 2, min_points=0)
            rets_0 = checkered_predictions(
                X,
                y,
                latlons,
                idxs_a,
                idxs_b,
                radius,
                return_hp_idxs=return_hp_idxs,
                return_models=return_models,
                crit=crit,
                solve_function=solve_function,
                **solve_kwargs,
            )

        # If we are jittering, do the same thing as above but num_jitter_positions_sqrt^2 times.
        # use checkered_predictions_just_return_results to get the solve results, and aggregate
        # them with the other data to be returned, per jitter.
        else:
            center_offsets = np.linspace(0, radius, num_jitter_positions_sqrt)
            # print(center_offsets)
            rets_0 = {}

            # mimic the returns of kfold_solve
            jitter_metrics_test = []
            jitter_metrics_train = []
            jitter_preds_test = []
            jitter_preds_train = []
            jitter_models = []
            jitter_y_true_train = []
            jitter_y_true_test = []
            jitter_hp_warning = []

            for dx1, delta_1 in enumerate(center_offsets):
                for dx2, delta_2 in enumerate(center_offsets):
                    n_sample = num_jitter_positions_sqrt * dx1 + dx2 + 1
                    print(f"{n_sample}/{num_jitter_positions_sqrt**2}", end="...")
                    idxs_a, idxs_b = put_in_checkers(
                        latlons,
                        extent,
                        radius * 2,
                        offset_x1=delta_1,
                        offset_x2=delta_2,
                        min_points=min_points,
                    )

                    rets_offset = checkered_predictions_just_return_results(
                        X,
                        y,
                        latlons,
                        idxs_a,
                        idxs_b,
                        radius,
                        return_hp_idxs=return_hp_idxs,
                        return_models=return_models,
                        crit=crit,
                        solve_function=solve_function,
                        **solve_kwargs,
                    )

                    ## case this into a kfold return - index everything by zero to avoid having an extra index
                    # record performance metrics
                    jitter_metrics_test.append(rets_offset["metrics_test"][0])
                    jitter_metrics_train.append(rets_offset["metrics_train"][0])
                    # record true y
                    jitter_y_true_train.append(rets_offset["y_true_train"][0])
                    jitter_y_true_test.append(rets_offset["y_true_test"][0])
                    # record optional preds and model parameters
                    jitter_preds_test.append(rets_offset["y_pred_test"][0])
                    jitter_preds_train.append(rets_offset["y_pred_train"][0])
                    # record the hp_warnings so that they can be passed to interpret_results
                    jitter_hp_warning.append(rets_offset["hp_warning"][0])
                    # record the model as well if desired
                    if return_models:
                        jitter_models.append(rets_offset["models"][0])

            # Return results
            jittered_results_this_delta = {
                "metrics_test": np.array(jitter_metrics_test),
                "metrics_train": np.array(jitter_metrics_train),
                "y_true_test": np.array(jitter_y_true_test),
                "y_true_train": np.array(jitter_y_true_train),
                "y_pred_test": np.array(jitter_preds_test),
                "y_pred_train": np.array(jitter_preds_train),
                "deltas": (delta_1, delta_2),
                "hp_warning": np.array(jitter_hp_warning),
            }

            # note: each jitter is treated like a fold of kfold cross validation
            (
                best_hp_idxs,
                metrics_best_idx,
                y_pred_best_idx,
            ) = interpret_results.interpret_kfold_results(
                jittered_results_this_delta, crit
            )

            # default return is a list but there's only one variable so index all at 0
            rets_0["hp_idxs_chosen"] = best_hp_idxs[0]
            rets_0["metrics_test"] = metrics_best_idx[:, 0]
            rets_0["preds_test"] = y_pred_best_idx[:, 0]

        rets_by_radius.append(rets_0)
        print("")

    return rets_by_radius


def checkered_predictions_just_return_results(
    X,
    y,
    latlons,
    idxs_train,
    idxs_test,
    radius,
    return_hp_idxs=False,
    return_models=False,
    crit="r2_score",
    solve_function=solve.ridge_regression,
    **solve_kwargs,
):

    """
    bare bones function for just returning solve function, to be parsed later
    (e.g. by checkered_predictions_by_radius()).
    args:
        X: n x d array of floats, feature matrix
        y: n x 1 array of floats, labels
        latlons: n x 2 array of floats, locations
        idxs_train: list of ints = indices into latlons defining the training grid
        idxs_test: list of ints = indices into latlons defining the testing grid
        radius (float): radius defining the grid
        return_hp_idxs (bool): whether to return the hyperparameter idxs
        return_models (bool): whether to return the models
        crit (string/list of strings): which criteria to optimize over in aggregating results
        solve_function (fxn): which solve_function to use
        **solve_kwargs (dict): keyword arguments that you want to pass to the solve function
     returns:
        dict of the following results:
            latlons_test: list of latlons corresponding to the test points
            latlons_train: list of latlons corresponding to the train points
            preds_test: vector of floats, predicted values for the test set
            metrics_test: dict, metrics on the predictions for the test set
            hp_idxs (optional): the list of lambda values that are used for the results return in preds_all and metrics_all
            model (optional): the model trained on the train set
    """

    # double check y vector size
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # make the train and test sets
    latlons_train, latlons_test = latlons[idxs_train], latlons[idxs_test]
    X_train, X_test = X[idxs_train, :], X[idxs_test, :]
    y_train, y_test = y[idxs_train, :], y[idxs_test, :]

    # if solve_kwargs already has return_preds, ignore b/c
    # we need to return preds regardless
    solve_kwargs.pop("return_preds", None)

    # return solve_results as they are given from single solve; will aggregate later
    solve_results = solve.single_solve(
        X_train,
        X_test,
        y_train,
        y_test,
        return_preds=True,
        return_model=return_models,
        solve_function=solve_function,
        **solve_kwargs,
    )

    return solve_results


def checkered_predictions(
    X,
    y,
    latlons,
    idxs_train,
    idxs_test,
    radius,
    return_hp_idxs=False,
    return_models=False,
    crit="r2_score",
    solve_function=solve.ridge_regression,
    **solve_kwargs,
):
    """
    Function for returning all necessary data for checkered predictions when there is only one jittering.
    args:
        X: n x d array of floats, feature matrix
        y: n x 1 array of floats, labels
        latlons: n x 2 array of floats, locations
        idxs_train: list of ints = indices into latlons defining the training grid
        idxs_test: list of ints = indices into latlons defining the testing grid
        radius (float): radius defining the grid
        return_hp_idxs (bool): whether to return the hyperparameter idxs
        return_models (bool): whether to return the models
        crit (string/list of strings): which criteria to optimize over in aggregating results
        solve_function (fxn): which solve_function to use
        **solve_kwargs (dict): keyword arguments that you want to pass to the solve function
     returns:
        dict of the following results:
            latlons_test: list of latlons corresponding to the test points
            latlons_train: list of latlons corresponding to the train points
            preds_test: vector of floats, predicted values for the test set
            metrics_test: dict, metrics on the predictions for the test set
            hp_idxs (optional): the list of lambda values that are used for the results return in preds_all and metrics_all
            model (optional): the model trained on the train set
    """

    # make the train and test sets
    latlons_train, latlons_test = latlons[idxs_train], latlons[idxs_test]
    X_train, X_test = X[idxs_train, :], X[idxs_test, :]
    y_train, y_test = y[idxs_train, :], y[idxs_test, :]

    # if solve_kwargs already has return_preds, ignore b/c
    # we need to return preds regardless
    solve_kwargs.pop("return_preds", None)

    # solve
    solve_results = solve.single_solve(
        X_train,
        X_test,
        y_train,
        y_test,
        return_preds=True,
        return_model=return_models,
        solve_function=solve_function,
        **solve_kwargs,
    )

    hp_best_idx, metrics_best, preds_best = interpret_results.interpret_single_results(
        solve_results, crit
    )

    rets = {
        "latlons_train": latlons_train,
        "latlons_test": latlons_test,
        "metrics_test": metrics_best,
        "preds_test": preds_best,
    }
    if return_hp_idxs:
        rets["hp_idxs"] = hp_best_idx
    if return_models:
        num_outcomes = len(solve_results["models"][0])
        if num_outcomes > 1:
            models_all.append(
                [
                    solve_results["models"][0][i][l_best_idx[i]]
                    for i in range(num_outcomes)
                ]
            )
        else:
            models_all.append(solve_results["models"][0][0][l_best_idx])
    return rets


def put_in_checkers(
    latlons, extent, cell_width, offset_x1=0.0, offset_x2=0.0, min_points=0
):
    """
    from a bounding box, return a grid of points such that the bounding box is covered,
    and cells are non-overlapping with width cell_width. Only cells with values in them are returned.
    args:
        latlons: n x 2 list/array of points in space
        extent: 1x4 list of floats = [latmin,latmax,lonmin,lonmax] of the total bounding box
        cell_width: float, width of each (sqaure) cell
        offset_x1: float, offset by which to start the grid in the x1 dimension
        offset_x2: float, offset by which to start the grid in the x1 dimension
        min_points: int, optional arguments for how many points must fall within the cell for it
                    to be included in the grid

    returns:
        idxs_a, idxs_b: list of indices splitting a and b, which index into latlons.
    """
    [latmin, latmax, lonmin, lonmax] = extent
    radius = cell_width / 2.0

    idxs_a = []
    idxs_b = []
    assert offset_x1 <= cell_width
    assert offset_x2 <= cell_width

    for i, (lat_this, lon_this) in enumerate(latlons):
        # need to find the maximal i such that latlon-offet + cell_width*i <= lat_this
        grid_idx_i = int(np.floor((lat_this - (latmin - offset_x1)) / cell_width))
        grid_idx_j = int(np.floor((lon_this - (lonmin - offset_x2)) / cell_width))
        if (grid_idx_i - grid_idx_j) % 2 == 0:
            idxs_a.append(i)
        else:
            idxs_b.append(i)
    return np.array(idxs_a, dtype=int), np.array(idxs_b, dtype=int)


def results_to_metrics(results_checkered):
    """
    Helper function for converting radius checkerboard results into plottable objects.
    """
    return np.array(
        [
            results_this_radius["metrics_test"]
            for results_this_radius in results_checkered
        ]
    )
