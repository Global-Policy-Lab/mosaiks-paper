import warnings

import dask.array as da
import dask.config
import numpy as np
from mosaiks.solve import solve_functions
from mosaiks.utils import io
from threadpoolctl import threadpool_limits

# is there a gpu
GPU = io.GPU

# this seems to be a reasonable optimization for both the GPU single thread and the
# multi-thread CPU scheduler
DASK_CHUNKSIZE = 2000


# GPU vs. CPU specific settings
if GPU:
    import cupy as cp

    mempool = cp.get_default_memory_pool()
    xp = cp

    # if GPU, need single threaded to not overwhelm gpu
    scheduler = "single-threaded"

    # no need to limit non-dask threads
    non_dask_thread_limit = None

else:
    xp = np

    # if not GPU, use threaded scheduler
    scheduler = "threads"

    # limiting openMP/BLAS threads to not conflict w/ dask threads
    non_dask_thread_limit = 1

# configure schduler
dask.config.set({"scheduler": scheduler})


def smooth(D, y, sigma):
    """Given a matrix ``D`` defining the squared distance between training and
    prediction points, and a matrix or vector y defining one or more lets of labels at
    the training points, return predicitons using an RBF kernel.

    Parameters
    ----------
    D : array-like
        Squared distance matrix. $D_{i,j}$ defines the squared distance between training
        point ``i`` and prediction point ``j``.
    y : array-like
        Labels for training points. May be 1d if a single prediction task or 2d if >1
        prediction tasks.
    sigma : int
        The length scale for RBF smoothing. i.e. the weight of a training point is
        proportional to $exp((-.5/sigma**2)*D_{i,j})$

    Returns
    -------
    smoothed_predictions : array-like
        Predicted labels
    """
    y = solve_functions.y_to_matrix(y)

    # get RBF smoothing matrix
    S = da.exp((-0.5 / sigma ** 2) * D)

    # if sigma is small enough, weights could turn to 0, so we reset those to
    # just guess the average of the training set
    S = da.where(S.sum(axis=1).reshape(-1, 1) > 0, S, 1)
    smoothed_predictions = S.dot(y) / S.sum(axis=1).reshape(-1, 1)

    return smoothed_predictions


def dist_sq_matrix(latlon1, latlon2):
    """Get the squared distance between two sets of points defined by lat/lon.

    Parameters
    ----------
    latlon1,2 : :class:`numpy.ndarray`
        N x 2 and M x 2 arrays defining the lat, lon coordinates of two sets of points

    Returns
    -------
    out : :class:`numpy.ndarray`
        N x M array defining the squared great circle distance between each point in
        ``latlon1`` and each point in ``latlon2``
    """
    # broadcast manually
    latlon1 = latlon1[:, xp.newaxis, :].repeat(latlon2.shape[0], axis=1)
    latlon2 = latlon2[xp.newaxis, ...].repeat(latlon1.shape[0], axis=0)

    # get dist
    out = _great_circle_dist_par(latlon1, latlon2, sq=True)

    # need to manage memory if on gpu
    if GPU:
        mempool.free_all_blocks()

    return out


def _great_circle_dist_par(latlon1, latlon2, sq=False):
    """Get the Great Circle distance between two sets of points defined by lat/lon.

    Parameters
    ----------
    latlon1,2 : :class:`numpy.ndarray`
        N x M x 2 arrays defining the lat, lon coordinates of two sets of points.
        ``latlon1`` is repeated (broadcast) along axis 1, and ``latlon2`` is repeated
        along axis 0.
    sq : bool, optional
        If True, square the result before returning

    Returns
    -------
    out : :class:`numpy.ndarray`
        N x M array defining the (squared) great circle distance between each point in
        ``latlon1`` and each point in ``latlon2``
    """

    # convert to rad
    lat1, lng1 = xp.radians(latlon1[:, :, 0]), xp.radians(latlon1[:, :, 1])
    lat2, lng2 = xp.radians(latlon2[:, :, 0]), xp.radians(latlon2[:, :, 1])

    # get trig funcs
    sin_lat1, cos_lat1 = xp.sin(lat1), xp.cos(lat1)
    sin_lat2, cos_lat2 = xp.sin(lat2), xp.cos(lat2)
    delta_lng = lng2 - lng1
    cos_delta_lng, sin_delta_lng = xp.cos(delta_lng), xp.sin(delta_lng)

    # calculate great circle dist
    d = xp.arctan2(
        xp.sqrt(
            (cos_lat2 * sin_delta_lng) ** 2
            + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_delta_lng) ** 2
        ),
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lng,
    )

    # convert back to degrees
    out = xp.rad2deg(d)

    # square distance if requested
    if sq:
        out = out ** 2

    return out


def rbf_interpolate_solve(
    latlons_train,
    latlons_val,
    y_train,
    y_val,
    return_preds=True,
    return_model=True,
    interpolate_train=False,
    clip_bounds=None,
    sigmas=[1],
):

    """
    Uses latlons to do non-parametric estimation of prediction points using RBF kernel
    Note that the latlons have taken the place of the features.

    latlons_train: training instances
    latlons_val: validation instances
    y_train: training labels
    y_val: validation labels
    return_preds (optional): bool, whether to return predictions
    return_preds (optional): bool, whether to return the model
    interpolate_train (optional): bool, whether to also interpolate training instance
        (e.g. for computing training set error)
    clip_bounds (None or n_outcomes X 2 2darray): If not None, clip the predictions to these bounds.
            If any of the elements of the array are None, ignore that bound (e.g. if a row of the array
            is [None, 10], apply an upper bound of 10 but no lower bound).
    sigmas: rbf kernel params to sweep over in the solve (as hyperparamers like lambda for ridge regression)
    """
    # if you've got a one dimensional response variable (y is just one column), make sure that it is properly formatted
    y_train, y_val = (
        solve_functions.y_to_matrix(y_train),
        solve_functions.y_to_matrix(y_val),
    )

    # get dimensions needed to shape arrays
    n_ftrs, n_outcomes, n_obs_train, n_obs_test = solve_functions.get_dim_lengths(
        latlons_train, y_train, y_val
    )
    n_sigmas = len(sigmas)

    # set up the data structures for reporting results
    results_dict = solve_functions._initialize_results_arrays(
        (n_outcomes, n_sigmas), return_preds, return_model
    )

    # to take advantage of GPU/CPU cores, convert to dask arrays
    # making sure we're setting threadpool limits to avoid oversubscribing threads
    with threadpool_limits(non_dask_thread_limit):
        latlons_train, latlons_val, y_train_da = [
            da.from_array(xp.asarray(i), chunks=(DASK_CHUNKSIZE, None))
            for i in [latlons_train, latlons_val, y_train]
        ]

        # calculate distances (ignore dask warnings about chunk size increase) -
        # this is intentional
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=da.PerformanceWarning)
            D_valtrain = da.blockwise(
                dist_sq_matrix,
                "ij",
                latlons_val,
                "ik",
                latlons_train,
                "jk",
                dtype=xp.float64,
                concatenate=True,
            )
            if GPU:
                mempool.free_all_blocks()
            if interpolate_train:
                D_traintrain = da.blockwise(
                    dist_sq_matrix,
                    "ij",
                    latlons_train,
                    "ik",
                    latlons_train,
                    "jk",
                    dtype=xp.float64,
                    concatenate=True,
                )

        # loop over all length scales that we are testing
        for g, sigma in enumerate(sigmas):
            smoothed_predictions_val = smooth(D_valtrain, y_train_da, sigma)
            if interpolate_train:
                smoothed_predictions_train = smooth(D_traintrain, y_train_da, sigma)
            else:
                smoothed_predictions_train = y_train_da

            # transfer from gpu if needed and turn into in-mem numpy arrays
            smoothed_predictions_train, smoothed_predictions_val = [
                io.gpu_return_and_clear(i).compute()
                for i in [smoothed_predictions_train, smoothed_predictions_val]
            ]

            # clip if needed (this is more easily done in numpy b/c dask does
            # not support assignment by slices)
            if clip_bounds is not None:
                for ix, i in enumerate(clip_bounds):
                    # only apply if both bounds aren't None for this outcome
                    if not (i == None).all():
                        smoothed_predictions_train[:, ix] = smoothed_predictions_train[
                            :, ix
                        ].clip(*i)
                        smoothed_predictions_val[:, ix] = smoothed_predictions_val[
                            :, ix
                        ].clip(*i)

            # assign "model" as sigma param
            model = sigma

            # create tuple of lambda index to match argument structure
            # of _fill_results_arrays function
            hp_tuple = (g,)

            # populate results dict with results from this sigma
            results_dict = solve_functions._fill_results_arrays(
                y_train,
                y_val,
                smoothed_predictions_train,
                smoothed_predictions_val,
                model,
                hp_tuple,
                results_dict,
            )

    # should not actually return r2 of 1 if didn't smooth training
    # instead return NaN
    for i in results_dict["metrics_train"][0]:
        for j in i.keys():
            i[j] = np.nan

    return results_dict
