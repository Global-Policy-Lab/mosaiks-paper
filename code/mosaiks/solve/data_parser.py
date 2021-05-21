import csv

import numpy as np

from .. import transforms
from ..utils import io as mio


def csv_to_np(csv_file, y_labels, id_label="ID", return_ll=True):
    """
    args:
        csv_file: a path to csv file which has the csvs
        y_labels: list of strings of output variables to collect
        id_label: string is associated to the id column in the order of y_labels
    returns:
        result: a list of ids,
                a matrix of ys (where if you have just one attribute you would index it as ys[0]),
                and a matrix of latlons, such that all outputs have rows corresponding to the other outputs.
    """
    datafile = open(csv_file)

    data_reader = csv.DictReader(datafile)
    column_names = data_reader.fieldnames
    assert (
        id_label in column_names
    ), f"{id_label} not in column_names. column names are {column_names}"
    for y_label in y_labels:
        assert y_label in column_names, "{0} not in column_names".format(y_label)
    ids = []
    lons = []
    lats = []
    ys = [[] for i in range(len(y_labels))]

    for row in data_reader:
        for i, y_label in enumerate(y_labels):
            if row[y_label] == "":
                ys[i].append(np.nan)
            elif row[y_label] == "NA":
                ys[i].append(np.nan)
            else:
                ys[i].append(float(row[y_label]))
        ids.append(row[id_label])
        if return_ll:
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
    ys = np.array(ys).T

    result = [ids, ys]
    if return_ll:
        result.append(np.vstack((np.array(lats), np.array(lons))).T.astype("float64"))

    return result


def merge(df_w_subset, *dfs):
    """Take dataframes containing relevant values (e.g. lat/lons, y, X matrices) and
    return numpy arrays that are properly sorted to the index of the first input
    dataframe.

    Parameters
    ----------
    df_w_subset : :class:`pandas.DataFrame`
        The dataframe with the index that we are going to reindex all other dataframes
        to before converting to numpy array
    *dfs : :class:`pandas.DataFrame`
        The other dataframes that will be sorted and returned as arrays

    Returns
    -------
    list of :class:`numpy.ndarray`
        The consistently sorted arrays, in the same order as the input DataFrames.
    """
    return [df_w_subset.values] + [d.reindex(df_w_subset.index).values for d in dfs]


def split_idxs_train_test(n, nums_train_test=None, frac_test=None, seed=0):
    """separate indices into train and test sets by using frac_test
    specificy either nums_train_test or frac_test but not both.
    don't change the seed unless you have a good reason to."""
    assert not (
        (frac_test is None) and (nums_train_test is None)
    ), "must specify either fraction test or numbers train and test"
    assert (frac_test is None) or (
        nums_train_test is None
    ), " cannot specify both fraction test and numbers train and test"

    if not (nums_train_test is None):
        assert len(nums_train_test) == 2, "must speficy two numbers"
        assert (
            nums_train_test[0] + nums_train_test[1] <= n
        ), "asking for a bigger dataset than you gave!"

        train_size, test_size = nums_train_test
    else:
        test_size = int(frac_test * n)
        train_size = n - test_size

    r = np.random.RandomState(seed=seed)
    new_idxs = r.choice(n, n, replace=False)

    train_idxs = new_idxs[0:train_size]
    test_idxs = new_idxs[train_size : train_size + test_size]
    return train_idxs, test_idxs


def split_data_train_test(
    X, y, frac_test=None, nums_train_test=None, seed=0, return_idxs=False
):
    """separate X and y  into train and test sets by using frac_test
    specificy either nums_train_test or frac_test but not both.
    don't change the seed unless you have a good reason to.
    I suggest using return_idxs=True so that you get both the split dataset, as well as the
    indicies, for e.g. spliting the latlon values later."""

    assert X.shape[0] == len(y), "dimensions of X and y do not match!"
    train_idxs, test_idxs = split_idxs_train_test(
        X.shape[0], frac_test=frac_test, nums_train_test=nums_train_test, seed=seed
    )
    if return_idxs:
        return (
            X[train_idxs],
            X[test_idxs],
            y[train_idxs],
            y[test_idxs],
            train_idxs,
            test_idxs,
        )
    else:
        return X[train_idxs], X[test_idxs], y[train_idxs], y[test_idxs]


def split_world_sample(data):
    """
    creates a new column in your pandas dataframe containing an indicator from 0 to 5 for which subregion of the globe
    each lat-lon pair falls into.

    input:
        must be a pandas dataframe containing lon and lat columns. currently not generalized to allow for splitting
        the world into any more or less pieces than 6.

    returns:
        the same dataframe that is input, with an additional column called 'samp' indicating the sample
    """

    data["samp"] = 0
    data.loc[
        (data["lon"] >= -180) & (data["lon"] < -31.5) & (data["lat"] >= 12.5), "samp"
    ] = 0
    data.loc[
        (data["lon"] >= -31.5) & (data["lon"] < 60) & (data["lat"] >= 12.5), "samp"
    ] = 1
    data.loc[(data["lon"] >= 60) & (data["lat"] >= 5.5), "samp"] = 2
    data.loc[
        (data["lon"] >= -180) & (data["lon"] < -31.5) & (data["lat"] < 12.5), "samp"
    ] = 3
    data.loc[
        (data["lon"] >= -31.5) & (data["lon"] < 60) & (data["lat"] < 12.5), "samp"
    ] = 4
    data.loc[(data["lon"] >= 60) & (data["lat"] < 5.5), "samp"] = 5

    return data


def creategrid(Klon, Klat):
    """Create a vector of longitude values and latitude values that define the edges of a grid
    with Klon intervals in the x-dimension and Klat intervals in the y-dimension.
    """
    if 360 % Klon == 0:
        steplon = 360 / Klon
    else:
        print("Stop! 360 degrees of longitude must be divisible by K!")
        return
    if 180 % Klat == 0:
        steplat = 180 / Klat
    else:
        print("Stop! 360 degrees of longitude must be divisible by K!")
        return
    xvec = [-180 + i * steplon for i in range(Klon + 1)]
    yvec = [-90 + i * steplat for i in range(Klat + 1)]
    return xvec, yvec


def split_world_sample_regular(data, xvec, yvec):
    """
    very similar to split_world_sample_regular, but creates a regular grid instead of a custom grid that overlaps well with continents.
    this function creates a new column in your pandas dataframe containing an indicator from 0 to 5 for which subregion of the globe
    each lat-lon pair falls into.

    input:
        data: must be a pandas dataframe containing lon and lat columns. currently not generalized to allow for splitting
        the world into any more or less pieces than 6.
        xvec: output from creategrid, this is a list of longitude values that define the grid into which you will divide up the world.
        yvec: same as xvec, but for latitude.

    returns:
        the same dataframe that is input, with an additional column called 'samp' indicating the sample
    """
    # initial parameters we need
    N = [i for i in range((len(xvec) - 1) * (len(yvec) - 1))]
    data["samp"] = 9999

    # reverse y so that number of our grids goes from left,top to right,bottom
    yrev = yvec[::-1]

    # only need to go to N-1 spots to fill out the grid
    myy = yrev[:-1]
    myx = xvec[:-1]
    i = -1

    for y in myy:
        nexty = yrev[yrev.index(y) + 1]
        for x in myx:
            i += 1
            nextx = xvec[xvec.index(x) + 1]
            data.loc[
                (data["lat"] <= y)
                & (data["lat"] > nexty)
                & (data["lon"] >= x)
                & (data["lon"] < nextx),
                "samp",
            ] = N[i]

    return data


def merge_dropna_transform_split_train_test(c, app, X, latlons, ACS=False, seed=0):
    """This function performs a common workflow for many of our experiments. It involves
    the following steps:
        1. Load label values
        2. Reindex X matrix and lat/lon values to the y labels if they happen to have
           differently ordered indices.
        3. Drop observations where y is null
        4. Transform the y variable (e.g. take the natural logarithm)
        5. Split off the training/validation set from the test set (default 80/20 split)

    Parameters
    ----------
    c : :module:`mosaiks.config`
        The MOSAIKS config module
    app : str
        The name of the label you are running this workflow for
    X : :class:`pandas.DataFrame`
        The feature matrix for this analysis, indexed by the same IDs that the y
        variables are indexed by.
    latlons : :class:`pandas.DataFrame`
        A dataframe with the same index values as X and two columns ``lat`` and ``lon``
    ACS : bool, optional
        If True, amend filepath when searching for label data to include the ACS dir.
    seed : int, optional
        A seed used to split the training/validation set

    Returns
    -------
    X_train : :class:`numpy.ndarray`
        The feature matrix for your training data
    X_test : :class:`numpy.ndarray`
        The feature matrix for your test data
    Y_train : :class:`numpy.ndarray`
        The labels for your training data
    Y_test : :class:`numpy.ndarray`
        The labels for your test data
    latlons_train : :class:`numpy.ndarray`
        The locations of your training data, where column 0 is lat and colum 1 is lon
    latlons_test : :class:`numpy.ndarray`
        The locations of your test data, where column 0 is lat and colum 1 is lon
    """
    c = mio.get_filepaths(c, app)
    c_app = getattr(c, app)

    ## get labels
    print("Loading labels...")
    Y = mio.get_Y(c, c_app["colname"], ACS=ACS)

    ## merge X and Y accounting for different ordering
    ## and the sampling type
    print("Merging labels and features...")
    Y, X, latlons = merge(Y, X, latlons)

    ## drop obs and log transform if needed
    X, Y, latlons = transforms.dropna_and_transform(X, Y, latlons, c_app)

    ## Split the data into the training/validation sample vs. test sample
    ## (discarding test set for now to keep memory low)
    print("Splitting training/test...")
    X_train, X_test, Y_train, Y_test, idxs_train, idxs_test = split_data_train_test(
        X, Y, frac_test=c.ml_model["test_set_frac"], seed=seed, return_idxs=True
    )
    latlons_train = latlons[idxs_train]
    latlons_test = latlons[idxs_test]

    return X_train, X_test, Y_train, Y_test, latlons_train, latlons_test
