"""This module contains functions for transforming datasets"""

import numpy as np


def dropna_Y(Y, label):
    Y = Y.squeeze()
    # mark locations with missing labels:
    valid = ~np.isnan(Y) & ~np.isinf(Y) & ~(Y == -999)

    # mark outliers in individual datasets
    if label == "nightlights":
        valid = valid & (Y <= 629)

    # drop
    Y = Y[valid]

    return Y, valid


def dropna(X, Y, latlon, c_app):
    Y = Y.squeeze()

    # drop obs with missing labels:
    Y, valid = dropna_Y(Y, c_app["application"])
    latlon = latlon[valid]
    X = X[valid]
    return X, Y, latlon


def dropna_and_transform(X, Y, latlon, c_app):
    name = c_app["application"]
    X, Y, latlon = dropna(X, Y, latlon, c_app)
    transform_func = globals()["transform_" + name]
    return transform_func(X, Y, latlon, c_app["logged"])


def transform_elevation(X, Y, latlon, log):
    return X, Y, latlon


def transform_population(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_housing(X, Y, latlon, log):
    if log:
        Y = np.log(Y)
    return X, Y, latlon


def transform_income(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_nightlights(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_roads(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_treecover(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def log_all(Y, c_app):
    name = c_app["application"]
    if name == "housing":
        logY = np.log(Y)
    else:
        logY = np.log(Y + 1)
    return logY


##################################################
### ACS functions
##################################################


def transform_B08303(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B15003(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B19013(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B19301(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_C17002(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B22010(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B25071(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B25001(X, Y, latlon, log):
    # Drop observations with extremely high (top 0.1%) values:
    upperEnd = np.percentile(Y, 99.9)
    valid = ~(Y >= upperEnd)
    Y = Y[valid]
    X = X[valid]

    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B25002(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B25035(X, Y, latlon, log):
    # Transform from year to age
    Y = 2015.00 - Y
    # Drop the top .1% of obs -- buildings in the world just aren't that old.
    # http://www.oldest.org/structures/buildings-america/
    upperEnd = np.percentile(Y, 99.9)
    valid = Y < upperEnd
    Y = Y[valid]
    X = X[valid]
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B25017(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon


def transform_B25077(X, Y, latlon, log):
    if log:
        Y = np.log(Y + 1)
    return X, Y, latlon
