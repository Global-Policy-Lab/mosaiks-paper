"""This module defines utility functions for spatial tasks."""

from os.path import join

import numpy as np
import pandas as pd


def ids_to_ij(ids):
    """Convert a list of 'i,j' ID strings to array
    of i and j

    Args:
        (list of str) ids

    Returns:
        Nx2 array of int: Lon, Lat
    """
    i = np.array([int(ix.split(",")[0]) for ix in ids])
    j = np.array([int(ix.split(",")[1]) for ix in ids])
    return np.vstack((i, j)).T


def ij_to_ll(i, j, grid_dir, grid_area, zoom=16, numPixels=640):
    """Converts from grid space to lat-lon space, returning
    the centroid of the grid cell passed to it.

    Args:
        i, j (int): row, column within selected grid. Indexed at 1.
        grid_dir (str) : directory containing grid files
        grid_area (str) : e.g. 'CONTUS'
        zoom (int) : google zoom level
        numPixels (int) : number of pixels per tile

    Returns:
        float,float: lon, lat of centroid of selected point
    """

    ll = np.load(join(grid_dir, "grid_{}_{}_{}.npz".format(grid_area, zoom, numPixels)))

    lats = ll["lat"]
    lons = ll["lon"]

    return lons[j - 1], lats[i - 1]


def ids_to_ll(ids, grid_dir, grid_area, zoom=16, numPixels=640):
    """Converts a list of 'i,j' ID strings to lists of the lon and
    lat values of the corresponding grid cell centroids.

    Args:
        ids (str) : 'i,j' corresponding to a given grid cell.
        grid_dir (str) : directory containing grid files
        grid_area (str) : e.g. 'CONTUS'
        zoom (int) : google zoom level
        numPixels (int) : number of pixels per tile

    Returns:
        float,float: lon, lat of centroid of selected point
    """

    ij = ids_to_ij(ids)
    return ij_to_ll(ij[:, 0], ij[:, 1], grid_dir, grid_area, zoom, numPixels)


def ll_to_ij(lon, lat, grid_dir, grid_area, zoom=16, numPixels=640):
    """
    Converts from lat lon to the grid space.
    Args:
        lon, lat (array-like): locations that you want to know the grid index of.
            Must be same length.
        grid_dir (str) : directory containing grid files
        grid_area (str) : e.g. 'CONTUS'
        zoom (int) : google zoom level
        numPixels (int) : number of pixels per tile
    Returns:
        :py:class:`pandas.DataFrame`: i, j values of supplied points
    """

    assert len(lon) == len(lat), "Your lon and lat arrays are different dimensions."

    # load the grid
    grid = np.load(
        join(grid_dir, "grid_{}_{}_{}.npz".format(grid_area, zoom, numPixels))
    )

    # turn lats and lons from full grid into dataframe with lat/lon as index and i/j as
    # value
    lons = (
        pd.DataFrame(grid["lon"], columns=["lon"])
        .reset_index()
        .set_index("lon")
        .rename(columns={"index": "j"})
    )
    lats = (
        pd.DataFrame(grid["lat"], columns=["lat"])
        .reset_index()
        .set_index("lat")
        .rename(columns={"index": "i"})
    )

    # adjust for 1-indexed i and j
    lons += 1
    lats += 1

    # need to flip lats b/c it's monotonic decreasing
    lats = lats.sort_index()

    # confirm that lons are sorted
    assert (lons == lons.sort_index()).all().all()

    # get both lon and lat in one dataframe
    data_df = pd.DataFrame({"lon": np.array(lon), "lat": np.array(lat)})

    # merge on i and j
    data_df = data_df.sort_values(by="lon")
    data_df["j"] = pd.merge_asof(
        data_df[["lon"]], lons, left_on="lon", direction="nearest", right_index=True
    ).loc[:, "j"]

    data_df = data_df.sort_values(by="lat")
    data_df["i"] = pd.merge_asof(
        data_df[["lat"]], lats, left_on="lat", direction="nearest", right_index=True
    ).loc[:, "i"]

    # correct for values outside of grid
    min_lat = lats.index[0] - (lats.index[1] - lats.index[0]) / 2
    max_lat = lats.index[-1] + (lats.index[-1] - lats.index[-2]) / 2
    min_lon = lons.index[0] - (lons.index[1] - lons.index[0]) / 2
    max_lon = lons.index[-1] + (lons.index[-1] - lons.index[-2]) / 2

    data_df = data_df.where(
        (data_df["lat"] <= max_lat)
        & (data_df["lat"] >= min_lat)
        & (data_df["lon"] <= max_lon)
        & (data_df["lon"] >= min_lon)
    )

    # get data back in original order
    result = data_df.sort_index()[["i", "j"]].values

    return result
