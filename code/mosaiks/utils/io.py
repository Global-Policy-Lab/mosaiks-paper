import io
from os.path import join
from pathlib import Path

import dask.array as da
import dill
import imageio
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from cartopy.io import shapereader
from sklearn.metrics import r2_score
from torch import cuda

from .. import config as cfg
from .. import transforms
from . import spatial

GPU = cuda.is_available()


def get_suffix(c, appname):
    """Return the standard filename suffix that we are using to
    keep track of different model settings.

    Args:
        c (config module): The module you get when importing config.py
        appname (str): The name of the application (e.g. housing)
    Returns:
        str: The sub-suffix representing grid-specific settings
        str: The sub-suffix representing sample-specific settings.
            grid_str + '_' + sample_str is the full suffix you will
            need, but sometimes you need them separately as well.
    """

    c_grid = c.grid
    c_smpl = c.sampling
    c_img = c.images
    c_app = getattr(c, appname)

    grid_str = "{}_{}_{}".format(c_grid["area"], c_img["zoom_level"], c_img["n_pixels"])
    sample_str = "{}_{}_{}".format(
        c_app["sampling"], c_smpl["n_samples"], c_smpl["seed"]
    )

    return grid_str, sample_str


def get_filepaths(c, app, feattype="random", is_ACS=False):
    grid_str, smpl_str = get_suffix(c, app)
    c.data_suffix = grid_str + "_" + smpl_str
    c.full_suffix = (
        c.data_suffix
        + "_"
        + c.features[feattype]["type"]
        + "_"
        + str(c.features[feattype]["patch_size"])
        + "_"
        + str(c.features[feattype]["seed"])
    )

    # Used for spatial experiments in Figure 2
    c.full_suffix_image = c.full_suffix
    c.full_suffix_latlon = c.full_suffix_image.replace(
        "random_features", "latlonRBF_features"
    )

    c.fig_dir = join(c.out_dir, "applications", app, "figures")
    c.model_dir = join(c.out_dir, "applications", app, "models")

    # Figure locations full paths
    c.fig_dir_diag = join(c.fig_dir, "diagnostics")
    c.fig_dir_prim = join(c.fig_dir, "primary_analysis")
    c.fig_dir_sec = join(c.fig_dir, "secondary_analysis")

    # File path to labels
    if is_ACS:
        c.outcomes_fpath = join(
            c.data_dir,
            "int",
            "applications",
            "ACS",
            app,
            "outcomes_sampled_{}_{}.csv".format(app, c.data_suffix),
        )
    else:
        c.outcomes_fpath = join(
            c.data_dir,
            "int",
            "applications",
            app,
            "outcomes_sampled_{}_{}.csv".format(app, c.data_suffix),
        )

    return c


def get_X_latlon(c, sampling_type):
    """Get random features matrices for the main (CONUS) analysis.

    Parameters
    ----------
    c : :module:`mosaiks.config`
        Config object
    sampling_type : "UAR" or "POP"
        The sample that you want images for

    Returns
    -------
    X : :class:`pandas.DataFrame`
        100000 x 8192 array of features, indexed by i,j ID
    latlons :class:`pandas.DataFrame`
        100000 x 2 array of latitudes and longitudes, indexed by i,j ID
    """

    # Load the feature matrix locally
    local_path = join(
        c.features_dir,
        f"{c.grid['area']}_{sampling_type}.pkl",
    )
    with open(local_path, "rb") as f:
        arrs = dill.load(f)
    X = pd.DataFrame(
        arrs["X"].astype(np.float64),
        index=arrs["ids_X"],
        columns=["X_" + str(i) for i in range(arrs["X"].shape[1])],
    )

    # get latlons
    latlons = pd.DataFrame(arrs["latlon"], index=arrs["ids_X"], columns=["lat", "lon"])

    # sort both
    latlons = latlons.sort_values(["lat", "lon"], ascending=[False, True])
    X = X.reindex(latlons.index)

    return X, latlons


def get_Y(c, y_labels, ACS=False):
    """Load one or more ground truth variables from a single application"""
    fpath = Path(c.outcomes_fpath)

    if ACS:
        # need to add an ACS layer in the filepath
        fpath = fpath.parent.parent / "ACS" / fpath.parent.name / fpath.name
    return pd.read_csv(fpath, index_col="ID")[y_labels].sort_index()


def get_multiple_Y(
    c, labels=cfg.app_order, allow_logs=True, sampling_dict={}, colname_dict={}
):
    """Return a DataFrame containing the labels specified in ``labels``, and containing
    only the grid cells with non-null values across all labels. Note that this should
    typically only be run for labels that share a sampling pattern, because there is
    little overlap in grid cells between POP and UAR samples.

    Parameters
    ----------
    c : :module:`mosaiks.config`
        MOSAIKS config object
    labels : iterable of str
        Labels to pull Y values for
    allow-logs : bool
        If True, log all of the variables that are logged in the main analysis
        (according to the config module)
    sampling_dict : dict
        Override the sampling strategy (UAR vs POP) in the config file for these labels.
        e.g. ``{"housing": "UAR"}``
    colname_dict : dict
        Override the column name pulled from the label data csv.
        e.g. ``{"nightlights": "luminosity"}``.

    Returns
    -------
    :class:`pandas.DataFrame`
        The dataframe of values for all labels
    """
    dfs_y = []
    for t in labels:

        # adjust values if provided
        c_app = getattr(c, t)
        c_app["sampling"] = sampling_dict.get(t, c_app["sampling"])
        c_app["colname"] = colname_dict.get(t, c_app["colname"])

        # load filepaths for this task
        c = get_filepaths(c, t)

        this_Y = get_Y(c, c_app["colname"])

        # drop null
        this_Y, _ = transforms.dropna_Y(this_Y, t)

        # transform if logged
        # using this_Y multiple times due to function signature
        if allow_logs and c_app["logged"]:
            logged = True
        else:
            logged = False
        _, this_Y, _ = getattr(transforms, f"transform_{t}")(
            this_Y, this_Y, this_Y, logged
        )

        dfs_y.append(this_Y)

    return pd.concat(dfs_y, axis=1, join="inner")


def load_superres_X(c):
    """Load MOSAIKS features from superresolution-structured features file.

    Parameters
    ----------
    c : :module:`mosaiks.config`

    Returns
    -------
    X, latlons : :class:`numpy.ndarray`
        Features (N, K) and locations (N, 2) for images
    net_pred : :class:`mosaiks.featurization.BasicCoatesNgNet`
        Featurization object used to create activation maps
    """
    pool_stride = c.superres["pool_stride"]

    # load the intermediate representation of the features.
    with open(join(c.features_dir, c.superres["features_fname"]), "rb") as f:
        data = dill.load(f)

    # get the X matrices
    features = data["X"].astype("float64")
    ids = data["ids_X"]
    latlons = data["latlon"]
    colnames = [f"X_{i}" for i in range(features.shape[1])]
    X = pd.DataFrame(features, index=ids, columns=colnames)
    latlons = pd.DataFrame(latlons, index=ids, columns=["lat", "lon"])

    # get other necessary objects
    net_pred = data["net"]

    net_pred.pool_size = pool_stride
    net_pred.pool_stride = pool_stride

    return X, latlons, net_pred


########################
# Imagery
########################


def load_img_from_ids_local(img_id, image_dir, c=cfg):
    """Load image from local directory, referenced by ID. If you don't know the length
    of the lat and lon strings used to define the filepath, you need to look by ID. This
    function searches over various numbers of sig-figs to find the correct file.

    Parameters
    ----------
    img_id : str
        i,j-style location of the grid cell for which you want an image returned.
    image_dir : str or :class:`pathlib.Path`, optional
        Path to folder where images will be found
    c : :module:`mosaiks.config`, optional
        Config object

    Returns
    -------
    :class:`numpy.ndarray`
        The image
    """
    zoom = c.images["zoom_level"]
    n_pixels = c.images["n_pixels"]
    ll = spatial.ids_to_ll(
        [img_id], c.grid_dir, c.grid["area"], zoom=zoom, numPixels=n_pixels
    )
    ll = [i[0] for i in ll]

    # first try with whatever sig-figs get assigned when converting float to str
    try:
        return load_img_from_local(
            ll[::-1], image_dir=image_dir, zoom=zoom, pix=n_pixels
        )
    except FileNotFoundError:
        pass

    # now try for different permutations of sig figs
    for i in range(15, 11, -1):
        for j in range(15, 11, -1):
            this_ll = [f"{ll[1]:.{i}f}", f"{ll[0]:.{j}f}"]
            try:
                return load_img_from_local(
                    this_ll, image_dir=image_dir, zoom=zoom, pix=n_pixels
                )
            except FileNotFoundError:
                pass
    raise FileNotFoundError


def load_img_from_local(
    latlon,
    image_dir=Path(cfg.data_dir) / "raw" / "imagery" / "CONTUS_UAR",
    zoom=cfg.images["zoom_level"],
    pix=cfg.images["n_pixels"],
):
    """Load image from a local directory, referenced by lat/lon.

    Parameters
    ----------
    latlon : array-like (2,)
        Latitude and longitude of image, with same number of sig-figs as are used in
        filename of image
    image_dir : str or :class:`pathlib.Path`, optional
        Path to image directory

    Returns
    -------
    class:`numpy.ndarray`
        The image
    """
    fpath = generate_key_name(latlon, image_dir, zoom, pix)
    return imageio.imread(fpath)


def generate_key_name(latlon, image_dir, zoom, pix):
    lat, lon = latlon[0], latlon[1]
    outkey = f"{image_dir}/{lat}_{lon}_{zoom}_{pix}_{pix}.png"
    return outkey


def get_lambdas(
    c,
    app,
    lambda_name="lambdas",
    best_lambda_name="best_lambda",
    best_lambda_fpath=None,
):
    """Return the lambdas in :module:`mosaiks.config` unless ``fixed_lambda`` is True,
    in which case return a 1-element array that is the previously chosen best lambda.
    """
    c = get_filepaths(c, app)
    if best_lambda_fpath:
        return np.unique(
            np.asarray(
                np.load(
                    best_lambda_fpath,
                    allow_pickle=True,
                )[best_lambda_name]
            )
        )
    return getattr(c, app)[lambda_name]


def gpu_return_and_clear(x):
    if GPU:
        import cupy as cp

        mempool = cp.get_default_memory_pool()

        if isinstance(x, da.Array):
            x = da.map_blocks(cp.asnumpy, x)
        else:
            x = cp.asnumpy(x)
        mempool.free_all_blocks()
    return x


########################
# Shapefiles
########################


def get_us_from_shapefile(border=False, simplify=None):
    shapefile_dir = join(
        cfg.root_dir,
        "data/raw/shapefiles/gadm36_USA_shp/gadm36_USA_0.shp",
    )
    rdr = shapereader.Reader(shapefile_dir)
    us = list(rdr.geometries())[0]
    if simplify is not None:
        us = us.simplify(simplify)
    if border:
        us_border = cfeature.ShapelyFeature(
            us, crs=ccrs.PlateCarree(), facecolor="None", edgecolor="k"
        )
        us = (us, us_border)
    return us


########################
# CNN
########################


def load_cnn_performance(task, c, extra_return=None):
    with open(
        Path(c.data_dir) / "output" / "cnn_comparison" / f"resnet18_{task}.pickle", "rb"
    ) as f:
        data_this = dill.load(f)

    # for housing, have to deal with the fact that cnn performance on publicly released
    # subset will be different. So we subset to the available outcomes and then
    # calculate r2
    c = get_filepaths(c, task)
    valid_labels = get_Y(c, getattr(c, task)["colname"]).index
    y_test = pd.DataFrame(
        {
            "pred": data_this["y_test_pred"].squeeze(),
            "obs": data_this["y_test"].squeeze(),
        },
        index=data_this["ids_test"],
    )
    y_test = y_test[y_test.index.isin(valid_labels)]
    out = r2_score(y_test.obs, y_test.pred)
    if extra_return is not None:
        out = [out] + [data_this[i] for i in extra_return]
    return out
