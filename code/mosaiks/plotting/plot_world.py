import glob
import os

import dill
import matplotlib.pyplot as plt
import mosaiks.config as c
import numpy as np
import pandas as pd
import seaborn as sns
from cartopy import crs as ccrs
from matplotlib import ticker

# plotting variables
cs = c.world_app_order
c_by_app = [getattr(c, i) for i in cs]

applications = [config["application"] for config in c_by_app]
variables = [config["variable"] for config in c_by_app]
sample_types = [config["sampling"] for config in c_by_app]
disp_names = [config["disp_name"] for config in c_by_app]
logged = [config["logged"] for config in c_by_app]
colorbar_bounds = [config["world_bounds_colorbar"] for config in c_by_app]
clip_bounds = [config["world_bounds_pred"] for config in c_by_app]

units = [config["units_disp"] for config in c_by_app]

c_plotting = getattr(c, "plotting")
colors = [config["color"] for config in c_by_app]
cmap_fxn = c_plotting["cmap_fxn"]

cmaps = [cmap_fxn(color) for color in colors]

# matches the order the data is stored in
task_to_data_idxs = {"treecover": 0, "elevation": 1, "population": 2, "nightlights": 3}
task_to_cfg_idxs = {"treecover": 0, "elevation": 1, "population": 2, "nightlights": 3}

# matches each task to the continent it falls into for the continent specific model
task_to_continent_idxs = {
    "treecover": 3,
    "elevation": 2,
    "population": 4,
    "nightlights": 2,
}


def plot_world_binned(
    latlons,
    y_true,
    y_pred,
    vmin,
    vmax,
    task_name="Title Me!",
    log_cbar=False,
    cmap_this="viridis",
    sub_select=False,
    show_coasts=True,
    agg_scale=10.0,
    units_this="units?",
    land_mass=None,
    proj=ccrs.Robinson(central_longitude=0),
):

    # parse data input
    if sub_select:
        sub_ids = np.random.choice(len(latlons), 5000)
        lls = latlons[sub_ids, :]
        preds = y_pred[sub_ids]
        labels = y_true[sub_ids].ravel()
    else:
        lls = latlons
        preds = y_pred
        labels = y_true.ravel()

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.01, hspace=0.05)

    x0, y0, labels_binned = points_to_bin(lls[:, 1], lls[:, 0], labels, scale=agg_scale)
    x0, y0, preds_binned = points_to_bin(lls[:, 1], lls[:, 0], preds, scale=agg_scale)

    ax_truth = fig.add_subplot(gs[0, 0], projection=proj)
    ax_truth.outline_patch.set_visible(False)
    ax_truth.background_patch.set_visible(False)
    ax_colorbar = ax_truth.pcolormesh(
        x0,
        y0,
        labels_binned,
        transform=ccrs.PlateCarree(),
        cmap=cmap_this,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )

    ax_pred = fig.add_subplot(gs[0, 1], projection=proj)
    ax_pred.outline_patch.set_visible(False)
    ax_pred.background_patch.set_visible(False)
    ax_pred.pcolormesh(
        x0,
        y0,
        preds_binned,
        transform=ccrs.PlateCarree(),
        cmap=cmap_this,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )

    if land_mass is not None:
        ax_pred.add_geometries(
            [land_mass],
            crs=ccrs.PlateCarree(),
            facecolor="grey",
            edgecolor="none",
            zorder=-100,
        )
        ax_truth.add_geometries(
            [land_mass],
            crs=ccrs.PlateCarree(),
            facecolor="grey",
            edgecolor="none",
            zorder=-100,
        )

    if show_coasts:
        ax_truth.coastlines(color="grey", linewidth=0.5)
        ax_pred.coastlines(color="grey", linewidth=0.5)

    ax_truth.set_title("Labels", fontsize=24)
    ax_pred.set_title("Predicted", fontsize=24)

    # colorbar for the first two
    bb_truth = ax_truth.get_position()
    bb_pred = ax_pred.get_position()
    height = bb_truth.height * 0.05
    width = (bb_pred.x1 - bb_truth.x0) * 0.95
    y0 = bb_truth.y0 - height
    x0 = bb_truth.x0 + width * 0.025
    ax_cbar = fig.add_axes((x0, y0, width, height))
    cb = fig.colorbar(ax_colorbar, cax=ax_cbar, orientation="horizontal")
    cb.locator = ticker.MaxNLocator(nbins=6, integer=True)

    cb.update_ticks()
    ax_cbar.set_xlabel(units_this, labelpad=1.0)
    return fig


def points_to_bin(x, y, vals, scale=10.0):
    """args:
     x,y: nx1 arrays of locations in 1 dimension each
     preds: nx1 array of values to be averaged
     scale: the edge of a bin/box in {x,y} units.
    returns:
     x0, y0: kx1, mx1 arrays of the x and y gridpoints
     vals_grid: (m-1)x(k-1) resulting aggregated values
    """
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    bin_shapes = [int(y_range / scale), int(x_range / scale)]

    sums_grid, y0, x0 = np.histogram2d(y, x, bins=bin_shapes, weights=vals)
    counts, y1, x1 = np.histogram2d(y, x, bins=bin_shapes)
    vals_grid = sums_grid / counts
    vals_grid = np.ma.masked_invalid(vals_grid)

    return x0, y0, vals_grid


def task_to_world_bin_plot(task, world_data, agg_scale):
    assert task in task_to_data_idxs.keys(), print(
        "task name not reconized, options are {0}".format(task_to_data_idxs.keys())
    )

    # grab data for this task
    this_idx_data = task_to_data_idxs[task]
    this_idx_config = task_to_cfg_idxs[task]

    latlons_this = world_data["latlons_train"]

    y_this = world_data["y_train"][:, this_idx_data]
    y_pred_this = world_data["y_pred_train_cross_val"][:, this_idx_data]

    proj = ccrs.Robinson(central_longitude=0)

    vmin_this, vmax_this = colorbar_bounds[this_idx_config]
    disp_name_this = disp_names[this_idx_config]
    cmap_this = cmaps[this_idx_config]
    units_this = units[this_idx_config]
    sns.set(
        rc={
            "axes.facecolor": "lightgrey",
            "figure.facecolor": "lightgrey",
            "axes.grid": False,
        }
    )
    fig = plot_world_binned(
        latlons_this,
        y_this,
        y_pred_this,
        vmin_this,
        vmax_this,
        sub_select=False,
        task_name=disp_name_this,
        cmap_this=cmap_this,
        units_this=units_this,
        agg_scale=agg_scale,
        show_coasts=True,
        land_mass=None,
        proj=proj,
    )

    return fig


def predict_y_dense_sample(task, world_wts, labels_to_run):

    # get features for each zoom
    feats = [None] * len(labels_to_run)
    for tt in labels_to_run:
        path = os.path.join(c.features_dir, "dense_" + tt + ".pkl")
        with open(path, "rb") as f:
            data = dill.load(f)
        tloc = task_to_data_idxs[tt]
        feats[tloc] = data["X"].astype("float64")

    # get weights estimated from continent model (from optimal hyperparameter)
    idx = task_to_data_idxs[task]

    # clip predictions for this task
    mylb = clip_bounds[idx][0]
    myub = clip_bounds[idx][1]

    ypreds = [None] * len(labels_to_run)

    # for each zoom, predict for this task
    for z in range(len(labels_to_run)):

        # this is the continent needed for this zoom
        zcont = task_to_continent_idxs[task]

        # get the right weights for this zoom and task
        mywts = world_wts["weights"][zcont][idx]

        # predictions
        ypreds[z] = np.dot(feats[z], mywts)
        ypreds[z][ypreds[z] < mylb] = mylb
        ypreds[z][ypreds[z] > myub] = myub

    return ypreds


def merge_zoompreds(zoom, labels_to_run, allpreds):
    fl = glob.glob(os.path.join(c.grid_dir, "*" + zoom + "*"))
    file = np.load(fl[0])

    # Create pandas dataframe from npz
    sampdf = pd.DataFrame(file["ID"])
    sampdf["lon"] = file["lon"]
    sampdf["lat"] = file["lat"]
    sampdf.columns = ["ID", "lon", "lat"]

    # which entry in allpreds[][] is this zoom?
    idz = task_to_data_idxs[zoom]

    for task in labels_to_run:

        # where is this task located in the task vector
        idx = task_to_data_idxs[task]

        # pull the predictions for this task and zoom
        sampdf[task] = allpreds[idx][idz]

    return sampdf
