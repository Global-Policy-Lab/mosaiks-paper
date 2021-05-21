import os
import pickle

import matplotlib as mpl
import numpy as np
import seaborn as sns
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from matplotlib import ticker
from mosaiks import config as c
from mosaiks.plotting.general_plotter import scatter_preds
from mosaiks.utils.io import get_us_from_shapefile

mpl.rcParams["pdf.fonttype"] = 42


def setup_plotting_context(scale):
    sns.set(
        context="talk",
        style="white",
        palette="colorblind",
        font_scale=scale / 2,
        rc={
            "axes.linewidth": 1.0 * scale,
            "xtick.major.width": 1.0 * scale,
            "xtick.minor.width": 0.5 * scale,
            "ytick.major.width": 1.0 * scale,
            "ytick.minor.width": 0.5 * scale,
            "xtick.major.size": 4.5 * scale,
            "lines.linewidth": 0.75 * scale,
        },
    )

    return sns.plotting_context()


def plot_figure_2(
    tasks, data_by_task, marker_scale=0.4, scale=3, plot_error=False, is_ACS=False
):
    """
    plots figure 2 from the main text.
    input:
        tasks: list of task names to consider
        data_by_task: formatted lists of data as specified by the output of
            'aggregrate_and_bin_data' (below)
    returns:
        (none) plots data.
    """
    # unpack data
    truth_by_task = data_by_task["truth_by_task"]
    preds_by_task = data_by_task["preds_by_task"]
    lon_points_by_task = data_by_task["lon_points_by_task"]
    lat_points_by_task = data_by_task["lat_points_by_task"]
    truth_binned_by_task = data_by_task["truth_binned_by_task"]
    preds_binned_by_task = data_by_task["preds_binned_by_task"]
    if is_ACS:
        bounds_by_task = data_by_task["bounds_by_task"]

    num_tasks = len(tasks)
    # set up the figure with sizes
    num_plot_types = 3
    fig_width = 7.2 * scale
    fig_height = 2.0 * num_tasks * scale
    figsize = (fig_width, fig_height)

    fig = plt.figure(figsize=figsize)

    # relative figure sizes
    gs = fig.add_gridspec(
        num_tasks, num_plot_types, width_ratios=[1, 1, 0.4], wspace=0.01, hspace=0.05
    )

    pc = setup_plotting_context(scale)

    mpl.rcParams["pdf.fonttype"] = 42

    # plotting config variables
    pa = c.plotting
    c_by_app = [getattr(c, i) for i in c.app_order]
    disp_names = [config["disp_name"] for config in c_by_app]
    units = [config["units_disp"] for config in c_by_app]

    colors = [config["color"] for config in c_by_app]
    cmap_fxn = pa["cmap_fxn"]
    cmaps = [cmap_fxn(color) for color in colors]

    scatter_bounds = pa["scatter_bounds"]
    cmap_bounds = pa["cmap_bounds"]
    cbar_extend = pa["cbar_extend"]
    pa["bg_color"] = "lightgrey"

    # get bounds for us
    us = get_us_from_shapefile(simplify=0.1)

    for t in range(num_tasks):
        app = tasks[t]
        ## get colormap/scatter bounds
        if is_ACS:
            scatter_bounds_this = bounds_by_task[t][0]
            cmap_bounds_this = bounds_by_task[t][0]
        else:
            scatter_bounds_this = scatter_bounds[app]
            cmap_bounds_this = cmap_bounds[app]

        ### Make the maps:
        ax_truth = fig.add_subplot(gs[t, 0], projection=ccrs.PlateCarree())
        ax_truth.outline_patch.set_visible(False)
        ax_truth.background_patch.set_visible(False)
        ax_truth.add_geometries(
            [us],
            crs=ccrs.PlateCarree(),
            facecolor=pa["bg_color"],
            edgecolor="none",
            zorder=-100,
        )

        sc_truth = ax_truth.pcolormesh(
            lat_points_by_task[t],
            lon_points_by_task[t],
            truth_binned_by_task[t],
            cmap=cmaps[t],
            vmin=cmap_bounds_this[0],
            vmax=cmap_bounds_this[1],
            edgecolors="none",
        )

        ax_truth.text(
            0,
            0.6,
            disp_names[t].replace(" ", "\n"),
            va="bottom",
            ha="center",
            rotation="vertical",
            rotation_mode="anchor",
            transform=ax_truth.transAxes,
            weight="bold",
        )

        # set up axes
        ax_pred = fig.add_subplot(gs[t, 1], projection=ccrs.PlateCarree())
        ax_pred.outline_patch.set_visible(False)
        ax_pred.background_patch.set_visible(False)
        ax_pred.add_geometries(
            [us],
            crs=ccrs.PlateCarree(),
            facecolor=pa["bg_color"],
            edgecolor="none",
            zorder=-100,
        )
        if t == 0:
            ax_truth.set_title("Labels", weight="bold")
            ax_pred.set_title("Predictions", weight="bold")
            if plot_error:
                ax_truth.set_title("Labels", weight="bold")
                ax_pred.set_title("Prediction Errors", weight="bold")

        # If not plotting error, then the right column is the mosaiks predictions
        if not plot_error:
            ## plot preds
            sc_pred = ax_pred.pcolormesh(
                lat_points_by_task[t],
                lon_points_by_task[t],
                preds_binned_by_task[t],
                cmap=cmaps[t],
                vmin=cmap_bounds_this[0],
                vmax=cmap_bounds_this[1],
                edgecolors="none",
            )
        # If we are plotting errors, then the right column is model error. Here, the
        # name is kept as ax_pred for compatibility with future lines.
        else:
            ## plot preds
            cmap_brownteal = sns.diverging_palette(
                53, 188, s=90, l=70, sep=25, center="light", as_cmap=True
            )
            diverging_palette = cmap_brownteal
            mask_diff = (
                preds_binned_by_task[t] - truth_binned_by_task[t]
            )  # this looks good.

            sc_pred = ax_pred.pcolormesh(
                lat_points_by_task[t],
                lon_points_by_task[t],
                mask_diff,
                cmap=diverging_palette,  # makes it teal and brown
                vmin=-mask_diff.std(),  # sets the bounds for the color scales
                vmax=mask_diff.std(),
                edgecolors="none",
            )

        ### Make the scatter plots of predicted and observed
        if not plot_error:
            ## scatter preds and obs
            ax_scatter = fig.add_subplot(gs[t, 2])
            ax_scatter = scatter_preds(
                preds_by_task[t],
                truth_by_task[t],
                app,
                ax=ax_scatter,
                bounds=scatter_bounds_this,
                c="dimgrey",
                s=marker_scale * (scale ** 2),
                linewidth=pc["lines.linewidth"],
                fontsize=pc["font.size"] * 1,
                rasterize=True,
                despine=True,
                axis_visible=True,
                is_ACS=is_ACS,
            )

            # clip the scatter plot at bounds specified by the config file
            min_point = scatter_bounds_this[0]
            if min_point is None:
                min_point = np.min(np.hstack((preds_by_task[t], truth_by_task[t])))
            max_point = scatter_bounds_this[1]
            if max_point is None:
                max_point = np.max(np.hstack((preds_by_task[t], truth_by_task[t])))

            # format tick marks on the scatter plot to show the bounds of colormaps
            # of the left two plots with minor_ticks.
            major_ticks = [max_point, min_point]
            minor_ticks = []
            if not cmap_bounds_this[0] is None:
                minor_ticks.append(cmap_bounds_this[0])
            if not cmap_bounds_this[1] is None:
                minor_ticks.append(cmap_bounds_this[1])

            def tick_formatter(x, pos):
                if x == 0 or x == 100:
                    return str(int(x))
                if abs(x) < 10:
                    return f"{x:.1f}"
                if abs(x) < 1000:
                    return str(int(x))
                if abs(x) < 100000:
                    return f"{x/1000:.1f}k"
                return str(int(x / 1000)) + "k"

            ax_scatter.xaxis.set_major_locator(mpl.ticker.FixedLocator(major_ticks))
            ax_scatter.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(tick_formatter)
            )
            ax_scatter.yaxis.set_major_locator(mpl.ticker.FixedLocator(major_ticks))

            ax_scatter.xaxis.set_minor_locator(mpl.ticker.FixedLocator(minor_ticks))
            ax_scatter.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minor_ticks))
            # major
            ax_scatter.tick_params(
                axis="x",
                which="major",
                direction="out",
                bottom=True,
                length=5,
                color="black",
            )
            # minor
            ax_scatter.tick_params(
                axis="x",
                which="minor",
                direction="in",
                bottom=True,
                length=5,
                color="black",
            )
            ax_scatter.tick_params(
                axis="y",
                which="minor",
                direction="in",
                left=True,
                length=5,
                color="black",
            )
            ax_scatter.yaxis.set_ticklabels([])

            sns.despine(ax=ax_scatter, left=False, bottom=False)

        ### Make C-Axis:
        # Observations and predictions share the same c-axis so make one big one:
        ## colorbar for the first two
        bb_truth = ax_truth.get_position()
        bb_pred = ax_pred.get_position()
        height = bb_truth.height * 0.05
        width = (bb_pred.x1 - bb_truth.x0) * 0.95
        # Need to have a smaller c-axis for the error plot
        if plot_error:
            width = (bb_pred.x1 - bb_pred.x0) * 0.95
        y0 = bb_truth.y0 - height
        x0 = bb_truth.x0 + width * 0.025
        ax_cbar = fig.add_axes((x0, y0, width, height))
        cb = fig.colorbar(
            sc_truth, cax=ax_cbar, orientation="horizontal", extend=cbar_extend[app]
        )
        cb.locator = ticker.MaxNLocator(nbins=6, integer=True)
        cb.update_ticks()
        ax_cbar.set_xlabel(units[t], labelpad=1.0, weight="bold")

        # If you are plotting error then we need a separate c-axis for the truth and the
        # error
        if plot_error:
            ## colorbar for the error
            bb_diff = ax_pred.get_position()
            height = bb_diff.height * 0.05
            width = (bb_diff.x1 - bb_diff.x0) * 0.95
            y0 = bb_diff.y0 - height
            x0 = bb_diff.x0 + width * 0.025
            ax_cbar2 = fig.add_axes((x0, y0, width, height))
            # Plots COLOR BAR IN FIGURE
            fig.colorbar(sc_pred, cax=ax_cbar2, orientation="horizontal", extend="both")
            cb.locator = ticker.MaxNLocator(nbins=6, integer=True)
            cb.update_ticks()
            ax_cbar2.set_xlabel(units[t], labelpad=1.0, weight="bold")

    return fig


def points_to_bin(x, y, vals, scale=10.0):
    """bins points over 2d space with bin sizes specified by scale
     args:
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
    counts, _, _ = np.histogram2d(y, x, bins=bin_shapes)
    vals_grid = sums_grid / counts
    vals_grid = np.ma.masked_invalid(vals_grid)

    return x0, y0, vals_grid


def aggregrate_and_bin_data(agg_scale=0.2, is_ACS=False):
    """Aggregated labels from the saved output of the primary analysis notebooks.
    Aggregate to 'agg_scale' for vizualization"""
    c_by_app = [getattr(c, i) for i in c.app_order]

    tasks = [config["application"] for config in c_by_app]
    num_tasks = len(tasks)
    variables = [config["variable"] for config in c_by_app]
    sample_types = [config["sampling"] for config in c_by_app]

    # get file paths for data
    file_name_template = (
        "outcomes_scatter_obsAndPred_{0}_{1}_CONTUS_16_640_{2}_100000_0_random_features"
        "_3_0.data"
    )
    file_names_by_task = [
        file_name_template.format(tasks[i], variables[i], sample_types[i])
        for i in range(len(tasks))
    ]
    path_name_template = os.path.join(
        c.out_dir, "applications/{0}/figures/primary_analysis/{1}"
    )
    file_paths_local = [
        path_name_template.format(tasks[i], file_names_by_task[i])
        for i in range(len(tasks))
    ]

    # store aggregated data in lists
    lat_points_by_task, lon_points_by_task = [], []
    truth_binned_by_task, preds_binned_by_task = [], []
    truth_by_task, preds_by_task = [], []
    bounds_by_task = []  # For ACS

    for t in range(num_tasks):
        # grab the entire data
        f = file_paths_local[t]
        with open(f, "rb") as file_this:
            data_this = pickle.load(file_this)
        truth = data_this["truth"]
        preds = data_this["preds"]
        # store unbinned data
        truth_by_task.append(truth)
        preds_by_task.append(preds)
        # store bounds for ACS
        if is_ACS:
            # bounds_by_task.append(data_this["bounds"])
            # Set better bounds for display
            # print(data_this["bounds"])
            boundMin = np.amin([np.amin(truth), np.amin(preds)])
            boundMax = np.amax([np.amax(truth), np.amax(preds)])
            # print([np.array([boundMin,boundMax])])
            bounds_by_task.append([np.array([boundMin, boundMax])])

        # aggregate the data into averaged bins
        lat_points, lon_points, truth_binned = points_to_bin(
            data_this["lon"], data_this["lat"], truth, scale=agg_scale
        )
        _, _, preds_binned = points_to_bin(
            data_this["lon"], data_this["lat"], preds, scale=agg_scale
        )

        # store binned data
        lat_points_by_task.append(lat_points)
        lon_points_by_task.append(lon_points)
        truth_binned_by_task.append(truth_binned)
        preds_binned_by_task.append(preds_binned)

    if is_ACS:
        return {
            "truth_by_task": truth_by_task,
            "preds_by_task": preds_by_task,
            "lat_points_by_task": lat_points_by_task,
            "lon_points_by_task": lon_points_by_task,
            "truth_binned_by_task": truth_binned_by_task,
            "preds_binned_by_task": preds_binned_by_task,
            "bounds_by_task": bounds_by_task,
        }
    else:
        return {
            "truth_by_task": truth_by_task,
            "preds_by_task": preds_by_task,
            "lat_points_by_task": lat_points_by_task,
            "lon_points_by_task": lon_points_by_task,
            "truth_binned_by_task": truth_binned_by_task,
            "preds_binned_by_task": preds_binned_by_task,
        }
