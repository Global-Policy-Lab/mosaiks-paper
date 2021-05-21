from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
import shapely as shp
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt
from matplotlib import rcParams
from mosaiks import config as c
from mosaiks.utils.io import get_us_from_shapefile

suffix = (
    f"{{app}}_{{var}}_{c.grid['area']}_{c.images['zoom_level']}_{c.images['n_pixels']}_"
    f"{{sampling}}_{c.sampling['n_samples']}_{c.sampling['seed']}_"
    f"{c.features['random']['type']}_{c.features['random']['patch_size']}_"
    f"{c.features['random']['seed']}"
)

checkerboard_str = "checkerboardJitterInterpolation_" + suffix

YTICKS = [0, 0.25, 0.5, 0.75, 1]
YTICKLABELS = ["0", "", "0.5", "", "1"]


def setup_plotting_context(scale):
    sns.set(
        context="paper",
        style="ticks",
        palette="colorblind",
        font_scale=scale,
        rc={
            "lines.linewidth": 0.75,
            "axes.linewidth": 0.75,
            "font.size": 6.0 * scale,
            "xtick.major.size": 3.6,
            "xtick.minor.size": 2.4,
            "ytick.major.size": 3.6,
            "ytick.minor.size": 2.4,
        },
    )


def get_app_data_path(app, filename):
    c_app = getattr(c, app)

    cb_path_base = join(
        c.out_dir,
        "applications",
        app,
        "figures",
        "secondary_analysis",
        filename + ".data",
    )
    return cb_path_base.format(
        app=app, var=c_app["variable"], sampling=c_app["sampling"]
    )


def load_app_data(app, filename, **kwargs):
    return np.load(get_app_data_path(app, filename, **kwargs), allow_pickle=True)


def get_checkerboard_geoms(delta, us, minx, miny, maxx, maxy):
    b = miny
    l = minx
    t = b + delta
    r = l + delta

    cells = []
    col = 0
    row = 0

    while (r < maxx) or (t < maxy):
        ctr = col + row
        if ctr % 2:
            fc = "k"
        else:
            fc = "w"
        geom = shp.geometry.box(l, b, r, t).intersection(us)
        if geom.bounds != ():
            cells.append(
                cfeature.ShapelyFeature(
                    [geom],
                    facecolor=fc,
                    edgecolor=None,
                    linewidth=0,
                    crs=ccrs.PlateCarree(),
                )
            )
        l = r
        col += 1

        if l >= maxx:
            l = minx
            b = t
            row += 1
            col = 0

        t = b + delta
        r = l + delta

    return cells


def plot_checkerboard_maps(gs_placement, axs_placment):

    # get bounding box and rearrange bound order
    extent = c.plotting["extent"]
    minx, miny, maxx, maxy = extent[2], extent[0], extent[3], extent[1]
    maxy += (
        2  # needed so the top of the US doesn't get cutoff in checkerboard demo plots
    )
    extent = (minx, miny, maxx, maxy)

    # get deltas
    deltas = (
        np.array(c.checkerboard["deltas"]) * 2
    )  # convert from list to array and double to get cell width
    delta_min = deltas[0]
    delta_mid = 8
    delta_max = deltas[-1]

    us, us_border = get_us_from_shapefile(border=True, simplify=0.01)

    # get polygons for cells
    cells_min = get_checkerboard_geoms(delta_min, us, *extent)
    cells_mid = get_checkerboard_geoms(delta_mid, us, *extent)
    cells_max = get_checkerboard_geoms(delta_max, us, *extent)

    cells_plot = [cells_min, cells_mid, cells_max]
    deltas_plot = [delta_min, delta_mid, delta_max]

    for cells_ix, cells in enumerate(cells_plot):

        ax = plt.subplot(gs_placement[cells_ix], projection=ccrs.PlateCarree())
        ax.set_extent((minx, maxx, miny, maxy), crs=ccrs.PlateCarree())

        for cell in cells:
            ax.add_feature(cell)
        ax.add_feature(us_border)
        ax.outline_patch.set_visible(False)

        ## add titles
        ax.text(
            0.5,
            -0.1,
            "$\delta = {:.1f}\degree$".format(deltas_plot[cells_ix]),
            va="top",
            ha="center",
            rotation="horizontal",
            rotation_mode="anchor",
            fontsize=rcParams["axes.labelsize"],
            transform=ax.transAxes,
        )
        axs_placment.append(ax)

    # legend
    ax = plt.subplot(gs_placement[-1])
    ax.axis("off")
    train_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor="black", edgecolor="black")
    test_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black")
    labels = ["Training set", "Validation set"]
    ax.legend(
        [train_patch, test_patch],
        labels,
        loc="center",
        fontsize="medium",
        frameon=False,
        handlelength=1,
        handleheight=1,
    )
    axs_placment.append(ax)


def plot_model_diagnostic_lineplots(gs_placement, axs_placement):

    r2_ftr_str = "r2_score_vs_featsize_" + suffix
    r2_trn_str = "r2_score_vs_trainsize_" + suffix

    prefixes = [r2_ftr_str, r2_trn_str]

    for col, t in enumerate(["Feature Vector Size (K)", "Training Set Size (N)"]):
        plot_handles = []
        fill_handles = []
        labs = []
        maxxval = 0
        minxval = np.inf
        ax = plt.subplot(gs_placement[col])
        for app in c.app_order:
            c_app = getattr(c, app)
            title = c_app["disp_name"]
            vals = load_app_data(app, prefixes[col])
            x = vals["x_vals"]
            maxxval = max(maxxval, max(x))
            minxval = min(minxval, min(x))
            y = np.array(vals["y_vals"])
            y_mean = y.mean(axis=0)
            y_min = y.min(axis=0)
            y_max = y.max(axis=0)

            p0 = ax.semilogx(x, y_mean, label=title, color=c_app["color"])
            p1 = ax.fill_between(
                x, y_min, y_max, color=c_app["color"], alpha=0.5, linewidth=0
            )
            plot_handles.append(p0[0])
            fill_handles.append(p1)
            labs.append(title)
            ax.set_xlabel(t)
            ax.set_ylim(-0.1, 1)
            ax.set_yticks(YTICKS)
            ax.set_xlim(minxval, maxxval)
            sns.despine(ax=ax)
        axs_placement.append(ax)

    axs_placement[0].set_ylabel("$R^2$")
    axs_placement[0].set_yticklabels(YTICKLABELS)
    axs_placement[1].set_yticklabels([""] * len(YTICKS))

    # add legends for these plots
    handles = [(plot, fill_handles[ix]) for ix, plot in enumerate(plot_handles)]
    return handles


def plot_spatial_lineplots(gs_placement, axs_placement):

    deltas = (
        np.array(c.checkerboard["deltas"]) * 2
    )  # convert from list to array and double to get cell width
    delta_max = deltas[-1]
    # instantiate dataframe of checkerboard performance
    df = pd.DataFrame(
        index=deltas,
        columns=pd.MultiIndex.from_product([c.app_order, ["rand_feat", "rbf"]]),
    )
    df.index.name = "cell_width"
    idx = pd.IndexSlice

    # checkerboard performance plots
    xticks = [0.5, 4, 8, 12, 16]

    order = [(app, gs_placement[ix]) for ix, app in enumerate(c.app_order)]

    labs = []
    handles = []
    plot_handles = []
    fill_handles = []

    for aix, item in enumerate(order):
        app = item[0]
        c_app = getattr(c, app)
        title = c_app["disp_name"]

        ## axis config
        ax = plt.subplot(item[1])
        axs_placement.append(ax)
        ax.set_ylim(-0.1, 1)
        ax.set_xlim(0, delta_max)
        ax.set_xticks(xticks)
        ax.set_yticks(YTICKS)
        if aix in [4, 5, 6]:
            ax.set_xticklabels(xticks)
            ax.set_xlabel("$\delta (\degree)$")
        else:
            ax.set_xticklabels([""] * len(xticks))
        if aix in [0, 4]:
            ax.set_ylabel("$R^2$")
            ax.set_yticklabels(YTICKLABELS)
        else:
            ax.set_yticklabels([""] * len(YTICKS))

        # load the file
        data = load_app_data(app, checkerboard_str)
        r2_im = np.array(
            [[i["r2_score"] for i in j] for j in data["metrics"]["image_features"]]
        )
        r2_ll = np.array(
            [
                [i["r2_score"] for i in j]
                for j in data["metrics"]["latlon features sigma tuned"]
            ]
        )

        ## fill dataframe
        df.loc[:, idx[app, "rand_feat"]] = r2_im.mean(axis=1)
        df.loc[:, idx[app, "rbf"]] = r2_ll.mean(axis=1)

        ## add baseline dot
        r2_ftr_str = "r2_score_vs_featsize_" + suffix
        data_baseline = load_app_data(app, r2_ftr_str)
        baseline_r2 = np.array(data_baseline["y_vals"])[:, -1]
        r2_mean = np.mean(baseline_r2)
        ax.scatter([0], [r2_mean], marker=".", clip_on=False, color=c_app["color"])

        ## plot performance of the random features and the baseline RBF Kernel Smoother.
        for rx, r in enumerate(
            [
                (r2_im, c_app["color"], "Random Features", "-"),
                (r2_ll, "grey", "RBF Kernel Smoother", "--"),
            ]
        ):
            p0 = ax.plot(
                deltas, r[0].mean(axis=1), color=r[1], linestyle=r[3], label=r[2]
            )
            p1 = ax.fill_between(
                deltas,
                r[0].min(axis=1),
                r[0].max(axis=1),
                color=r[1],
                alpha=0.5,
                linewidth=0,
            )
            if rx == 0:
                plot_handles.append(p0[0])
                fill_handles.append(p1)
                labs.append(title)

        # put the dot on the line
        ax.plot([], [], marker=".", color="darkred")
        ax.fill_between([], [], [], color="darkred", alpha=0.5, linewidth=0)

        ax.set_title(title)
        sns.despine(ax=ax)

    # we end with the grey so now we add that to the legend
    plot_handles.append(p0[0])
    fill_handles.append(p1)
    labs.append(r[2])

    handles = [(plot_handles[ix], fill_handles[ix]) for ix in range(len(plot_handles))]

    ax = plt.subplot(gs_placement[-1])
    ax.axis("off")
    ax.legend(
        handles,
        labs,
        loc="lower center",
        fontsize="medium",
        bbox_to_anchor=(0, -0.3, 1, 1),
        frameon=False,
    )

    return None
