from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mosaiks import config
from mosaiks.utils import io
from skimage.transform import downscale_local_mean

TREECOVER_CMAP_BOUNDS = [0, 100]
POPULATION_CMAP_BOUNDS = [0, 1000]


def plot_both(
    id_to_match,
    data_tree,
    data_pop,
    true_tree,
    true_pop,
    img_crop,
    widths=[14, 28, 56, 112, 224],
    save_dir=None,
):
    """Plot image, scene-level label, and super-res predictions at multiple scales.

    Parameters
    ----------
    id_to_match : str ("int,int")
        The grid ID of the image to plot
    data_[tree,pop] : :class:`numpy.ndarray`
        The pixel-level superresolution predictions as arrays for treecover and
        population.
    true_[tree,pop] : float
        The scene-level labels for treecover and population
    img_crop : len-2 tuple
        The amount to crop the image along left/top and right/bottom dimensions. This is
        needed right, top, bottom. This is needed both because the feature extraction
        causes the predictions to be slightly smaller than the image (e.g. 256x256
        image --> 254x254 predictions with a 3x3 filter), and because we may have
        cropped the predictions to be divisible by each scale of superresolution factor
        that we are displaying.
    widths : list of int
        The widths (in pixels) of the superresolution predictions to make. The last
        element should be equal to the length of the dimensions of ``data_[tree,pop]``
    save_dir : str
        Path to directory where these images are saved. If None, do not save
    """
    # plotting contexts
    context = sns.plotting_context("paper", font_scale=2)
    lines = True
    style = {
        "axes.grid": False,
        "axes.edgecolor": "0.0",
        "axes.labelcolor": "0.0",
        "axes.spines.right": lines,
        "axes.spines.top": lines,
        "axes.spines.left": lines,
        "axes.spines.bottom": lines,
    }

    sns.set_context(context)
    sns.set_style(style)

    # set plotting context
    plot_bounds_tree = TREECOVER_CMAP_BOUNDS
    plot_bounds_pop = [0, 1000]
    plot_bounds = [plot_bounds_tree, plot_bounds_pop]
    names = ["treecover", "population"]
    names_disp = ["% Forest", "Pop. Dens."]

    # grab plotting constants from config file
    c_plotting = getattr(config, "plotting")
    cmap_fxn = c_plotting["cmap_fxn"]
    cmaps = [cmap_fxn(getattr(config, task)["color"]) for task in names]

    # get the image (returning blank if imagery is not saved)
    image_dir = Path(config.data_dir) / "raw" / "imagery" / "CONTUS_UAR"
    try:
        image_this = io.load_img_from_ids_local(id_to_match, image_dir, c=config)
    except FileNotFoundError:
        image_this = None

    for t, data_this in enumerate([(data_tree, true_tree), (data_pop, true_pop)]):
        pred_map = data_this[0]
        label_this = data_this[1]
        task_this = names[t]

        # get clipping bounds
        c_app = getattr(config, task_this)
        if c_app["logged"]:
            bounds = [np.exp(i) for i in c_app["us_bounds_log_pred"]]
        else:
            bounds = c_app["us_bounds_pred"]

        # collect maps by downscale level
        superres_preds_by_scale = []
        for w in widths:

            # downscale
            assert pred_map.shape[0] % w == 0
            this_preds = downscale_local_mean(pred_map, (w, w))

            # clip if both bounds aren't None for this outcome
            if not (np.asarray(bounds) == None).all():
                this_preds = np.clip(this_preds, *bounds)

            superres_preds_by_scale.append(this_preds)

        # plot for this variable
        cmap_this = cmaps[t]
        bounds_this = plot_bounds[t]
        plot_img_and_heatmap_and_preds_multiscale(
            image_this,
            superres_preds_by_scale,
            widths,
            label_this,
            cmap=cmap_this,
            vmin=bounds_this[0],
            vmax=bounds_this[1],
            name=names_disp[t],
        )
        # save with descriptive name
        if save_dir is not None:
            plt.savefig(
                "{2}/{1}_multires_{0}.pdf".format(id_to_match, names[t], save_dir),
                bbox_inches="tight",
            )


def plot_img_and_heatmap_and_preds_multiscale(
    image_0,
    preds_list,
    deltas_list,
    true_val,
    cmap="rocket",
    vmin=0,
    vmax=100,
    name=None,
):
    """
    plotting function for taking heatmaps and laying them out in a row with original
    image and colorbar
    """
    num_pred_maps = len(preds_list)
    fig, ax = plt.subplots(1, 2 + num_pred_maps, figsize=(5 * (2 + num_pred_maps), 5))

    # label first
    ax[0].imshow(np.array(true_val).reshape(1, 1), vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0].set_title("Label")

    if image_0 is not None:
        ax[1].imshow(image_0)
        ax[1].set_title("Image")

    for i in range(num_pred_maps):
        hm = ax[i + 2].imshow(preds_list[i], vmin=vmin, vmax=vmax, cmap=cmap)
        ax[i + 2].set_title("Predictions {0} x {0}".format(deltas_list[i]))

    [ax_this.set_axis_off() for ax_this in ax]

    cax = plt.axes([0.92, 0.2, 0.03, 0.6])
    plt.colorbar(hm, cax=cax)
    cax.set_title(name)

    for i in range(2 + num_pred_maps):
        # no grids
        ax[i].grid(False)
        # no numbers
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    return


def plot_sr_vs_delta(degrees_sr, widths, across_image_r2s_demeaned, save_path=None):
    context = sns.plotting_context("paper", font_scale=2)
    style = {
        "axes.grid": False,
        "axes.edgecolor": "0.0",
        "axes.labelcolor": "0.0",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
    }
    c_tree = getattr(config, "treecover")
    color_tree = c_tree["color"]

    sns.set_context(context)
    sns.set_style(style)
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.plot(
        degrees_sr,
        across_image_r2s_demeaned * 100,
        lw=2,
        label="Forest Cover",
        color=color_tree,
    )

    plt.scatter(degrees_sr, across_image_r2s_demeaned * 100, color=color_tree, s=64)

    plt.axhline(0, color="grey")
    plt.legend()

    # format the legends
    plt.legend()
    plt.xlabel("superRes width")
    plt.ylabel("% superRes label variance explained")
    plt.title("Sub-Image Prediction at Varying Resolutions")
    ax.set_xticks(ticks=degrees_sr)
    ax.set_xticklabels(labels=["{0}".format(s) for s in degrees_sr])
    sns.set_style({"xtick.bottom": True})

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax
