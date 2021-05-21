import pickle

import matplotlib.pyplot as plt
import mosaiks.plotting.general_plotter as plots
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics


def checkerboard_vs_delta_with_jitter(
    metrics_checkered,
    best_hps,
    deltas,
    crit,
    val_name,
    app_name=None,
    save_dir=None,
    prefix=None,
    suffix=None,
    figsize=(10, 5),
    ax=None,
    overwrite=False,
):

    """Plot metrics (e.g. r2) or vs delta (degree of spatial extrapolation). Used for visualizing results of
    'checkerboard' analyses. Saves the result. Note this is for a single outcome. If you have multiple outcomes,
    you must obtain the proper metrics_checkered object for each outcome and pass to this function.

    Args:
     metrics_checkered (dict of list of dict) : e.g.
         {'random_features': [{'mse': 117.05561471285215, 'r2_score': 0.875037330527241},
                              {'mse': 119.84752736626068, 'r2_score': 0.8735189806862442}],
         'latlon_features':  [{'mse': 117.05561471285215, 'r2_score': 0.875037330527241},
                              {'mse': 119.84752736626068, 'r2_score': 0.8735189806862442}]}
        Keys of outer dict are labels for the lines that will be plotted. The list indexes different deltas,
        as produced by diagnostics.spatial_experiments.checkered_predictions_by_radius. Within each element,
        the keys of the dict are different metric names.
     best_hps (dict) : keys are ``best_lambda_rcf`` and ``best_sigma_smooth``. Values
         are the list of hyperparameters chosen for each value of ``delta``.
     deltas (list of numeric) : values (e.g. [1,2,3,4,5,6,7,8]) where larger values imply more spatial extrapolation
     crit (str) : Name of criteria that you want to plot (e.g. 'r2_score').
     val_name (str) : Name of outcome.
     app_name (str) : The name of the application (e.g. 'housing'). Only needed if saving
     save_dir (str) : Path to directory in which to save output files. If None, no figures will be saved.
     prefix (str) : Filename prefix identifying what is being plotted (e.g. test_outof_cell_r2). Only
         needed if figure is being saved
     suffix (str) : The suffix containing the grid and sample parameters which will be appended to the
         filename when saving, in order to keep track of various sampling and gridding schemes.
     overwrite (bool) : Whether to overwrite data and figure files

    Returns:
     None (plots function)

    """

    # Make the plot:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for feat in metrics_checkered.keys():
        metrics_checkered_this = metrics_checkered[feat]
        # Get the values by jitter into one list
        crit_by_delta = np.zeros(
            (len(metrics_checkered_this), len(metrics_checkered_this[0]))
        )
        for i in range(crit_by_delta.shape[0]):
            for j in range(crit_by_delta.shape[1]):
                crit_by_delta[i, j] = metrics_checkered_this[i][j][crit]

        # plot
        ax.plot(deltas, np.average(crit_by_delta, axis=1), label=feat)
        ax.fill_between(
            deltas,
            np.min(crit_by_delta, axis=1),
            np.max(crit_by_delta, axis=1),
            alpha=0.5,
        )

    ax.set_title("Performance vs. checkerboard size")
    ax.set_xlabel("$\delta$")
    ax.set_ylabel(crit)
    ax.legend()

    if save_dir is not None:
        plots._savefig(
            fig, save_dir, app_name, val_name, prefix, suffix, overwrite=overwrite
        )

        # save pickle of data
        to_save = {"metrics": metrics_checkered, "deltas": deltas, **best_hps}
        plots._save_fig_data(
            to_save, save_dir, app_name, val_name, prefix, suffix, overwrite=overwrite
        )
