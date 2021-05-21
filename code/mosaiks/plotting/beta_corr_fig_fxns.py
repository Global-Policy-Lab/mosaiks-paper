import matplotlib
import matplotlib.pyplot as plt
import mosaiks.config as c
import numpy as np
import scipy
import seaborn as sns
import sklearn

matplotlib.rcParams["pdf.fonttype"] = 42

c_by_app = [getattr(c, i) for i in c.app_order]

applications = [config["application"] for config in c_by_app]
variables = [config["variable"] for config in c_by_app]
sample_types = [config["sampling"] for config in c_by_app]
disp_names = [config["disp_name_short"] for config in c_by_app]
logged = [config["logged"] for config in c_by_app]
units = [config["units_disp"] for config in c_by_app]

c_plotting = getattr(c, "plotting")
colors = [config["color"] for config in c_by_app]

colors_dict = {"treecover": colors[0], "elevation": colors[1], "population": colors[2]}


def plot_beta_correlations(
    weights_this_a,
    weights_this_b,
    task_name_a,
    task_name_b,
    d,
    save_path=None,
    **fig_kwargs,
):

    num_splits = len(weights_this_a)
    context = sns.plotting_context("paper", font_scale=1)
    lines = True
    style = {
        "axes.grid": False,
        "axes.edgecolor": "0.0",
        "axes.labelcolor": "0.0",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.left": lines,
        "axes.spines.bottom": lines,
    }

    sns.set_context(context)
    sns.set_style(style)

    fig, ax = plt.subplots(num_splits, num_splits, figsize=(10, 10), **fig_kwargs)
    r2s = np.zeros((num_splits, num_splits))
    r2s_norm = np.zeros((num_splits, num_splits))

    for i in range(num_splits):
        for j in range(num_splits):
            if i > j:
                ax[i, j].scatter(
                    weights_this_a[i],
                    weights_this_a[j],
                    color=colors_dict[task_name_a],
                    label="treecover weights vs. treecover weights",
                    s=1,
                )
                r2s[i, j] = sklearn.metrics.r2_score(
                    weights_this_a[j], weights_this_a[i]
                )
                r2s_norm[i, j] = (
                    scipy.stats.pearsonr(weights_this_a[j], weights_this_a[i])[0] ** 2
                )

            else:
                ax[i, j].scatter(
                    weights_this_a[i],
                    weights_this_b[j],
                    color="grey",
                    label="treecover weights vs. population weights",
                    s=1,
                )
                r2s[i, j] = sklearn.metrics.r2_score(
                    weights_this_b[j], weights_this_a[i]
                )
                r2s_norm[i, j] = (
                    scipy.stats.pearsonr(weights_this_b[j], weights_this_a[i])[0] ** 2
                )

    for i in range(num_splits):
        ax[0, i].set_title(i)
        ax[i, 0].set_ylabel(i)

    fig.suptitle(
        "Regression weights between/within domains \n"
        + "(colored {0}-{0})\n ".format(task_name_a)
        + "(grey {0}-{1}) \n K={2}".format(task_name_a, task_name_b, d)
    )

    print(f"Cross-fold r2 ({task_name_a}, {task_name_b}): {r2s_norm}")
    if save_path is not None:
        fig.savefig(save_path, dpi="figure")
    return fig, ax
