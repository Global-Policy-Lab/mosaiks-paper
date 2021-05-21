import pickle
from os.path import dirname, isfile, join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
from mosaiks import config
from mosaiks.solve import solve_functions as solve
from mosaiks.solve.interpret_results import interpret_kfold_results

from ..utils import OVERWRITE_EXCEPTION


def _adjust_val_names_str(val_names):
    if isinstance(val_names, str):
        val_names = [val_names]
    return val_names


def _savefig(
    fig, save_dir, app_name, val, prefix, suffix, tight_layout=False, overwrite=True
):
    if tight_layout:
        fig.tight_layout()
    save_str = join(save_dir, "{}_{}_{}_{}.png".format(prefix, app_name, val, suffix))
    if isfile(save_str) and (not overwrite):
        raise OVERWRITE_EXCEPTION
    fig.savefig(save_str)


def _save_fig_data(data, save_dir, app_name, val, prefix, suffix, overwrite=True):
    data_str = join(save_dir, "{}_{}_{}_{}.data".format(prefix, app_name, val, suffix))
    if isfile(data_str) and (not overwrite):
        raise OVERWRITE_EXCEPTION
    with open(data_str, "wb") as f:
        pickle.dump(data, f)


def _save_hyperparams_csv(data, save_dir, app_name, val, prefix, suffix, colnames):
    data_str = join(save_dir, "{}_{}_{}_{}".format(prefix, app_name, val, suffix))
    np.savetxt(data_str + ".csv", data, delimiter=",", fmt="%i", header=colnames)


def _get_bounds(bounds, data):
    """Helper func to return data bounds if
    no bounds specified; otherwise return
    specified bounds."""
    bounds_out = []
    if bounds[0] is None:
        bounds_out.append(data.min())
    else:
        bounds_out.append(bounds[0])
    if bounds[1] is None:
        bounds_out.append(data.max())
    else:
        bounds_out.append(bounds[1])

    return bounds_out


def scatter_preds(
    y_preds,
    y_true,
    appname=None,
    title=None,
    ax=None,
    c=None,
    s=0.08,
    alpha=0.4,
    edgecolors="none",
    bounds=None,
    linewidth=0.75,
    axis_visible=False,
    fontsize=6.3,
    despine=True,
    rasterize=False,
    is_ACS=False,
):
    """give a scatter plot of predicted vs. actual values, and set the title as
    specified in the arguments + add some info on the metrics in the title.
    y_true is a vector of true values, y_preds the corresponding predictions."""
    if ax == None:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))

    # first pull defaults from app
    if appname is not None:
        pa = config.plotting
        if not is_ACS:
            this_bounds = pa["scatter_bounds"][appname]

    # now override if you specified
    if bounds is not None:
        this_bounds = bounds
    if alpha is not None:
        this_alpha = alpha

    this_bounds = _get_bounds(this_bounds, np.hstack((y_true, y_preds)))
    # scatter and 1:1 line
    ax.scatter(
        y_preds,
        y_true,
        alpha=this_alpha,
        c=c,
        s=s,
        edgecolors=edgecolors,
        rasterized=rasterize,
    )
    ax.plot(this_bounds, this_bounds, color="k", linewidth=linewidth)

    # fix up axes shape
    ax.set_ylim(*this_bounds)
    ax.set_xlim(*this_bounds)
    ax.set_aspect("equal")
    ax.set_title(title)
    if not axis_visible:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    if despine:
        sns.despine(ax=ax, left=True, bottom=True)

    # add r2
    ax.text(
        0.05,
        1,
        "$R^2 = {:.2f}$".format(metrics.r2_score(y_true, y_preds)),
        va="top",
        ha="left",
        transform=ax.transAxes,
        fontsize=fontsize,
    )

    return ax


def metrics_vs_size(
    results,
    best_lambdas,
    num_folds,
    val_names,
    num_vector,
    xtitle,
    crits="r2_score",
    app_name=None,
    save_dir=None,
    prefix=None,
    suffix=None,
    figsize=(10, 5),
    overwrite=False,
):

    """Plot metrics (e.g. r2) vs number of training observations used to train model.

    Args:
     results (list of dictionaries) : e.g. [{'mse': 117.05561471285215, 'r2_score': 0.875037330527241},
         {'mse': 119.84752736626068, 'r2_score': 0.8735189806862442}]
     best_lambdas (1darray-like) : chosen hyperparameters
     num_folds (scalar) : number of folds stored in the results dictionary.
     crit (str or list of str) : Names of criteria that you want to plot (e.g. 'r2_score') for
        each outcome.
     val_names (str or list of str) : Names of outcome(s). If multiple outcomes, this must be
         a list
     num_vector (list of scalars) : list of scalars to loop over and re-train. E.g. for plotting performance
        against number of training samples, this is a vector of sample sizes.
     xtitle (str) : Either "train set size" or "number of features", depending on which you're plotting on the x axis
     crits (str or list of str) : Names of criteria that you want to plot (e.g. 'r2_score') for
                  each outcome.
     app_name (str) : The name of the application (e.g. 'housing'). Only needed if saving
     save_dir (str) : Path to directory in which to save output files. If None, no figures will be saved.
     prefix (str) : Filename prefix identifying what is being plotted (e.g. test_outof_cell_r2). Only
         needed if figure is being saved
     suffix (str) : The suffix containing the grid and sample parameters which will be appended to the
         filename when saving, in order to keep track of various sampling and gridding schemes.
     overwrite (bool, optional) : If ``overwrite==False`` and the filename that we will
         save to exists, it will raise an error
    Returns:
    None
    """
    val_names = _adjust_val_names_str(val_names)

    for j, val in enumerate(val_names):

        # initialize stack of outcomes
        yvals_by_fold = []

        # initialize plot
        fig, ax = plt.subplots(figsize=figsize)

        # loop over each fold, store metric and plot
        for i in range(num_folds):
            yvals = [res[i][j][crits[j]] for res in results]
            yvals_by_fold.append(yvals)

            ax.plot(num_vector[:], yvals[:], label="fold {0}".format(i))
        ax.set_xscale("log")
        ax.set_title("Performance vs. " + xtitle + " " + val)
        ax.set_xlabel(xtitle)
        ax.set_ylabel(crits[j])
        ax.legend()

        if save_dir is not None:
            _savefig(fig, save_dir, app_name, val, prefix, suffix, overwrite=overwrite)

            # save pickle of data
            to_save = {
                "y_vals": yvals_by_fold,
                "x_vals": num_vector,
                "best_lambda": best_lambdas,
            }
            _save_fig_data(
                to_save, save_dir, app_name, val, prefix, suffix, overwrite=overwrite
            )

    return None


def performance_density(
    kfold_results,
    model_info,
    val,
    lims={},
    save_dir=None,
    app_name=None,
    suffix=None,
    kind="kde",
    bw="scott",
    cut=3,
    size=10,
    alpha=0.25,
):
    """Plots a KDE plot of OOS preds across all folds vs obs.

    Args:
        kfold_results (dict of ndarray) :
            As returned using kfold_solve()
        model_info (str) :
            To append to title of the scatter plot,
            e.g. could pass in formation about which solve...etc it was.
        val (str or list of str):
            An ordered list of names of the outcomes in this model. If not
            multiple outcomes, this can be string. Otherwise must be a list of strings
            of length n_outcomes
        lims (dict of 2-tuple) : Apply lower and upper bounds to KDE plot for a particular val.
            The format of this dict is val : (lower_bound,upper_bound). If no lim is set
            for a particular val, the default is the lower and upper bound of the observed
            and predicted outcomes combined.
        save_dir (str) : Path to directory in which to save output files. If None, no figures will be saved.
        app_name (str) : The name of the application (e.g. 'housing'). Only needed if saving
        suffix (str) : The suffix containing the grid, sample, and featurization parameters
            which will be appended to the filename when saving, in order to keep track of
            various sampling and gridding schemes. Only needed if saving
        kind (str) : Type of plot to draw. Default is KDE. Options:
            { “scatter” | “reg” | “resid” | “kde” | “hex”
        bw (‘scott’ | ‘silverman’ | scalar | pair of scalars, optional) : Bandwidth to use for kernel in kde
            plots. Default is 'scott'. Only implemented for kind='kde'
        cut (numeric) : Kernel is set to go to 0 at min/max data -/+ cut*bw. Only implemented for kind='kde'
    """

    val = _adjust_val_names_str(val)

    # get metrics and preds for best HP's
    best_lambda_idx, best_metrics, best_preds = interpret_kfold_results(
        kfold_results, crits="r2_score"
    )

    # flatten over fold predictions
    preds = np.vstack([solve.y_to_matrix(i) for i in best_preds.squeeze()])
    truth = np.vstack(
        [solve.y_to_matrix(i) for i in kfold_results["y_true_test"].squeeze()]
    )

    # loop over all outcome dimensions
    n_outcomes = preds.shape[1]
    for i in range(n_outcomes):

        this_truth = truth[:, i]
        this_preds = preds[:, i]
        this_val = val[i]

        # calc r2 before clipping
        r2 = metrics.r2_score(this_truth, this_preds)

        # set axis limits for kde plot
        if this_val in lims.keys():
            this_lims = lims[this_val]
        else:
            # select the min and max of input data, expanded by a tiny bit
            offset = (
                max(
                    [
                        this_truth.max() - this_truth.min(),
                        this_preds.max() - this_preds.min(),
                    ]
                )
                / 1000
            )
            this_min = min([this_preds.min(), this_truth.min()]) - offset
            this_max = max([this_preds.max(), this_truth.max()]) + offset
            this_lims = (this_min, this_max)

        print("Plotting {}...".format(this_val))

        # note that below code clips to axes limits before running kernel
        # so if you clip below a large amount of data, that data will be
        # ignored in the plotting (but not in the r2)
        marginal_kws = {}
        if kind == "kde":
            marginal_kws["bw"] = bw
            marginal_kws["clip"] = this_lims
            marginal_kws["cut"] = cut

        # extend the drawing of the joint distribution to the extremes of the
        # data
        joint_kws = marginal_kws.copy()
        if kind == "kde":
            joint_kws["extend"] = "both"

        with sns.axes_style("white"):
            jg = sns.jointplot(
                this_preds,
                this_truth,
                kind=kind,
                height=10,
                xlim=this_lims,
                ylim=this_lims,
                joint_kws=joint_kws,
                marginal_kws=marginal_kws,
                size=size,
                alpha=alpha,
            )

        ## add 1:1 line
        jg.ax_joint.plot(this_lims, this_lims, "k-", alpha=0.75)
        jg.ax_joint.set_xlabel("Predicted")
        jg.ax_joint.set_ylabel("Observed")
        jg.ax_joint.text(
            0.05, 0.95, "r2_score: {:.2f}".format(r2), transform=jg.ax_joint.transAxes
        )

        ## calc metrics
        plt.suptitle(
            "{} Model OOS Performance w/ k-fold CV ({})".format(
                this_val.title(), model_info.title()
            )
        )
        if save_dir:
            fig = plt.gcf()
            _savefig(
                fig,
                save_dir,
                app_name,
                this_val,
                "predVobs_kde",
                suffix,
                tight_layout=True,
            )

            kde_data = {"truth": this_truth, "preds": this_preds}
            _save_fig_data(
                kde_data, save_dir, app_name, this_val, "predVobs_kde", suffix
            )


def spatial_scatter_obs_v_pred(
    kfold_results,
    latlons,
    model_info,
    val,
    s=4,
    save_dir=None,
    app_name=None,
    suffix=None,
    figsize=(14, 5),
    crit="r2_score",
    **kwargs
):
    """Plots side-by-side spatial scatters of observed and predicted values.

    Args:
        kfold_results (dict of ndarray) :
            As returned using kfold_solve()
        latlons (nx2 2darray) : lats (first col), lons (second col)
        model_info (str) :
            To append to title of the scatter plot,
            e.g. could pass in formation about which solve...etc it was.
        val (str or list of str):
            An ordered list of names of the outcomes in this model. If not
            multiple outcomes, this can be string. Otherwise must be a list of strings
            of length n_outcomes
        lims (dict of 2-tuple) : Apply lower and upper bounds to KDE plot for a particular val.
            The format of this dict is val : (lower_bound,upper_bound). If no lim is set
            for a particular val, the default is the lower and upper bound of the observed
            and predicted outcomes combined.
        save_dir (str) : Path to directory in which to save output files. If None, no figures will be saved.
        app_name (str) : The name of the application (e.g. 'housing'). Only needed if saving
        suffix (str) : The suffix containing the grid, sample, and featurization parameters
            which will be appended to the filename when saving, in order to keep track of
            various sampling and gridding schemes. Only needed if saving
    """

    val = _adjust_val_names_str(val)

    # get metrics and preds for best HP's
    best_lambda_idx, best_metrics, best_preds = interpret_kfold_results(
        kfold_results, crits=crit
    )

    # flatten over fold predictions
    preds = np.vstack([solve.y_to_matrix(i) for i in best_preds.squeeze()])
    truth = np.vstack(
        [solve.y_to_matrix(i) for i in kfold_results["y_true_test"].squeeze()]
    )

    # get latlons in same shuffled, cross-validated order
    ll = latlons[
        np.hstack([test for train, test in kfold_results["cv"].split(latlons)])
    ]

    vmin = kwargs.pop("vmin", np.percentile(truth, 10, axis=0))
    vmax = kwargs.pop("vmin", np.percentile(truth, 90, axis=0))

    # plot obs and preds
    for vx, v in enumerate(val):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        sc0 = ax[0].scatter(
            ll[:, 1],
            ll[:, 0],
            c=truth[:, vx],
            cmap="viridis",
            alpha=1,
            s=s,
            vmin=vmin[vx],
            vmax=vmax[vx],
            **kwargs
        )
        sc1 = ax[1].scatter(
            ll[:, 1],
            ll[:, 0],
            c=preds[:, vx],
            cmap="viridis",
            alpha=1,
            s=s,
            vmin=vmin[vx],
            vmax=vmax[vx],
            **kwargs
        )
        fig.colorbar(sc0, ax=ax[0])
        fig.colorbar(sc1, ax=ax[1])
        fig.suptitle(v.title())
        ax[0].set_title("Observed")
        ax[1].set_title("Predicted")
        if save_dir:
            data = {
                "lon": ll[:, 1],
                "lat": ll[:, 0],
                "truth": truth[:, vx],
                "preds": preds[:, vx],
            }
            _savefig(fig, save_dir, app_name, v, "outcomes_scatter_obsAndPred", suffix)
            _save_fig_data(
                data, save_dir, app_name, v, "outcomes_scatter_obsAndPred", suffix
            )
