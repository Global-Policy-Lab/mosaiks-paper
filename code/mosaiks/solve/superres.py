from pathlib import Path

import numpy as np
import torch
from mosaiks.utils.io import GPU
from skimage.transform import downscale_local_mean
from sklearn.metrics import r2_score

from .. import config as c
from ..utils import io, spatial
from . import solve_functions as solve


def heatmap_gen(x, net, use_gpu=GPU):
    """
    x -> is a 3 x M x M image
    net -> is a pytorch neural network
    """

    assert x.shape[0] == 1 or len(x.shape) == 3
    if x.shape[0] == 0:
        if x.shape[1] > 3:
            print("removing the extra channels ({0} channels given)".format(x.shape[1]))
            x = x[:, :3, :, :]
    elif len(x.shape) == 3:
        if x.shape[0] > 3:
            print("removing the extra channels ({0} channels given)".format(x.shape[0]))
            x = x[:3, :, :]
    net.use_gpu = use_gpu
    net.pool_size = 1
    net.pool_stride = 1

    image_torch = torch.from_numpy(x).unsqueeze(0).float()
    if use_gpu:
        image_torch = image_torch.cuda()

    # 256 x 256 x num_filters
    return net(image_torch)


def per_pixel_predictions(image, w_star, net_pred, use_gpu=GPU):
    """convert an image to the per-pixel predictions as dictated by
    the weights in w_star and the patches in net_pred"""

    num_features, num_tasks = w_star.shape
    w_star_torch = torch.from_numpy(w_star).float()
    if use_gpu:
        net_pred = net_pred.cuda()
        w_star_torch = w_star_torch.cuda()

    hmaps_d = heatmap_gen(image, net_pred, use_gpu=use_gpu).squeeze(0).permute(1, 2, 0)
    image_shape = image.shape
    conv_shape = (image_shape[1] - 2, image_shape[2] - 2, num_tasks)
    idxs_all = np.arange(num_features)
    batch_stride = 1024
    start_idx = 0
    per_pixel_preds = np.zeros(conv_shape)
    while start_idx < num_features:
        idxs_batch = idxs_all[start_idx : start_idx + batch_stride]
        per_pixel_preds += (
            (w_star_torch[idxs_batch] * hmaps_d[..., idxs_batch, None])
            .sum(dim=-2)
            .cpu()
            .detach()
            .numpy()
        )
        start_idx += batch_stride
    return per_pixel_preds


def make_superres_predictions(
    latlons,
    w_star,
    net_pred,
    local_dir=Path(c.data_dir) / "raw" / "imagery" / "CONTUS_UAR",
):
    """Make superresolution predictions for multiple images, indexed by lat/lon.

    Parameters
    ----------
    latlons : array-like (N x 2)
        Lat/lon locations of images (lat is first column)
    w_star : array-like (K x L)
        Learned weights to apply to activation maps. K is number of features; L is
        number of labels
    net_pred : :class:`mosaiks.featurization.BasicCoatesNgNet`
        Featurization object used to create activation maps
    local_dir : str or :class:`pathlib.Path`, optional
        Path to local directory containing images.

    Returns
    -------
    pred_maps : :class:`numpy.ndarray` (N x M x M)
        Activation maps (M is width/height of image)
    """
    torch.cuda.empty_cache()
    pred_maps = []

    ids = spatial.ll_to_ij(
        latlons[:, 1],
        latlons[:, 0],
        c.grid_dir,
        c.grid["area"],
        c.images["zoom_level"],
        c.images["n_pixels"],
    ).astype(str)
    ids = [",".join(i) for i in ids]

    print("going through test frames")
    for ix, i in enumerate(latlons):
        if ix % 50 == 0:
            print("completed {0} of {1}".format(ix, len(latlons)))

        # get the image
        image_0 = io.load_img_from_local(i, image_dir=local_dir)

        # transpose
        image_t = np.transpose(image_0, (2, 0, 1))

        # make predictions
        pred_maps.append(per_pixel_predictions(image_t, w_star, net_pred))

    return np.stack(pred_maps)


def scene_regression(
    X, Y, latlons, ids, lam, labels, c, save_dir, num_to_do, allow_logs=False
):
    """Run Ridge regression and save weights.

    Parameters
    ----------
    X,Y,latlons,ids : :class:`numpy.ndarray`
        Features (N, K), labels (N, L), locations (N, 2), image ids (N,)
    lam : float
        Regularization hyperparamter to use in scene-level Ridge Regression
    labels : iterable of str
        Names of tasks associated with L dimension
    c : :module:`mosaiks.config`
        Config object
    save_dir : str
        Path to save the model weights
    num_to_do : int
        Number of images to make predictions for
    allow_logs : bool
        If True, allow logs for each label if specified in config module. Generally not
        useful for super-resolution.

    Returns
    -------
    w_star : :class:`numpy.ndarray`
        (K, L) array of regression weights
    [latlons,ids,Y]_short : :class:`numpy.ndarray`
        shortened versions of the input values, to match the images for which we are
        making predictions
    y_pred : :class:`numpy.ndarray`
        Predictions (same size as ``Y_short``)
    """
    # only use a certain number of images for prediction
    X_short = X[-num_to_do:]
    Y_short = Y[-num_to_do:]
    latlons_short = latlons[-num_to_do:]
    ids_short = ids[-num_to_do:]

    # get appropriate bounds for regression
    bounds = []
    for lab in labels:
        c = io.get_filepaths(c, lab)
        c_app = getattr(c, lab)
        if c_app["logged"] and allow_logs:
            bounds.append(list(c_app["us_bounds_log_pred"]))
        else:
            bounds.append(list(c_app["us_bounds_pred"]))
    bounds = np.asarray(bounds)

    kwargs_rr = {
        "lambdas": [lam],
        "clip_bounds": bounds,
        "return_preds": True,
        "return_model": True,
        "svd_solve": False,
    }

    # get the best w
    results = solve.single_solve(
        X,
        X_short,
        Y,
        Y_short,
        **kwargs_rr,
    )
    models = results["models"][0]
    w_star = np.vstack([np.vstack(models[i]) for i in range(models.shape[0])]).T
    y_pred = np.dot(X_short, w_star)

    # save the best w
    np.save("{0}/w_star_all_data.npy".format(save_dir), w_star)

    return w_star, latlons_short, ids_short, Y_short, y_pred


def across_image_r2s(
    pred_maps,
    hmaps,
    widths,
    demean=True,
    clip=False,
    bounds=[None, None],
    impute_mean_baseline=False,
    rescale_to_match_labels=False,
):
    """ for superres analsyis across resolutions"""
    r2s_per_width = np.ones(len(widths)) * np.nan

    # ensure preds and truth are same shape
    assert pred_maps.shape == hmaps.shape

    for w, width in enumerate(widths):
        print(width)

        out_shape = int(hmaps.shape[2] / width)

        # crop so that there is not padding when downsampling
        this_hmaps = hmaps[:, : out_shape * width, : out_shape * width]
        this_pred_maps = pred_maps[:, : out_shape * width, : out_shape * width]

        # downsample
        hmap_true = downscale_local_mean(this_hmaps, (1, width, width))
        hmap_pred = downscale_local_mean(this_pred_maps, (1, width, width))

        # rescale
        if rescale_to_match_labels:
            hmap_pred = (
                hmap_pred
                * np.mean(hmap_true, axis=(1, 2))
                / np.mean(hmap_pred, axis=(1, 2))
            )

        # just choose mean (used as predictive skill baseline)
        if impute_mean_baseline:
            # use the mean to impute the values as a baseline
            hmap_pred = np.mean(hmap_pred, axis=(1, 2)) * np.ones_like(hmap_pred)

        # clip to upper and lower bounds
        # only apply if both bounds aren't None for this outcome
        if clip and (not (np.asarray(bounds) == None).all()):
            avg_pred = np.mean(hmap_pred, axis=(1, 2))
            lb, ub = bounds

            # if predictions above extremes, predict the extremes
            if ub is None:
                too_high = np.zeros_like(avg_pred, dtype=bool)
            else:
                too_high = avg_pred >= ub
            if lb is None:
                too_low = np.zeros_like(avg_pred, dtype=bool)
            else:
                too_low = avg_pred <= lb
            hmap_pred[too_high] = ub
            hmap_pred[too_low] = lb
            hmap_pred = np.clip(hmap_pred, *bounds)

        if demean:
            hmap_pred -= np.mean(hmap_pred, axis=(1, 2))[:, np.newaxis, np.newaxis]
            hmap_true -= np.mean(hmap_true, axis=(1, 2))[:, np.newaxis, np.newaxis]

        r2s_per_width[w] = r2_score(hmap_true.flatten(), hmap_pred.flatten())
    return r2s_per_width


def crop_rasters_for_sr(max_sr_factor, *hmaps):
    """Crop a list of rasters to a size such that they can be evently divided by
    ``max_sr_factor``. It assumes that each raster is centered identically. I.e. if one
    raster has size 256x256 and another 254x254, it assumes that it is a border of size
    1 that is removed symmetrically from the first raster to get the location of the
    second raster. It will crop off the bottom-right of the image to make it an evenly
    divisible size.

    Parameters
    ----------
    max_sr_factor : int
        The maximum amplicfication factor for super-resolution predictions that will be
        made using these rasters. I.e. 32 means that there will be 32x32 predictions per
        image.
    hmaps : list of :class:`numpy.ndarray`
        The rasters to crop. The final two dimensions must correpsond to i,j of the
        raster.

    Returns
    -------
    list of :class:`numpy.ndarray`
        Cropped versions of ``hmaps``
    """

    min_width = min([i.shape[-1] for i in hmaps])
    reduction = min_width % max_sr_factor
    out = []
    for h in hmaps:
        crop_width = (h.shape[-1] - min_width) / 2
        assert crop_width == int(crop_width)
        crop_width = int(crop_width)
        out.append(
            h[
                ...,
                crop_width : -(reduction + crop_width),
                crop_width : -(reduction + crop_width),
            ]
        )
    return out
