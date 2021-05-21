import argparse
from pathlib import Path

import mosaiks.solve.superres as sr
import numpy as np
import pandas as pd
from mosaiks import config as c
from mosaiks.solve import data_parser as parse
from mosaiks.utils import io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "num_to_do",
        type=int,
        help="How many images to include for training and super-resolution prediction",
        default=1e2,
    )
    parser.add_argument(
        "--LAMBDA",
        type=float,
        help="Regularization hyperparameter to use in Ridge solve",
        default=None,
    )
    parser.add_argument(
        "--TASK_NAMES",
        nargs="*",
        help="Which tasks to make super-resolution predictions for",
        default=c.superres["tasks_to_predict"],
    )
    args = parser.parse_args()

    # default lambda is to use the best one previously identified
    if args.LAMBDA is None:
        lambda_file = (
            Path(c.data_dir)
            / "int"
            / f"superres_hp_eval_n_{c.superres['val_set_size']}"
            / "preds.npz"
        )
        args.LAMBDA = int(np.load(lambda_file)["best_lambda"])

    save_data_dir = (
        Path(c.data_dir) / "output" / "superres" / "sr_n_{0}".format(args.num_to_do)
    )

    # make sure save_data_dir exists; if not, make it
    save_data_dir.mkdir(exist_ok=True, parents=True)

    # load X
    X, latlons, net_pred = io.load_superres_X(c)

    # load Y
    Y = io.get_multiple_Y(c, labels=args.TASK_NAMES, allow_logs=False)

    # shuffle
    Y = Y.sample(frac=1, random_state=0)

    # merge x and y
    Y, X, latlons, ids = parse.merge(Y, X, latlons, pd.Series(Y.index, index=Y.index))

    # run regression
    w_star, latlons_short, ids_short, Y_true, Y_pred = sr.scene_regression(
        X,
        Y,
        latlons,
        ids,
        args.LAMBDA,
        args.TASK_NAMES,
        c,
        save_data_dir,
        args.num_to_do,
        allow_logs=False,
    )

    # save scene-level results
    for tx, t in enumerate(args.TASK_NAMES):
        np.save("{0}/{1}_pred_vals".format(save_data_dir, t), Y_pred[:, tx])
        np.save("{0}/{1}_true_vals".format(save_data_dir, t), Y_true[:, tx])

    # make super-res predictions
    print(
        f"running super-ressolution results for {args.TASK_NAMES} then putting in "
        f"{save_data_dir}"
    )
    pred_maps = sr.make_superres_predictions(latlons_short, w_star, net_pred)

    # save super-res reults
    print("saving in directory {0}".format(save_data_dir))

    # reduce for filesize
    pred_maps = pred_maps.astype(np.float32)
    np.save("{0}/ids".format(save_data_dir), ids_short)
    [
        np.save("{0}/{1}_pred_maps".format(save_data_dir, t), pred_maps[..., tx])
        for tx, t in enumerate(args.TASK_NAMES)
    ]
