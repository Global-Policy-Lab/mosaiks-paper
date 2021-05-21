import argparse
import json
import pickle
import uuid
from pathlib import Path

import numpy as np
from loguru import logger
from mosaiks import config as c
from mosaiks.solve import cnn
from mosaiks.solve import data_parser as parse
from torch import nn
from torchvision import models

_models = {"resnet18": models.resnet18, "resnet50": models.resnet50}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train deep data satellite prediction for")
    parser.add_argument("out_file", type=str, help="where to save results")
    parser.add_argument("--outcome", default="housing")
    parser.add_argument("--model", type=str, help="model type", default="resnet18")
    parser.add_argument(
        "--test_frac", type=float, help="test frac", default=c.ml_model["test_set_frac"]
    )
    parser.add_argument("--seed", type=int, help="seed", default=0)
    parser.add_argument("--subset", type=int, help="seed", default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--milestones", default="10|20|10,20")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--initial_lrs", type=str, help="inital_lr", default="1e-4, 1e-3, 0.001"
    )
    parser.add_argument("--num_epochs", type=int, help="num_epochs", default=50)
    parser.add_argument("--location", default="CONTUS")
    parser.add_argument("--loss", default="mse")
    parser.add_argument("--num_examples", default=100000, type=int)
    parser.add_argument("--log-loc", type=str, default="./pytorch.logs")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory in which to store checkpoints",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(c.data_dir / "int" / "deep_models"),
        help="Directory in which to store checkpoints",
    )

    args = parser.parse_args()

    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.save_dir = Path(args.save_dir)
    assert args.outcome in c.app_order

    # TODO
    assert args.test_frac == 0.2

    n_samples = c.sampling["n_samples"]
    seed = c.sampling["seed"]
    grid_type = getattr(c, args.outcome)["sampling"]

    data_home = Path(c.data_dir) / "raw" / "imagery" / f"{c.grid['area']}_{grid_type}"

    Y_full, latlons_full, ids_full = cnn.grab_labels(args.outcome)
    Y_full = Y_full[:, np.newaxis]
    (
        ids_train,
        ids_test,
        Y_train,
        Y_test,
        idxs_train,
        idxs_test,
    ) = parse.split_data_train_test(
        ids_full, Y_full, frac_test=args.test_frac, return_idxs=True
    )

    all_val_results = []
    all_milestones = [
        [int(y) for y in x.split(",")] for x in args.milestones.split("|")
    ]
    all_initial_lr = [float(x) for x in args.initial_lrs.split(",")]
    model_ft = _models[args.model](pretrained=args.pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    if args.validate:
        (
            ids_val_train,
            ids_val_test,
            Y_val_train,
            Y_val_test,
            idxs_val_test,
            idxs_val_test,
        ) = parse.split_data_train_test(
            ids_train, Y_train, frac_test=args.test_frac, return_idxs=True
        )
        for lr in all_initial_lr:
            for milestones in all_milestones:
                val_model_uuid = uuid.uuid4()

                logger.debug(f"Val Model UUID: {str(val_model_uuid)}")
                logger.debug(f"trying {lr}, {milestones}")
                val_results_dict = cnn.deep_learning_solve_function(
                    val_model_uuid,
                    ids_val_train,
                    ids_val_test,
                    Y_val_train,
                    Y_val_test,
                    initial_lr=lr,
                    model=model_ft,
                    data_home=data_home,
                    num_epochs=args.num_epochs,
                    loss=args.loss,
                    augment=args.augment,
                    outcome=args.outcome,
                    milestones=milestones,
                    subset=args.subset,
                    log_loc=args.log_loc,
                    save_dir=args.save_dir,
                )
                all_val_results.append((val_results_dict["test_r2"], lr, milestones))
        _, best_initial_lr, best_milestones = max(all_val_results, key=lambda x: x[0])
    else:
        best_milestones = all_milestones[0]
        best_initial_lr = all_initial_lr[0]

    logger.info("================")
    logger.info(f"Best LR: {best_initial_lr}, Best milestones: {best_milestones}")
    assert np.all(np.isfinite(Y_train)), "All train outcomes must be finite"
    assert np.all(np.isfinite(Y_test)), "All test outcomes must be finite"
    train_args = args.__dict__.copy()
    model_uuid = uuid.uuid4()

    logger.info("Model UUID:", str(model_uuid))

    if args.checkpoint_dir is not None:
        with open(args.checkpoint_dir, "wb") as f:
            json.dump(train_args, f)

    results_dict = cnn.deep_learning_solve_function(
        model_uuid,
        ids_val_train,
        ids_val_test,
        Y_train,
        Y_test,
        initial_lr=best_initial_lr,
        model=model_ft,
        data_home=data_home,
        num_epochs=args.num_epochs,
        loss=args.loss,
        augment=args.augment,
        outcome=args.outcome,
        milestones=best_milestones,
        subset=args.subset,
        save_dir=args.save_dir,
    )

    with open(args.out_file.replace(".pickle", "") + ".pickle", "wb+") as f:
        f.write(pickle.dumps(results_dict))
