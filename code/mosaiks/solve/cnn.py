import io as python_io
import pickle
import time
from pathlib import Path

import numpy as np
import sklearn.metrics
import torch
from loguru import logger
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from .. import config as c
from .. import transforms as m_transforms
from ..featurization import RemoteSensingSubgridDataset
from ..utils import io, spatial


def grab_labels(outcome):
    c_local = io.get_filepaths(c, outcome)
    c_app = getattr(c_local, outcome)
    Y = io.get_Y(c_local, c_app["colname"])
    lons, lats = spatial.ids_to_ll(
        Y.index,
        c.grid_dir,
        c_local.grid["area"],
        c_local.images["zoom_level"],
        c_local.images["n_pixels"],
    )
    latlons = np.vstack((np.array(lats), np.array(lons))).T.astype("float64")
    ids, Y, latlons = m_transforms.dropna_and_transform(
        Y.index.values, Y.values, latlons, c_app
    )
    return Y, latlons, ids


def clip_bounds(y, c_app=None):
    if c_app["logged"]:
        lowb, higb = c_app["us_bounds_log_pred"]
    else:
        lowb, higb = c_app["us_bounds_pred"]
    return np.clip(y, lowb, higb)


def train_model(
    model_uuid,
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    outcome,
    num_epochs=25,
    loss="mse",
    mean=0,
    std=1,
    log_loc="./pytorch.logs",
    save_dir=Path(c.data_dir) / "int" / "deep_models",
):
    since = time.time()
    summary_writer = SummaryWriter(Path(log_loc) / f"{model_uuid}")
    global_step = 0
    preds = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(num_epochs):
        logger.debug("Epoch {}/{}".format(epoch + 1, num_epochs))
        logger.debug("-" * 10)
        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            all_labels = []
            all_predictions = []
            all_ids = []
            counter = 0
            lr = optimizer.param_groups[0]["lr"]
            summary_writer.add_scalar(
                tag="learning_rate", scalar_value=lr, global_step=global_step
            )
            for ids, inputs, labels in dataloaders[phase]:
                counter += 1
                global_step += 1
                all_labels += list(np.vstack(labels.numpy()))
                all_ids += list(ids)
                inputs = inputs.float()
                labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    all_predictions += list(outputs.detach().cpu().numpy())
                    loss = criterion(outputs, labels)
                    if counter % 100 == 0:
                        pass

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        summary_writer.add_scalar(
                            tag="train_loss",
                            scalar_value=loss.item(),
                            global_step=global_step,
                        )
                    else:
                        summary_writer.add_scalar(
                            tag="val_loss",
                            scalar_value=loss.item(),
                            global_step=global_step,
                        )

            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)

            # convert back
            all_labels *= std
            all_labels += mean

            all_predictions *= std
            all_predictions += mean

            r2_score = sklearn.metrics.r2_score(all_labels, all_predictions)
            preds[phase] = (all_labels, all_predictions, all_ids)

            bio = python_io.BytesIO()
            torch.save(model.state_dict(), bio)
            model_checkpoint = {}
            model_checkpoint["model_bytes"] = bio.getvalue()
            model_checkpoint["val_r2"] = r2_score
            model_checkpoint["epoch"] = epoch
            model_checkpoint["preds"] = preds[phase]
            model_checkpoint["domain_name"] = outcome

            if save_dir is not None:
                this_save_path = (
                    save_dir
                    / str(model_uuid)
                    / "checkpoints"
                    / phase
                    / f"epoch_{epoch}_{outcome}.pickle"
                )
                this_save_path.parent.mkdir(exist_ok=True, parents=True)
                with open(this_save_path, "wb") as f:
                    pickle.dump(model_checkpoint, f, protocol=4)
            if phase != "train":
                scheduler.step()
                summary_writer.add_scalar(
                    tag="val_r2", scalar_value=r2_score, global_step=global_step
                )
            else:
                summary_writer.add_scalar(
                    tag="train_r2", scalar_value=r2_score, global_step=global_step
                )

            logger.debug(
                "Epoch {0} Phase {1} complete, Aggregate R2 Score {2}".format(
                    epoch, phase, r2_score
                )
            )

    time_elapsed = time.time() - since
    logger.debug(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return model, preds


def transform_img_inputs(augment):
    out = [transforms.ToPILImage(), transforms.CenterCrop(224)]

    if augment:
        out += [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]

    out += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(out)


def get_dataloader(
    data_home,
    y_test,
    ids_test,
    augment=False,
    subset=None,
    batch_size=16,
    shuffle=True,
    num_workers=0,
):
    transform = transform_img_inputs(augment)
    r_grid = RemoteSensingSubgridDataset(
        data_home,
        y_test,
        ids_test,
        transform=transform,
    )
    if subset is not None:
        r_grid = Subset(r_grid, np.arange(subset))

    return torch.utils.data.DataLoader(
        r_grid, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def deep_learning_solve_function(
    model_uuid,
    ids_train,
    ids_test,
    y_train,
    y_test,
    initial_lr,
    model,
    num_epochs,
    data_home,
    loss,
    augment,
    outcome,
    milestones,
    subset=None,
    log_loc="./pytorch.logs",
    save_dir=Path(c.data_dir) / "int" / "deep_models",
    batch_size=16,
    shuffle=True,
    **kwargs,
):
    sort_dict_test = {x: i for i, x in enumerate(ids_test)}
    sort_dict_train = {x: i for i, x in enumerate(ids_train)}

    mean = y_train.mean()
    std = y_train.std()

    y_train = (y_train - mean) / std
    y_test = (y_test - mean) / std

    dataloaders = {}
    for kind in (
        ("train", y_train, ids_train, augment),
        ("test", y_test, ids_test, False),
    ):
        dataloaders[kind[0]] = get_dataloader(
            data_home,
            kind[1],
            kind[2],
            augment=kind[3],
            subset=subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if loss == "mse":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.L1Loss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)
    # Decay LR by 0.1 every time validation R2 score plataeus
    mile_stones = milestones
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_ft, milestones=mile_stones, gamma=0.5
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'max', factor=0.1)
    model_ft, preds = train_model(
        model_uuid,
        model,
        dataloaders,
        criterion,
        optimizer_ft,
        scheduler,
        outcome,
        num_epochs=num_epochs,
        mean=mean,
        std=std,
        log_loc=log_loc,
        save_dir=save_dir,
    )

    y_train, y_train_pred, ids_train_out = zip(
        *sorted(zip(*preds["train"]), key=lambda x: sort_dict_train[x[2]])
    )
    y_test, y_test_pred, ids_test_out = zip(
        *sorted(zip(*preds["test"]), key=lambda x: sort_dict_test[x[2]])
    )

    y_train_pred = clip_bounds(y_train_pred, c_app=getattr(c, outcome))
    y_test_pred = clip_bounds(y_test_pred, c_app=getattr(c, outcome))
    result_dict = {}
    result_dict["model"] = model_ft
    result_dict["y_train_pred"] = np.array(y_train_pred)
    result_dict["y_test_pred"] = np.array(y_test_pred)
    result_dict["y_train"] = np.array(y_train)
    result_dict["y_test"] = np.array(y_test)
    result_dict["ids_train"] = np.array(ids_train_out)
    result_dict["ids_test"] = np.array(ids_test_out)
    result_dict["initial_lr"] = initial_lr
    result_dict["num_epochs"] = num_epochs
    result_dict["domain_name"] = outcome
    result_dict["test_r2"] = sklearn.metrics.r2_score(y_test, y_test_pred)
    result_dict["train_r2"] = sklearn.metrics.r2_score(y_train, y_train_pred)
    result_dict["model_uuid"] = model_uuid
    return result_dict


def hybrid_adjust_weights_func(X, n_cnn_feat=0, l_rat=0):
    return np.concatenate(
        (X[:, :-n_cnn_feat], X[:, -n_cnn_feat:] * np.sqrt(l_rat)),
        axis=1,
    )


def get_hybrid_adjust_weights_transformer(n_cnn_feat=0, l_rat=1):
    return FunctionTransformer(
        hybrid_adjust_weights_func,
        validate=True,
        kw_args={"l_rat": l_rat, "n_cnn_feat": n_cnn_feat},
    )


def get_clip_transformer(c, app):
    return FunctionTransformer(
        inverse_func=clip_bounds,
        inv_kw_args={"c_app": getattr(c, app)},
        check_inverse=False,
    )


def get_bounded_ridge_regressor(c, app, alpha=1):
    return TransformedTargetRegressor(
        regressor=Ridge(alpha=alpha, fit_intercept=False, random_state=0),
        transformer=get_clip_transformer(c, app),
        check_inverse=False,
    )
