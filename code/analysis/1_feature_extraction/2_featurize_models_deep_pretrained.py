"""
This script creates features from a pre-trained ResNet-152 model.
"""

import argparse
import io as python_io
import os
import time
from pathlib import Path

import dill
import numpy as np
import skimage.io
import skimage.transform
import torch
from mosaiks import config as c
from mosaiks.featurization import RemoteSensingSubgridDataset, chunks
from mosaiks.utils import io
from torch.utils.data import Dataset
from torchvision import models


def resize_images(images):
    images_resized = []
    for im in images:
        im = im[:, :, :3]
        images_resized.append(
            skimage.transform.resize(
                im, (224, 224), mode="constant", anti_aliasing=True
            )
        )
    images = np.stack(images_resized, axis=0)
    return images


def resnet_features(images, model, batch_size=60, gpu=True):
    results = []
    if gpu:
        model = model.cuda()
    for images_chunk in chunks(images, batch_size):
        if len(images_chunk.shape) < 4:
            images_chunk = images[np.newaxis, :, :, :]
        images_chunk = images_chunk.astype("float32").transpose(0, 3, 1, 2)
        images_torch = torch.from_numpy(images_chunk)
        if gpu:
            images_torch = images_torch.cuda()
        x = model.conv1(images_torch)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x.cpu().data.numpy()
        results.append(x)
    torch.cuda.empty_cache()
    return np.concatenate(results, axis=0)


def full_featurize(dataset, model_ft, batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    output_features = []
    data = {}
    ids = []
    for j, (i, X_batch) in enumerate(dataloader):
        print("batch:", j)
        ids += i
        t = time.time()
        X_batch = resize_images(X_batch.numpy())
        X_features = resnet_features(X_batch, model_ft)
        e = time.time()
        print(f"batch: {j} took {e - t}")
        output_features.append(X_features)
    bio_features = python_io.BytesIO()
    np.save(bio_features, np.vstack(output_features))
    data["X"] = bio_features.getvalue()
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default=None, type=int)
    args = parser.parse_args()

    # Use treecover to get the "data_suffix" param used in subgrid path
    # (could use any label name)
    c = io.get_filepaths(c, "treecover")

    this_c = c.features["pretrained"]

    for sample in ["UAR", "POP"]:
        subgrid_path = c.grid_paths[sample]

        out_file = (
            Path(c.features_dir)
            / f"{this_c['model_type']}_pretrained_{c.grid_str}_{sample}.pkl"
        )

    model_ft = getattr(models, this_c["model_type"])(pretrained=True)

    subgrid = np.load(subgrid_path)
    ids = subgrid["ID"]
    latlons = np.hstack((subgrid["lat"][:, np.newaxis], subgrid["lon"][:, np.newaxis]))

    if args.subset is not None:
        latlons = latlons[: args.subset]
        ids = ids[: args.subset]

    dataset = RemoteSensingSubgridDataset(c.data_dir, latlons, ids)
    results_dict = full_featurize(dataset, model_ft, this_c["batch_size"])
    results_dict["latlon"] = latlons
    results_dict["ids_X"] = ids
    with open(out_file, "wb") as f:
        dill.dump(results_dict, f, protocol=4)
