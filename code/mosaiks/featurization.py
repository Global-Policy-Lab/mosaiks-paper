import concurrent.futures as fs
import pathlib
import time
from pathlib import Path

import dill
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms
from numba import jit
from numba.typed import List
from PIL import Image
from torch import nn
from tqdm import tqdm

from . import config as c
from .utils import io, spatial

MAX_THREADS = 16
TOT_PATCHES = int(1e5)


class OnDiskDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, load_img=True):
        self.root = pathlib.Path(root)
        self.files_lst = [x for x in self.root.glob("**/*") if x.is_file()]
        self.transform = transform
        self.load_img = load_img

    def __getitem__(self, idx):
        rel_path = self.files_lst[idx].relative_to(self.root)
        ret_lst = []
        ret_lst.append(str(rel_path))
        if self.load_img:
            img = Image.open(self.files_lst[idx])
            if self.transform is not None:
                ret_lst.append(self.transform(img))
            else:
                ret_lst.append(img)
        return ret_lst

    def __len__(self):
        return len(self.files_lst)


@jit(nogil=True, cache=True)
def __grab_patches(images, random_idxs, patch_size=6, tot_patches=1e6, seed=0, scale=0):
    patches = np.zeros(
        (len(random_idxs), images.shape[1], patch_size, patch_size), dtype=images.dtype
    )
    for i, (im_idx, idx_x, idx_y) in enumerate(random_idxs):
        out_patch = patches[i, :, :, :]
        im = images[im_idx]
        grab_patch_from_idx(im, idx_x, idx_y, patch_size, out_patch)
    return patches


@jit(nopython=True, nogil=True)
def grab_patch_from_idx(im, idx_x, idx_y, patch_size, outpatch):
    sidx_x = int(idx_x - patch_size / 2)
    eidx_x = int(idx_x + patch_size / 2)
    sidx_y = int(idx_y - patch_size / 2)
    eidx_y = int(idx_y + patch_size / 2)
    outpatch[:, :, :] = im[:, sidx_x:eidx_x, sidx_y:eidx_y]
    return outpatch


def grab_patches(
    images, patch_size=6, tot_patches=5e5, seed=0, max_threads=50, scale=0, rgb=True
):
    if rgb:
        images = images.transpose(0, 3, 1, 2)
    idxs = chunk_idxs(images.shape[0], max_threads)
    tot_patches = int(tot_patches)
    patches_per_thread = int(tot_patches / max_threads)
    np.random.seed(seed)
    seeds = np.random.choice(int(1e5), len(idxs), replace=False)

    tot_patches = int(tot_patches)

    with fs.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i, (sidx, eidx) in enumerate(idxs):
            images.shape[0]
            im_idxs = np.random.choice(
                images[sidx:eidx, :].shape[0], patches_per_thread
            )
            idxs_x = np.random.choice(
                int(images.shape[2]) - patch_size - 1, tot_patches
            )
            idxs_y = np.random.choice(
                int(images.shape[3]) - patch_size - 1, tot_patches
            )
            idxs_x += int(np.ceil(patch_size / 2))
            idxs_y += int(np.ceil(patch_size / 2))
            random_idxs = list(zip(im_idxs, idxs_x, idxs_y))

            # convert random_ixs to typed list for numba
            rix = List()
            [rix.append(i) for i in random_idxs]

            futures.append(
                executor.submit(
                    __grab_patches,
                    images[sidx:eidx, :],
                    patch_size=patch_size,
                    random_idxs=rix,
                    tot_patches=patches_per_thread,
                    seed=seeds[i],
                    scale=scale,
                )
            )
        results = np.vstack(list(map(lambda x: x.result(), futures)))
    idxs = np.random.choice(results.shape[0], results.shape[0], replace=False)
    return results[idxs], idxs


def normalize_patches(
    patches, min_divisor=1e-8, zca_bias=0.001, mean_rgb=np.array([0, 0, 0])
):
    if patches.dtype == "uint8":
        patches = patches.astype("float64")
        patches /= 255.0
    print("zca bias", zca_bias)
    n_patches = patches.shape[0]
    orig_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)
    # Zero mean every feature
    patches = patches - np.mean(patches, axis=1)[:, np.newaxis]

    # Normalize
    patch_norms = np.linalg.norm(patches, axis=1)

    # Get rid of really small norms
    patch_norms[np.where(patch_norms < min_divisor)] = 1

    # Make features unit norm
    patches = patches / patch_norms[:, np.newaxis]

    patchesCovMat = 1.0 / n_patches * patches.T.dot(patches)

    (E, V) = np.linalg.eig(patchesCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    patches_normalized = (patches).dot(global_ZCA).dot(global_ZCA.T)

    return patches_normalized.reshape(orig_shape).astype("float32")


def chunk_idxs(size, chunks):
    chunk_size = int(np.ceil(size / chunks))
    idxs = list(range(0, size + 1, chunk_size))
    if idxs[-1] != size:
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))


def chunk_idxs_by_size(size, chunk_size):
    idxs = list(range(0, size + 1, chunk_size))
    if idxs[-1] != size:
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))


class BasicCoatesNgNet(nn.Module):
    """ All image inputs in torch must be C, H, W """

    def __init__(
        self,
        filters,
        patch_size=6,
        in_channels=3,
        pool_size=2,
        pool_stride=2,
        bias=1.0,
        filter_batch_size=1024,
    ):
        super().__init__()
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.bias = bias
        self.filter_batch_size = filter_batch_size
        self.filters = filters.copy()
        self.active_filter_set = []
        self.start = None
        self.end = None
        self.gpu = False

    def _forward(self, x):
        # Max pooling over a (2, 2) window
        if "conv" not in self._modules:
            raise Exception("No filters active, conv does not exist")
        conv = self.conv(x)
        x_pos = F.avg_pool2d(
            F.relu(conv - self.bias),
            [self.pool_size, self.pool_size],
            stride=[self.pool_stride, self.pool_stride],
            ceil_mode=True,
        )
        x_neg = F.avg_pool2d(
            F.relu((-1 * conv) - self.bias),
            [self.pool_size, self.pool_size],
            stride=[self.pool_stride, self.pool_stride],
            ceil_mode=True,
        )
        return torch.cat((x_pos, x_neg), dim=1)

    def forward(self, x):
        num_filters = self.filters.shape[0]
        activations = []
        for start, end in chunk_idxs_by_size(num_filters, self.filter_batch_size):
            activations.append(self.forward_partial(x, start, end))
        z = torch.cat(activations, dim=1)
        return z

    def forward_partial(self, x, start, end):
        # We do this because gpus are horrible things
        self.activate(start, end)
        return self._forward(x)

    def activate(self, start, end):
        if self.start == start and self.end == end:
            return self
        self.start = start
        self.end = end
        filter_set = torch.from_numpy(self.filters[start:end])
        if self.use_gpu:
            filter_set = filter_set.cuda()
        conv = nn.Conv2d(self.in_channels, end - start, self.patch_size, bias=False)
        # print("rebounding nn.Parameter this shouldn't happen that often")
        conv.weight = nn.Parameter(filter_set)
        self.conv = conv
        self.active_filter_set = filter_set
        return self

    def deactivate(self):
        self.active_filter_set = None


class CoatesNgTrained(nn.Module):
    def __init__(self, feed_forward, weights, whitening_weights=None):
        super().__init__()
        self.feed_forward = feed_forward
        self.weights = weights
        self.classifier = torch.nn.Linear(*weights.shape, bias=False)
        self.classifier.weight = nn.Parameter(weights.t())
        self.whitening_weights = nn.Parameter(whitening_weights)

        if whitening_weights is None:
            self.normalize = False
        else:
            self.normalize = True

    def forward(self, x):
        if self.normalize:
            x = self.whiten(x)
        features = self.feed_forward(x)
        features = features.view(
            features.size(0), features.size(1) * features.size(2) * features.size(3)
        )
        return self.classifier(features)

    def whiten(self, x):
        orig_shape = x.shape
        x = x.view(orig_shape[0], -1)
        row_means = torch.mean(x, dim=1)
        x = x - row_means.unsqueeze(1).expand_as(x)
        row_norms = torch.norm(x, p=2, dim=1)
        x /= row_norms.unsqueeze(1).expand_as(x)
        return torch.mm(x, self.whitening_weights).view(*orig_shape)


def coatesng_featurize(
    net,
    dataset,
    data_batchsize=128,
    num_filters=None,
    filter_batch_size=None,
    gpu=False,
    rgb=True,
):
    net.use_gpu = gpu
    if filter_batch_size is None:
        filter_batch_size = net.filter_batch_size
    if num_filters is None:
        num_filters = len(net.filters)
    X_lift_full = []

    for start, end in chunk_idxs_by_size(num_filters, filter_batch_size):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=data_batchsize)
        X_lift_batch = []
        print(f"generating features {start} to {end}")
        names = []
        with tqdm(total=len(dataset)) as pbar:
            for X_batch_named in data_loader:
                X_batch = X_batch_named[1]
                if gpu:
                    X_batch = X_batch.cuda()
                X_var = X_batch
                names += [x for x in X_batch_named[0]]
                X_lift = net.forward_partial(X_var, start, end).cpu().data.numpy()
                X_lift_batch.append(X_lift)
                pbar.update(X_lift.shape[0])
        X_lift_full.append(np.concatenate(X_lift_batch, axis=0))
    conv_features = np.concatenate(X_lift_full, axis=1)
    net.deactivate()
    return conv_features.reshape(len(dataset), -1), names


def build_featurizer(
    patch_size,
    pool_size,
    pool_stride,
    bias,
    patch_distribution,
    num_filters,
    num_channels,
    seed,
    filter_scale,
    X_train=None,
    filter_batch_size=2048,
):
    dtype = "float32"
    if patch_distribution == "empirical":
        assert (
            X_train is not None
        ), "X_train must be provided when patch distribution == empirical"
        all_patches, idxs = grab_patches(
            X_train,
            patch_size=patch_size,
            max_threads=MAX_THREADS,
            seed=seed,
            tot_patches=TOT_PATCHES,
        )
        all_patches = normalize_patches(all_patches, zca_bias=filter_scale)
        idxs = np.random.choice(all_patches.shape[0], num_filters, replace=False)
        filters = all_patches[idxs].astype(dtype)
        print("filters shape", filters.shape)
    elif patch_distribution == "gaussian":
        filters = (
            np.random.randn(num_filters, num_channels, patch_size, patch_size).astype(
                dtype
            )
            * filter_scale
        )
        print("filters shape", filters.shape)
    elif patch_distribution == "laplace":
        filters = np.random.laplace(
            loc=0.0,
            scale=filter_scale,
            size=(num_filters * num_channels * patch_size * patch_size),
        ).reshape(num_filters, num_channels, patch_size, patch_size)
        filters = filters.astype("float32")
        print("filters shape", filters.shape)
    else:
        raise Exception(f"Unsupported patch distribution : {patch_distribution}")
    net = BasicCoatesNgNet(
        filters,
        pool_size=pool_size,
        pool_stride=pool_stride,
        bias=bias,
        patch_size=patch_size,
        filter_batch_size=filter_batch_size,
    )
    return net


def featurize(image_folder, c):
    fsettings = c.features["random"]
    return __featurize(
        image_folder,
        fsettings["patch_size"],
        fsettings["patch_distribution"],
        fsettings["num_filters"],
        fsettings["pool_size"],
        fsettings["pool_stride"],
        fsettings["bias"],
        fsettings["filter_scale"],
        fsettings["seed"],
    )


def featurize_and_save(image_folder, out_fpath, c):
    # run feature extraction
    X_lift, names, net = featurize(image_folder, c)

    # get lat/lons of images from names
    latlon = np.array([i.split("_")[:2] for i in names], dtype=np.float64)
    lon = latlon[:, 1]
    lat = latlon[:, 0]

    # get zoom level and n-pixels of image from names
    zoom_level, n_pixels = [int(i) for i in names[0].split("_")[2:4]]

    # get i,j IDs for these images
    ij = spatial.ll_to_ij(
        lon,
        lat,
        c.grid_dir,
        c.grid["area"],
        zoom_level,
        n_pixels,
    )
    ij = ij.astype(str)
    ids = np.char.add(np.char.add(ij[:, 0], ","), ij[:, 1])

    # save
    with open(out_fpath, "rb") as f:
        dill.dump(
            {"X": X_lift, "ids_X": ids, "net": net.cpu(), "latlon": latlon},
            f,
            protocol=4,
        )


def __featurize(
    image_folder,
    patch_size,
    patch_distribution,
    num_filters,
    pool_size,
    pool_stride,
    bias,
    filter_scale,
    seed,
    data_batchsize=8,
    filter_batch_size=1024,
    img_size=256,
    patch_dataset_loc=None,
):
    """ Featurize image folder"""

    assert patch_distribution in {"gaussian", "empirical"}
    resize = torchvision.transforms.Resize((256, 256))
    to_tensor = torchvision.transforms.ToTensor()
    transform = torchvision.transforms.Compose([resize, to_tensor])
    dataset = OnDiskDataset(image_folder, transform=transform)
    print("dataset size", len(dataset))
    gpu = torch.cuda.is_available()
    num_channels = 3
    if patch_distribution == "empirical":
        idxs = np.random.choice(len(dataset), 10, replace=False)
        X_train_sample = []
        for i in idxs:
            X_train_sample.append(dataset[i][1].numpy())
        X_train_sample = np.stack(X_train_sample).transpose(0, 2, 3, 1)
    else:
        X_train_sample = None

    featurizer = build_featurizer(
        patch_size,
        pool_size,
        pool_stride,
        bias,
        patch_distribution,
        num_filters,
        num_channels,
        seed,
        filter_scale,
        X_train_sample,
        filter_batch_size,
    )
    start = time.time()
    print(featurizer)
    X_lift, names = coatesng_featurize(
        featurizer, dataset, data_batchsize=data_batchsize, gpu=gpu
    )
    end = time.time()
    featurizer = featurizer.cpu()
    print(X_lift.shape)
    print(
        f"featurization complete, featurized {len(dataset)} training points "
        f"{X_lift.shape[1]} output features, took {end - start} seconds"
    )
    return X_lift, names, featurizer.cpu()


class RemoteSensingSubgridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_home,
        y,
        ids_y,
        transform=None,
    ):
        self.y = y
        self.ids = ids_y
        self.transform = transform
        self.data_home = data_home

    def __len__(self):
        return len(self.ids)

    def __get_image_from_id__(self, img_id):
        im = io.load_img_from_ids_local(img_id, image_dir=self.data_home, c=c)

        if len(im.shape) < 3:
            im = np.stack((im, im, im), axis=2)
        if im.shape[-1] > 3:
            im = im[:, :, :3]
        if im.shape[-1] == 1:
            im = np.concatenate((im, im, im), axis=2)
        return im

    def __getitem__(self, i):
        id_i = self.ids[i]
        y_i = self.y[i]
        x_i = self.__get_image_from_id__(id_i)
        x_i = x_i.transpose(2, 0, 1)
        if self.transform is not None:
            x_i = torch.from_numpy(x_i)
            x_i = self.transform(x_i)
        return id_i, x_i, y_i


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]
