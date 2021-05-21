""" 
This script creates the following feature matrices:

1. UAR and POP 100k samples in the Continental US (Used for main analysis - Fig 2 and 3)
2. UAR 1M global sample (Used for a global model - Fig 4)
3. Dense samples for 4 locations around the world (Used for prediction with global 
    model - Fig 4)
4. UAR 100k sample in the Continental US, using different patch sizes and only 1024 
    features (Used for evaluating performance of different patch sizes - Fig S6)
"""

from pathlib import Path

import dill
import numpy as np
from mosaiks import config as c
from mosaiks.featurization import featurize, featurize_and_save
from mosaiks.utils import io, spatial

if __name__ == "__main__":

    # CONTUS, WORLD, and DENSE samples
    subgrid_files = Path(c.grid_dir).glob("[!grid_]*.npz")
    base_image_dir = Path(c.data_dir) / "raw" / "imagery"
    for f in subgrid_files:
        grid_name = f.name
        grid_name_lst = grid_name.split("_")
        if grid_name.startswith("DenseSample"):
            label = grid_name_lst[1]
            image_folder = base_image_dir / "dense_samples" / f"dense_{label}"
        else:
            area = grid_name_lst[0]
            sample = grid_name_lst[3]
            image_folder = base_image_dir / f"{area}_{sample}"
        out_fpath = Path(c.features_dir) / f"{image_folder.name}.pkl"

        assert (
            image_folder.is_dir()
        ), f"You have not downloaded images to {image_folder}"

        featurize_and_save(image_folder, out_fpath, c)

    # PATCH SIZE_EXPERIMENT
    for sample in ["UAR", "POP"]:
        c = io.get_filepaths(c, "treecover")
        if sample == "UAR":
            data_suffix = c.data_suffix.replace("POP", "UAR")
        else:
            data_suffix = c.data_suffix.replace("UAR", "POP")
        subgrid_file = Path(c.grid_dir) / f"{data_suffix}.npz"
        image_folder = base_image_dir / f"{c.grid['area']}_{sample}"

        for patch_size in c.patch_size_exp["patch_sizes"]:
            out_fpath = (
                Path(c.features_dir)
                / "patch_size_experiment"
                / f"patch_size_{patch_size}_{c.patch_size_exp['num_filters']*2}_filters"
                f"_{image_folder.name}.pkl"
            )

            featurize_and_save(image_folder, out_fpath, c)

    # HEAD REP
    out_dir_base = Path(c.features_dir) / "head_rep"
    for ctry in ["Haiti", "Rwanda", "Nepal"]:
        image_folder = base_image_dir / "head_rep" / ctry
        out_fpath = out_dir_base / ctry / f"{ctry}_full.pkl"
        out_fpath.parent.mkdir(exist_ok=True, parents=True)

        # featurize
        X, names, net = featurize(image_folder, c)

        # save
        with open(out_fpath, "rb") as f:
            dill.dump(
                {"X": X, "names": names, "net": net.cpu()},
                f,
                protocol=4,
            )
