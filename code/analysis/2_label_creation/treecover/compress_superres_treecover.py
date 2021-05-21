from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
from mosaiks import config as c

sr_dir = (
    Path(c.data_dir) / "int" / "applications" / "treecover" / "superRes" / "tile256Data"
)
files = list(sr_dir.glob("tileID*.csv"))

# combine individual files
arrs = []
ids = []
for fx, f in enumerate(files):
    if fx % 1000 == 0:
        print(f"Finished {fx}/{len(files)}")
    try:
        in_df = pd.read_csv(f, usecols=[1, 4], dtype={"treecover": np.uint8})
    except ValueError:
        in_df = pd.read_csv(f, usecols=[1, 4])
        in_df["treecover"] = in_df.treecover.where(
            in_df.treecover.notnull(), 255
        ).astype(np.uint8)
    ids.append(in_df.iloc[0, 0])
    arrs.append(np.reshape(in_df.treecover.values, (256, 256)))

arrs = np.stack(arrs)
ids = np.asarray(ids)

# save
superres_data_file = Path(c.data_dir, "int", "applications", "treecover", "sr_true")
superres_data_file.parent.mkdir(exist_ok=True, parents=True)
np.savez_compressed(superres_data_file, ids=ids, frames=arrs)

# delete individual files
rmtree(sr_dir)
