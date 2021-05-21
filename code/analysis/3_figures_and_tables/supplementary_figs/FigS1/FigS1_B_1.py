"""This script downloads a satellite image that is used to demonstrate the road data coverage.
"""

import argparse
import warnings
from pathlib import Path

from imageio import imwrite
from mosaiks import config as c
from mosaiks.utils import io

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_id",
        type=str,
        default="1002,2778",
        help="i,j-style grid cell reference",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=str(
            Path(c.data_dir)
            / "int"
            / "imagery"
            / "example_images"
            / "roads_data_coverage.png"
        ),
        help="location to save image",
    )
    args = parser.parse_args()
    save_dir = Path(c.data_dir, "raw", "imagery", "CONTUS_POP")
    try:
        img = io.load_img_from_ids_local(args.img_id, save_dir)
        imwrite(args.out_path, img)
    except FileNotFoundError:
        warnings.warn("Fig S1 overlay image not found. Will produce blank")
