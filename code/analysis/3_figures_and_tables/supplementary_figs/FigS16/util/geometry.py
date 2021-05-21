import argparse

import numpy as np
from osgeo import gdal
from util.load_data import (
    get_map_from_i_j_to_example_index,
    read_education_records,
    read_water_records,
    read_wealth_records,
)


class MapGeometry(object):
    """
    This class is an eyesore, but unfortunately it's necessary for finding
    images within the vicinity of a latitude and longitude, so that we can
    predict a `y` using average features over a larger spatial area.
    """

    def __init__(self, raster_filename):
        self.load_raster(raster_filename)

    def load_raster(self, raster_filename):

        raster_dataset = gdal.Open(raster_filename, gdal.GA_ReadOnly)

        # get project coordination
        bands_data = []

        # Loop through all raster bands
        for b in range(1, raster_dataset.RasterCount + 1):
            band = raster_dataset.GetRasterBand(b)
            bands_data.append(band.ReadAsArray())
        bands_data = np.dstack(bands_data)
        rows, cols, n_bands = bands_data.shape

        # Get the metadata of the raster
        geo_transform = raster_dataset.GetGeoTransform()
        (
            upper_left_x,
            x_size,
            x_rotation,
            upper_left_y,
            y_rotation,
            y_size,
        ) = geo_transform

        # Get location of each pixel
        x_size = 1.0 / int(round(1 / float(x_size)))
        y_size = -x_size
        y_index = np.arange(bands_data.shape[0])
        x_index = np.arange(bands_data.shape[1])
        top_left_x_coords = upper_left_x + x_index * x_size
        top_left_y_coords = upper_left_y + y_index * y_size

        # Add half of the cell size to get the centroid of the cell
        centroid_x_coords = top_left_x_coords + (x_size / 2)
        centroid_y_coords = top_left_y_coords + (y_size / 2)

        self.x_size = x_size
        self.top_left_x_coords = top_left_x_coords
        self.top_left_y_coords = top_left_y_coords
        self.centroid_x_coords = centroid_x_coords
        self.centroid_y_coords = centroid_y_coords

    def get_cell_idx(self, lon, lat):
        lon_idx = np.where(self.top_left_x_coords < lon)[0][-1]
        lat_idx = np.where(self.top_left_y_coords > lat)[0][-1]
        return lon_idx, lat_idx

    def get_image_rect_from_cell_indexes(self, image_i, image_j):
        return self.get_image_rect_from_long_lat(
            self.centroid_x_coords[image_i], self.centroid_y_coords[image_j]
        )

    def get_image_rect_from_long_lat(self, longitude, latitude):
        """
        We want to get a 10x10 matrix of images around this image. All we have is the
        center cell indexes and latitude and longitude of the center. We can't just
        expand to 5 on either side, as this will give us an 11x11 matrix of images.
        So we can create this 11x11 matrix, and truncate whichever sides are
        farthest away from the center latitude and longitude.

        In practice, I compute this as follows. I pick the image 5 to the left of
        the center image (the left image). Then I compute the longitude of the
        ideal left boundary of the matrix, if the center coordinates were right in
        the middle of the 10x10 cell. If the ideal left boundary is closer to right
        side of the left image than the left side, then less than half of this
        column of images would fit within the ideal image matrix: we truncate the
        left side. Otherwise, less than half of the right column would fit in the
        ideal image matrix; we truncate the right side. We use the same logic to
        decide whether to truncate the top or bottom of the 11x11 matrix.
        """
        (image_i, image_j) = self.get_cell_idx(longitude, latitude)
        left_image_i = image_i - 5
        left_image_center_longitude = self.centroid_x_coords[left_image_i]
        ideal_left_longitude = longitude - self.x_size * 5
        truncate_left = ideal_left_longitude > left_image_center_longitude

        top_image_j = image_j - 5
        top_image_center_latitude = self.centroid_y_coords[top_image_j]
        ideal_top_latitude = (
            latitude + self.x_size * 5
        )  # (latitude gets more positive as we go up)
        truncate_top = ideal_top_latitude < top_image_center_latitude

        rect = {"width": 10, "height": 10}
        rect["left"] = image_i - 4 if truncate_left else image_i - 5
        rect["top"] = image_j - 4 if truncate_top else image_j - 5
        return rect

    def get_centroid_long_lat(self, image_i, image_j):
        return self.centroid_x_coords[image_i], self.centroid_y_coords[image_j]


def get_indexes_for_clusters(records, i_j_to_example_index_map, map_geometry):

    example_indexes = set()

    for record_index, record in enumerate(records):

        # Find the neighborhood of images for this record's location
        # Latitude and longitude are more precise, so if they're available, use
        # them for finding the closest set of images in the neighborhood
        if "longitude" in record and "latitude" in record:
            neighborhood = map_geometry.get_image_rect_from_long_lat(
                record["longitude"], record["latitude"]
            )
        else:
            neighborhood = map_geometry.get_image_rect_from_cell_indexes(
                record["i"], record["j"]
            )

        # Collect features for all images in the neighborhood
        for image_i in range(
            neighborhood["left"], neighborhood["left"] + neighborhood["width"]
        ):
            for image_j in range(
                neighborhood["top"], neighborhood["top"] + neighborhood["height"]
            ):
                if (image_i, image_j) not in i_j_to_example_index_map:
                    continue
                example_index = i_j_to_example_index_map[(image_i, image_j)]
                example_indexes.add(example_index)

    return example_indexes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute indexes of images"
        + "that will be used for computing indexes in clusters"
    )
    parser.add_argument(
        "wealth_csv",
        help="CSV file where "
        + "the top row is a header, col 1 (zero-indexed) is the wealth index, "
        + "col 7 is the latitude, and col 8 is the longitude.",
    )
    parser.add_argument(
        "education_csv",
        help="CSV file where "
        + "the top row is a header, col 3 (zero-indexed) is the education index, "
        + "col 1 is the cell's 'i' coordinate, and col 2 is the 'j' coordinate.",
    )
    parser.add_argument(
        "water_csv",
        help="CSV file where "
        + "the top row is a header, col 4 (zero-indexed) is the water index, "
        + "col 2 is the cell's 'i' coordinate, and col 3 is the 'j' coordinate.",
    )
    parser.add_argument(
        "nightlights_csv",
        help="CSV file where "
        + "the top row is a header, col 0 (zero-indexed) is the index of the "
        + "example (basename of eature file), and cols 2 and 3 are the "
        + "i and j of the cell in the nightlights data",
    )
    parser.add_argument(
        "nightlights_raster",
        help="Raster file of "
        + "nightlights, used for making a map from latitude and longitude "
        + "to cell indexes on the map.",
    )
    args = parser.parse_args()

    map_geometry = MapGeometry(args.nightlights_raster)
    i_j_to_example_index_map = get_map_from_i_j_to_example_index(args.nightlights_csv)

    # Read in records for wealth, education, and water
    wealth_records = read_wealth_records(args.wealth_csv)
    education_records = read_education_records(args.education_csv)
    water_records = read_water_records(args.water_csv)

    # Get indexes of images that we need to load for all three data types
    example_indexes = set()
    wealth_example_indexes = get_indexes_for_clusters(
        wealth_records, i_j_to_example_index_map, map_geometry
    )
    education_example_indexes = get_indexes_for_clusters(
        education_records, i_j_to_example_index_map, map_geometry
    )
    water_example_indexes = get_indexes_for_clusters(
        water_records, i_j_to_example_index_map, map_geometry
    )

    # Print example indexes to STDOUT
    example_indexes = example_indexes.union(
        wealth_example_indexes, education_example_indexes, water_example_indexes
    )
    for index in example_indexes:
        print(index)
