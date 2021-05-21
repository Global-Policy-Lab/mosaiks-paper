import numpy as np
from tqdm import tqdm


def get_features_from_ij(image_i, image_j, names, X):
    # Get the features from the index which matches the name of the image
    goodInd = np.where(
        [
            str(image_i) == line.split("_")[0]
            and str(image_j) == line.split("_")[1].split(".")[0]
            for line in names
        ]
    )

    # If there was no match, mark as missing
    if len(goodInd[0]) == 0:
        feat = "MISSING"
    # If there was a match get those features from X
    elif len(goodInd[0]) == 1:
        feat = X[goodInd][0]
    # Catch errors
    else:
        raise Exception("Should never have gotten here")

    return feat


def get_features_for_clusters(records, map_geometry, X, names):
    """# Returns a numpy array, where each row corresponds to one of the entries
    # in `wealth_records`.  Each row contains the average of the features for
    # all images in that record's cluster.
    # Also returns a list of all clusters for which *no* images were found
    # (may be those right on the border).  The prediction data for these ones
    # should probably be discarded."""

    avg_feature_arrays = tuple()
    missing_records = {}
    records_without_any_images = []
    for record_index, record in tqdm(
        enumerate(records), desc="Loading features for records", total=len(records)
    ):

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
            centroid_longitude, centroid_latitude = map_geometry.get_centroid_long_lat(
                record["i"], record["j"]
            )
            # Save references to tthe approximate latitude and longitude,
            # in case we want to use it for printing out debugging info later.
            record["longitude"] = centroid_longitude
            record["latitude"] = centroid_latitude

        # Collect features for all images in the neighborhood
        feature_arrays = tuple()
        count_missing = 0
        for image_i in range(
            neighborhood["left"], neighborhood["left"] + neighborhood["width"]
        ):
            for image_j in range(
                neighborhood["top"], neighborhood["top"] + neighborhood["height"]
            ):
                example_features = get_features_from_ij(image_i, image_j, names, X)

                if len(example_features) == 7 and example_features == "MISSING":
                    count_missing += 1
                else:
                    feature_arrays += (example_features,)

        # Compute the average of all features over all neighbors
        if len(feature_arrays) > 0:
            cluster_features = np.stack(feature_arrays)
            avg_feature_arrays += (np.average(cluster_features, axis=0),)

        if count_missing > 0:
            missing_records[record_index] = count_missing
            if len(feature_arrays) == 0:
                records_without_any_images.append(record_index)

    if len(missing_records.keys()) > 0:
        print(
            "Missing images for %d clusters. " % (len(missing_records.keys()))
            + ". This might not be a bad thing as some clusters may be near a "
            + "border.  These clusters are:"
        )
        for record_index, missing_count in missing_records.items():
            print(
                "Record %d (%f, %f): %d images"
                % (
                    record_index,
                    records[record_index]["latitude"],
                    records[record_index]["longitude"],
                    missing_count,
                )
            )

    avg_features = np.stack(avg_feature_arrays)
    return avg_features, records_without_any_images
