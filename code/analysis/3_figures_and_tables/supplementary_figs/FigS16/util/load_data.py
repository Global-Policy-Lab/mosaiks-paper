import csv

import numpy as np


# This method assumes that all labels are a string representing an integer
def load_labels(csv_filename):
    labels = []
    with open(csv_filename) as csvfile:
        rows = csv.reader(csvfile)
        first_row = True
        for row in rows:
            if first_row:
                first_row = False
                continue
            labels.append(row[6])
    return np.array(labels)


def load_test_indexes(test_index_filename):
    test_indexes = []
    with open(test_index_filename) as test_index_file:
        for line in test_index_file:
            if len(line.strip()) > 0:
                test_indexes.append(int(line.strip()))
    return test_indexes


def read_records(csv_path, metric_column):
    records = []
    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # We have inconsistent file formats for wealth
            if metric_column == "wealth" and "wealth_index" in row:
                row["wealth"] = float(row["wealth_index"])
                row["i"] = int(row["xcoord"].replace(".0", ""))
                row["j"] = int(row["ycoord"].replace(".0", ""))
            elif metric_column == "wealth":
                row["latitude"] = float(row["LATNUM"])
                row["longitude"] = float(row["LONGNUM"])
            else:
                row["longitude"] = float(row["xcoord"])
                row["latitude"] = float(row["ycoord"])

            records.append(row)
    return records


def read_wealth_records(csv_path):

    records = []

    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Support multiple formats of wealth index file (we have
            # inconsistent internal formats).
            if "wealth_index" in row:
                row["wealth"] = float(row["wealth_index"])
                row["i"] = int(row["xcoord"].replace(".0", ""))
                row["j"] = int(row["ycoord"].replace(".0", ""))
            else:
                row["wealth"] = float(row["wealth"])
                row["latitude"] = float(row["LATNUM"])
                row["longitude"] = float(row["LONGNUM"])
            records.append(row)

    return records


def read_education_records(csv_path):

    records = []

    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Cast cell i, j, and wealth to numbers
            # In the current data, all i's and j's end with an
            # unnecessary .0, so we strip it off
            row["i"] = int(row["xcoord"].replace(".0", ""))
            row["j"] = int(row["ycoord"].replace(".0", ""))
            row["education_index"] = float(row["avg_educ_index"])
            records.append(row)

    return records


def read_water_records(csv_path):

    records = []

    with open(csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Cast cell i, j, and wealth to numbers
            # In the current data, all i's and j's end with an
            # unnecessary .0, so we strip it off
            row["i"] = int(row["xcoord"].replace(".0", ""))
            row["j"] = int(row["ycoord"].replace(".0", ""))
            row["water_index"] = float(row["avg_water_index"])
            records.append(row)

    return records


def get_map_from_i_j_to_example_index(nightlights_csv_path):
    # Later on, we're going to have to go from an `i` and `j` of a cell
    # in the raster map to an example index.  We've luckily already
    # stored the relationship between these in a CSV file.  We just have
    # to hydrate it into a map.
    i_j_to_example_dict = {}
    with open(nightlights_csv_path) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # Cast longitude, latitude, and wealth to numbers
            id_ = int(row["id"])
            i = int(row["full_i"])
            j = int(row["full_j"])
            i_j_to_example_dict[(i, j)] = id_

    return i_j_to_example_dict
