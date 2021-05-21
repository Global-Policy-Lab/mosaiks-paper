"""This module is loaded to access all model settings.
The first group contains global settings, and then each
application area has its own settings. The variable name for
each application must be associated with a dictionary that
at the minimum has the following keys:


Keys:
-----
application : str
    Name of application area.
variable : str or list of str
    The name(s) of the variable(s) being predicted in this
    application area.
sampling : str
    The sampling scheme used for the preferred model for this
    application area (e.g. 'POP')
"""
######################
# PROJECT SETTINGS
######################

import os
from os.path import basename, dirname, join

import mosaiks
import numpy as np
import seaborn as sns

# get home directory
if "MOSAIKS_HOME" in os.environ:
    root_dir = os.environ["MOSAIKS_HOME"]
else:
    c_dir = dirname(dirname(mosaiks.__file__))
    # assume if we don't find site-packages at this layer, then this package was
    # installed in place (i.e. with -e flag in pip install)
    if basename(c_dir) == "site-packages":
        root_dir = "/"
    else:
        root_dir = dirname(c_dir)
    print(
        "env variable MOSAIKS_HOME not defined;"
        ' setting to: "{}"'.format(root_dir)
        + '\nIf not desired, please reset os.environ["MOSAIKS_NAME"]'
    )
    os.environ["MOSAIKS_HOME"] = root_dir

code_dir = os.environ.get("MOSAIKS_CODE", join(root_dir, "code"))
data_dir = os.environ.get("MOSAIKS_DATA", join(root_dir, "data"))
grid_dir = join(data_dir, "int", "grids")
features_dir = join(data_dir, "int", "feature_matrices")
out_dir = join(data_dir, "output")
res_dir = os.environ.get("MOSAIKS_RESULTS", join(root_dir, "results"))
os.makedirs(res_dir, exist_ok=True)


# GRID
grid = {"area": "CONTUS"}
sampling = {"n_samples": 100000, "seed": 0}


# IMAGES
images = {"source": "google", "zoom_level": 16, "n_pixels": 640}


# FEATURIZATION
features = {
    "random": {
        "patch_size": 3,
        "seed": 0,
        "type": "random_features",
        "num_filters": 4096,
        "pool_size": 256,
        "pool_stride": 256,
        "bias": 0.0,
        "filter_scale": 1e-3,
        "patch_distribution": "empirical",
    },
    "pretrained": {"model_type": "resnet152", "batch_size": 128},
}


# ML MODEL
ml_model = {
    "seed": 0,
    "test_set_frac": 0.2,
    "model_default": "ridge",
    "n_folds": 5,
    "global_lambdas": np.logspace(-4, 3, 9),
}


# SUPER-RESOLUTION
superres = {
    "features_fname": "CONTUS_UAR_superres.pkl",
    "pool_stride": 128,
    "tasks_to_predict": ["population", "treecover"],
    # testing performance config vals
    "test_label": "treecover",
    "lambdas_to_test": np.logspace(4, 7, 4),
    "sigmas_to_test": [8, 16, 32],
    "val_set_size": 1000,
    "n_pred_images": 16000,
    "factors_to_test": [2, 4, 8, 16, 32],
}


# PATCH SIZE EXPERIMENT
patch_size_exp = {
    "patch_sizes": [1, 2, 3, 6, 12, 24],
    "num_filters": 512,
}

# HEAD ET AL REPLICATOIN
head_rep = {
    "patch_size": 3,
    "num_filters": 8192,
}

# PLOTTING
plotting = {
    "extent": [25, 48, -126, -65],  # lat_min, lat_max, lon_min, lon_max
    "grid_res": 5,  # degrees
    "bg_color": "#cccccc",  # background color
    "cmap_fxn": lambda x: sns.light_palette(x, as_cmap=True),
    "cmap_bounds": {
        "nightlights": [1, 4],
        "elevation": [None, 3000],
        "income": [40000, 130000],
        "treecover": [0, 100],
        "roads": [0, 12500],
        "population": [0, 5],
        "housing": [3.5, 6],
    },
    "cbar_extend": {
        "nightlights": "both",
        "elevation": "max",
        "income": "both",
        "treecover": "neither",
        "roads": "max",
        "population": "max",
        "housing": "both",
        "B08303": "both",
        "B15003": "both",
        "B19013": "both",
        "B19301": "both",
        "C17002": "both",
        "B22010": "both",
        "B25071": "both",
        "B25001": "both",
        "B25002": "both",
        "B25035": "both",
        "B25017": "both",
        "B25077": "both",
    },
    "scatter_bounds": {
        "nightlights": [0, None],
        "elevation": [None, None],
        "income": [0, None],
        "treecover": [0, 100],
        "roads": [0, None],
        "population": [0, None],
        "housing": [0, None],
    },
}


# CHECKERBOARD
checkerboard = {
    "deltas": [0.25, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "num_jitter_positions_sqrt": 2,
    "sigmas": np.logspace(-3, 2, 6),
}

# PERFORMANCE TESTS
performance = {
    "num_samp_vector": [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000],
    "folds": 5,
    "random": {"num_feat_vector": [100, 200, 500, 1000, 2000, 4096, 8192]},
    "rgb": {"num_feat_vector": [1, 2, 3]},
}

# APPLICATIONS
app_order = [
    "treecover",
    "elevation",
    "population",
    "nightlights",
    "income",
    "roads",
    "housing",
]

world_app_order = [
    "treecover",
    "elevation",
    "population",
    "nightlights",
]


# MISC
ndv = -9999


# colors
colorblind_friendly_teal = "#029e73"


######################
# APPLICATION SETTINGS
######################

housing = {
    "data": {
        "ztrax": {
            "cutoff_yr": 2010,
            "pctile_clip": 0.99,
            "use_codes_include": ["RR", "RI"],
            "use_codes_not_include": [],
        }
    },
    "ml_model": {"model": "ridge"},
    "application": "housing",
    "colname": "price_per_sqft",
    "variable": "log_price_per_sqft",
    "sampling": "POP",
    "source": "ztrax",
    "color": sns.xkcd_rgb["dark hot pink"],
    "logged": True,
    "disp_name": "Housing price",
    "disp_name_short": "Housing",
    "units_disp": "log($/sqft)",
    "us_bounds_pred": [10, 13421],
    "us_bounds_log_pred": [2.4, 9.5],
    "lambdas": np.logspace(-3, 3, 9),
    "lambdas_patchSize": np.logspace(-5, 7, 16),
    "lambdas_checkerboard": np.logspace(0.125, 2.625, 5),
    "lambdas_testout": np.logspace(-2, 2, 4),
    "lambdas_testin": np.logspace(-2, 2, 4),
    "lambdas_sensNumSamp": np.logspace(-3, 3, 9),
    "lambdas_sensNumFeat": np.logspace(-5, 3, 9),
}

nightlights = {
    "application": "nightlights",
    "colname": "y",
    "variable": "log_nightlights",
    "sampling": "POP",
    "source": "viirs",
    "temporal_resolution": "annual",
    "units": "avg_rade9h",
    "logged": True,
    "us_bounds_pred": [0, None],
    "us_bounds_log_pred": [0, None],
    "world_bounds_pred": [0, 9.05],
    "year": 2015,
    "world_bounds_colorbar": [0, 0.5],
    "color": sns.xkcd_rgb["goldenrod"],
    "disp_name": "Nighttime lights",
    "disp_name_short": "Nightlights",
    "units_disp": "log(1 + nanoWatts/cm2/sr)",
    "lambdas": np.logspace(-3, 3, 9),
    "lambdas_patchSize": np.logspace(-5, 5, 16),
    "lambdas_checkerboard": np.logspace(0.125, 2, 4),
    "lambdas_testout": np.logspace(-2, 2, 4),
    "lambdas_testin": np.logspace(-2, 2, 4),
    "lambdas_sensNumSamp": np.logspace(-2, 3, 9),
    "lambdas_sensNumFeat": np.logspace(-6, 3, 9),
}

elevation = {
    "application": "elevation",
    "variable": "meters",
    "colname": "elevation",
    "units": "meters",
    "sampling": "UAR",
    "world_bounds_colorbar": [0, 4000],
    "color": sns.xkcd_rgb["brown"],
    "logged": False,
    "us_bounds_pred": [-86, 4418],
    "us_bounds_log_pred": [0, None],
    "world_bounds_pred": [-413, 8850],
    "disp_name": "Elevation",
    "disp_name_short": "Elevation",
    "units_disp": "meters",
    "lambdas": np.logspace(-3, 3, 9),
    "lambdas_patchSize": np.logspace(-4, 4, 12),
    "lambdas_checkerboard": np.logspace(-0.5, 2, 5),
    "lambdas_testout": np.logspace(-2, 2, 4),
    "lambdas_testin": np.logspace(-2, 2, 4),
    "lambdas_sensNumSamp": np.logspace(-3, 3, 9),
    "lambdas_sensNumFeat": np.logspace(-5, 3, 9),
}

population = {
    "application": "population",
    "variable": "log_population",
    "colname": "population",
    "sampling": "UAR",
    "color": sns.xkcd_rgb["cerulean blue"],
    "disp_name": "Population density",
    "disp_name_short": "Population",
    "units_disp": "log(1 + people per km$^2$)",
    "logged": True,
    "us_bounds_pred": [0, 30000],
    "world_bounds_pred": [0, 11.4],
    "us_bounds_colorbar": [0, 30000],
    "world_bounds_colorbar": [0, 6],
    "us_bounds_log_pred": [0, 10],
    "us_bounds_log_colorbar": [0, 5],
    "lambdas": np.logspace(-3, 3, 7),
    "lambdas_patchSize": np.logspace(-4, 4, 12),
    "lambdas_checkerboard": np.logspace(-0.8, 1.6, 3),
    "lambdas_testout": np.logspace(-2, 2, 4),
    "lambdas_testin": np.logspace(-2, 2, 4),
    "lambdas_sensNumSamp": np.logspace(-2, 4, 9),
    "lambdas_sensNumFeat": np.logspace(-5, 3, 9),
}

treecover = {
    "application": "treecover",
    "variable": "treecover",
    "colname": "treecover",
    "sampling": "UAR",
    "color": sns.xkcd_rgb["emerald green"],
    "disp_name": "Forest cover",
    "disp_name_short": "Forests",
    "units_disp": "% forest",
    "logged": False,
    "us_bounds_pred": [0, 100],
    "world_bounds_pred": [0, 100],
    "us_bounds_colorbar": [0, 100],
    "world_bounds_colorbar": [0, 100],
    "us_bounds_log_pred": [0, 4.6],
    "us_bounds_log_colorbar": [0, 4.6],
    "lambdas": np.logspace(-3, 3, 9),
    "lambdas_patchSize": np.logspace(-3, 3, 9),
    "lambdas_checkerboard": np.logspace(-2, 1.75, 4),
    "lambdas_testout": np.logspace(-2, 2, 4),
    "lambdas_testin": np.logspace(-2, 2, 4),
    "lambdas_sensNumSamp": np.logspace(-3, 3, 9),
    "lambdas_sensNumFeat": np.logspace(-5, 3, 9),
}

income = {
    "application": "income",
    "variable": "income",
    "colname": "income",
    "sampling": "POP",
    "source": "ACS",
    "color": sns.xkcd_rgb["purple"],
    "disp_name": "Income per household",
    "disp_name_short": "Income",
    "units_disp": "$ per household",
    "logged": False,
    "us_bounds_pred": [0, 300000],
    "us_bounds_colorbar": [0, 300000],
    "us_bounds_log_pred": [0, 13],
    "us_bounds_log_colorbar": [0, 13],
    "lambdas": np.logspace(-3, 3, 9),
    "lambdas_patchSize": np.logspace(-5, 5, 16),
    "lambdas_checkerboard": np.logspace(-0.75, 3, 4),
    "lambdas_testout": np.logspace(-2, 2, 4),
    "lambdas_testin": np.logspace(-2, 2, 4),
    "lambdas_sensNumSamp": np.logspace(-1, 5, 9),
    "lambdas_sensNumFeat": np.logspace(-9, 1, 9),
}

roads = {
    "application": "roads",
    "variable": "length",
    "colname": "length",
    "sampling": "POP",
    "source": "USGS",
    "units": "meters",
    "logged": False,
    "year": 2016,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "Road length",
    "disp_name_short": "Roads",
    "units_disp": "meters",
    "us_bounds_pred": [0, 20000],
    "us_bounds_colorbar": [0, 20000],
    "us_bounds_log_pred": [0, 10],
    "lambdas": np.logspace(-3, 3, 9),
    "lambdas_patchSize": np.logspace(-5, 5, 16),
    "lambdas_checkerboard": np.logspace(-0.625, 7.625, 7),
    "lambdas_testout": np.logspace(-2, 2, 4),
    "lambdas_testin": np.logspace(-2, 2, 4),
    "lambdas_sensNumSamp": np.logspace(-1, 6, 9),
    "lambdas_sensNumFeat": np.logspace(-5, 3, 9),
}


####################################################################
# ACS VARIABLES:
####################################################################

B08303 = {
    "application": "B08303",
    "variable": "MinToWork",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "Minutes",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "MinToWork",
    "disp_name_short": "MinToWork",
    "units_disp": "MinToWork",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, None],
}

B15003 = {
    "application": "B15003",
    "variable": "PctBachDeg",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "Percent",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "PctBachDeg",
    "disp_name_short": "PctBachDeg",
    "units_disp": "PctBachDeg",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, 100],
}

B19013 = {
    "application": "B19013",
    "variable": "MedHHIncome",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "USD",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "MedHHIncome",
    "disp_name_short": "MedHHIncome",
    "units_disp": "MedHHIncome",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, None],
}


B19301 = {
    "application": "B19301",
    "variable": "MedPerCapIncome",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "USD",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "MedPerCapIncome",
    "disp_name_short": "MedPerCapIncome",
    "units_disp": "MedPerCapIncome",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, None],
}


C17002 = {
    "application": "C17002",
    "variable": "PctBelowPov",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "Percent",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "PctBelowPov",
    "disp_name_short": "PctBelowPov",
    "units_disp": "PctBelowPov",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, 100],
}

B22010 = {
    "application": "B22010",
    "variable": "PctFoodStamp",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "Percent",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "PctFoodStamp",
    "disp_name_short": "PctFoodStamp",
    "units_disp": "PctFoodStamp",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, 100],
}

B25071 = {
    "application": "B25071",
    "variable": "PctIncomeRent",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "Percent",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "PctIncomeRent",
    "disp_name_short": "PctIncomeRent",
    "units_disp": "PctIncomeRent",
    "lambdas": np.logspace(2, 5, 4),
    "us_bounds_pred": [0, 100],
}


B25001 = {
    "application": "B25001",
    "variable": "NumHouseUnits",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "Number",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "NumHouseUnits",
    "disp_name_short": "NumHouseUnits",
    "units_disp": "NumHouseUnits",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, 2641],  # max corresponds to 99th percentile of full data
}

B25002 = {
    "application": "B25002",
    "variable": "PctVacant",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "Percent",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "PctVacant",
    "disp_name_short": "PctVacant",
    "units_disp": "PctVacant",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, 100],
}

B25035 = {
    "application": "B25035",
    "variable": "YrBuilt",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "Year",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "BuildingAge",
    "disp_name_short": "BuildingAge",
    "units_disp": "Years",
    "lambdas": np.logspace(-1, 3, 5),
    "us_bounds_pred": [0, 75.3],
}

B25017 = {
    "application": "B25017",
    "variable": "NumHouseRooms",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "Number",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "NumHouseRooms",
    "disp_name_short": "NumHouseRooms",
    "units_disp": "NumHouseRooms",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, None],
}

B25077 = {
    "application": "B25077",
    "variable": "MedHouseValue",
    "colname": "Val",
    "sampling": "POP",
    "source": "ACS",
    "units": "USD",
    "logged": False,
    "year": 2015,
    "color": sns.xkcd_rgb["blood orange"],
    "disp_name": "MedHouseValue",
    "disp_name_short": "MedHouseValue",
    "units_disp": "MedHouseValue",
    "lambdas": np.logspace(-3, 3, 5),
    "us_bounds_pred": [0, None],
}


#################
# DERIVED VALUES
#################
grid_paths = {
    sampling_type: join(
        grid_dir,
        "_".join(
            [
                str(i)
                for i in [
                    grid["area"],
                    images["zoom_level"],
                    images["n_pixels"],
                    sampling_type,
                    sampling["n_samples"],
                    sampling["seed"],
                ]
            ]
        )
        + ".npz",
    )
    for sampling_type in ["UAR", "POP"]
}
