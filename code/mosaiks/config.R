######################

# This configuration script is loaded to access all settings for all R scripts.
# To adjust filepaths or other settings across all R scripts, users should only
# have to update this configuration script.

######################

library(here)
root_dir <- paste0(Sys.getenv("MOSAIKS_HOME"),"/")
if (root_dir=="/") {
    root_dir = here("")
}

code_dir <- paste0(Sys.getenv("MOSAIKS_CODE"),"/")
if (code_dir=="/") {
    code_dir = file.path(root_dir,"code")
}

data_dir <- paste0(Sys.getenv("MOSAIKS_DATA"),"/")
if (data_dir=="/") {
    data_dir = file.path(root_dir,"data")
}

res_dir <- paste0(Sys.getenv("MOSAIKS_RESULTS"),"/")
if (res_dir=="/") {
    res_dir = file.path(root_dir,"results")
}

# create results directory if it doesn't already exist
dir.create(res_dir, showWarnings=FALSE, recursive=TRUE)

# grid_dir is root/data/grids
grid_dir = file.path(data_dir, "int", "grids")

# features is root/data/features
features_dir = file.path(data_dir, "int", "feature_matrices")

# utils is root/mosaiks/utils
utils_dir = file.path(code_dir, "mosaiks", "utils")

# output is root/data/output
out_dir = file.path(data_dir, "output")
