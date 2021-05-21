############################################################

# This script makes Figure S1-B, an overlay of private road 
# segments on top of a satellite image in Indiana. 

############################################################

# ----------------- # 
#       set up      # 
# ----------------- #

## Package
library(png)
library(reticulate)
library(tidyverse)
library(here)
library(sf)

rm(list=ls())

## Import config.R to set filepaths
mosaiks_code <- Sys.getenv("MOSAIKS_CODE")
if (mosaiks_code=="") {
    mosaiks_code = here("code")
}
source(file.path(mosaiks_code,"mosaiks", "config.R"))

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

###--- Set any constants ---###
sampling_type <- "POP"
filename <- paste0("CONTUS_16_640_",sampling_type, "_100000_0") #Or change POP to UAR

# ----------------- # 
#   load centroids  # 
# ----------------- # 
np <- import("numpy")
npz1 <- np$load(file.path(data_dir, "int/grids", paste0(filename, ".npz")))
npz1$files
npz1$f[["pixels"]]
head(npz1$f[["lat"]])

# define a subgrid data frame
lon <- npz1$f[["lon"]]
lat <- npz1$f[["lat"]]
ID <- npz1$f[["ID"]]
images_zoom = npz1$f[["zoom"]]
images_pixels = npz1$f[["pixels"]]
subgrid <- data.frame(lon = lon, lat = lat, ID = ID)

# convert subgrid to SpatialPointsDataFrame and define CRS
centroids <- SpatialPointsDataFrame(coords = subgrid[, c("lon", "lat")], 
                                    data = subgrid, 
                                    proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))


# --------------------------------------------- #
#    Extract Road Segment in a Selected Tile    # 
# --------------------------------------------- #
state_list <- c("IN")

# read in road data for Indiana
path <- file.path(data_dir,"raw/applications/roads", state_list)
line <- read_sf(file.path(path, "Trans_RoadSegment.shp"))
line <- st_transform(line, crs = ("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))


# define centroid of interest: 40.314, -86.754
centroids_interest <- subgrid %>% 
  dplyr::filter(lat < 40.315 & lat > 40.310 & 
                lon < -86.753 & lon > -86.755)

# convert centroid to tile 
#LATMAX = 500
tiles_interest <- centroidsToTiles(
  centroids_interest$lat, centroids_interest$lon, 
  images_zoom, images_pixels)


# assign CRS to tiles and convert to sf object
tiles_interest <- spTransform(tiles_interest, 
                  CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
tiles_interest <- st_as_sf(tiles_interest)


# extract road lines that fall in a tile 
lines_interest <- st_intersection(line, tiles_interest)


# -------------------------------------- #
#    Overlay Road Segment on an Image    # 
# -------------------------------------- #

# Obtain the plotting region based on road segments and adjust 
png(file.path(res_dir, "figures/FigS1/FigS1B.png"))
plot(lines_interest$geometry)
b = st_bbox(tiles_interest)

# read in satellite image of a selected tile 
tryCatch({
  image = readPNG(file.path(data_dir, "int/imagery/example_images/roads_data_coverage.png"))
  rasterImage(image,
              xleft = b$xmin, ybottom = b$ymin,
              xright = b$xmax, ytop = b$ymax)
  plot(lines_interest$geometry, add = T, col = "white", lwd = 3.5)
}, error=function(x){
  message(paste0("----- Imagery file not present, cannot plot panel B --------"))
  message("Here's the original error message:")
  message(x)
}
)

dev.off()