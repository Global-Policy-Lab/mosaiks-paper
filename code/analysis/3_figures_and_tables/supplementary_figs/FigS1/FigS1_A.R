############################################################

# This script makes Figure S1-A, plot of private road segments
# in northern Midwest 

############################################################

# ----------------- # 
#       set up      # 
# ----------------- #

## Package
library(tidyverse)
library(sf)
library(USAboundaries) 
library(sp)
library(here)

rm(list=ls())

## Import config.R to set filepaths
mosaiks_code <- Sys.getenv("MOSAIKS_CODE")
if (mosaiks_code=="") {
    mosaiks_code = here("code")
}
source(file.path(mosaiks_code,"mosaiks", "config.R"))

###--- Set any constants ---###
sampling_type <- "POP"
filename <- paste0("CONTUS_16_640_",sampling_type, "_100000_0") #Or change POP to UAR
out_dir <- file.path(res_dir, "figures", "FigS1")

dir.create(out_dir, showWarnings=FALSE, recursive=TRUE)

# ----------------------- # 
#   Extract Road Segment  # 
# ----------------------- #
state_list <- c("IN", "OH", "MI", "KY", "IL") # northern Midwest states 
road_of_interest <- c("S1740") # private roads 

# read in road data for a selected road type and state 
road_segment <- list()
for (i in 1:length(state_list)){
  path <- file.path(data_dir,"raw/applications/roads", state_list[i])
  lines <- read_sf(file.path(path, "Trans_RoadSegment.shp"))
  lines <- st_transform(lines, crs = ("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
  
  # extract road segment  
  road_segment[[i]] <- lines %>%
    dplyr::select(MTFCC_CODE, geometry) %>%
    dplyr::filter(MTFCC_CODE %in% road_of_interest)
}
road_segments_combined <- do.call("rbind", road_segment)


# ---------------------------- # 
#  Obtain State Boundary Data  # 
# ---------------------------- #
# filter state boundary data by northern Midwest
state_names <- c("indiana", "ohio", "kentucky", "michigan", "illinois")
states_of_interest <- us_states(resolution = "high", states = state_names) %>% 
  st_transform(crs = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")) %>% 
  mutate(lon = map_dbl(geometry, ~st_centroid(.x)[[1]]), 
         lat = map_dbl(geometry, ~st_centroid(.x)[[2]])) 


# ---------------------------------------- # 
#   Plot State Boundary and Road Segments  # 
# ---------------------------------------- #
# define plot range 
par(mgp = c(3, 0.5, 0), mar = c(5, 4, 4, 2) + 0.1)
# plot state boundaries, a tile, and road segments 
png(file.path(out_dir, "FigS1A.png"))
plot(st_geometry(states_of_interest))
points(x = -86.754, y = 40.314, pch = 0, add = T, col = "black", cex = 2)
plot(road_segments_combined$geometry, lwd = 0.1, add = T, col = "black")
dev.off()
