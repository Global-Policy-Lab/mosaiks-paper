############################################################

# This script takes in a set of lat-lons (representing the subgrid we sample)
# and assigns a value for total road length to each grid cell, using
# USGS TIGER shapefiles of road length. 

############################################################

# ----------------- # 
#       set up      # 
# ----------------- #

library(foreach)
library(doParallel)
library(sf)
library(tidyverse)
library(reticulate)
library(raster)
library(here)

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

saveOutput = F

# ----------------- # 
#   load centroids  # 
# ----------------- # 
np <- import("numpy")
npz1 <- np$load(file.path(data_dir, "int/grids", paste0(filename, ".npz")))
npz1$files

# define a subgrid data frame
lon <- npz1$f[["lon"]]
lat <- npz1$f[["lat"]]
ID <- npz1$f[["ID"]]
images_zoom = npz1$f[["zoom"]]
images_pixels = npz1$f[["pixels"]]
subgrid <- data.frame(lon = lon, lat = lat, ID = ID)

# convert subgrid to SpatialPointsDataFrame
centroids <- SpatialPointsDataFrame(coords = subgrid[, c("lon", "lat")], data = subgrid, 
                                    proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))

# --------------------------------------- #
#    extract road length for each state   # 
# --------------------------------------- #
# For each state, identify centroids that fall in that state. 
# Convert these centroids to 1km x 1 km tiles, 
# identify roads that fall in the tiles, then 
# compute total length of roads. 
# Input: list of state names 
# Output: Data with the length of each road type that is observed in the state 


state_list <- c("WY","VT","DC","ND","SD","DE","MT",
                "RI","ME","NH","ID","WV","NE","NM","KS","NV","AR","MS",
                "UT","IA","CT","OK","OR","KY","LA","AL","SC","MN","CO","WI",
                "MD","MO","IN","TN","MA","AZ","WA","VA","NJ","MI","NC","GA",
                "OH","PA","IL","NY","FL","TX","CA")


for (i in 1:length(state_list)){ 

  # read in state road data
  path <- file.path(data_dir,"raw/applications/roads/", state_list[i])
  line <- read_sf(file.path(path, "Trans_RoadSegment.shp"))
  line <- st_transform(line, crs = ("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
  
  # define state polygon by converting line into spatial polgyon object
  state_data <- as(line, 'Spatial')
  
  # convert road data extent to polygons
  xmin <- xmin(state_data) - 0.01
  xmax <- xmax(state_data) + 0.01
  ymin <- ymin(state_data) - 0.01
  ymax <- ymax(state_data) + 0.01
  bigger_extent <- extent(c(xmin, xmax, ymin, ymax))
  state_polygon <- as(bigger_extent, 'SpatialPolygons')
  proj4string(state_polygon) <- "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"
  
  # assign 1 to centroids that fall within a state polygon 
  centroids_in_state_ind <- over(centroids, state_polygon)
  
  # filter centroids that fall within a state polygon 
  centroids_in_state <- cbind(subgrid, centroids_in_state_ind) %>% 
    dplyr::filter(centroids_in_state_ind == 1) %>% 
    dplyr::select(-centroids_in_state_ind)

  ## extract road length 
  if(nrow(centroids_in_state) != 0){
    # convert centroids to tiles
    tiles_in_state <- centroidsToTiles(
      centroids_in_state$lat, centroids_in_state$lon, 
      images_zoom, images_pixels)
    
    # assign CRS to tiles and convert to sf object
    tiles_in_state <- spTransform(tiles_in_state, 
                                  CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
    tiles_in_state <- st_as_sf(tiles_in_state)
    
    
    # set the number of groups to split the tiles into
    nCores <- k <- 10
    registerDoParallel(nCores)
    length_by_state <- data.frame()
    
    
    # in this loop, length of each road segment that fall in a tile is computed
    state_subset <- foreach(j = 1:k) %dopar%{
      tryCatch({
        # set image pixel, number of centroids
        n <- nrow(centroids_in_state)
        m <- round(n / k)
        
        # define subtiles in a state
        if (j %in% seq(1, k - 1, 1)){
          subtiles_in_state <- tiles_in_state[I(m * (j - 1) + 1): I(m * j), ]
        } else{
          subtiles_in_state <- tiles_in_state[I(m * (j - 1) + 1): n, ]
        }
        
        # extract road lines that fall within a tile 
        lines_in_subtiles <- st_intersection(line, subtiles_in_state)
        
        # obtain the length of each line segment
        lines_in_subtiles$len = st_length(lines_in_subtiles)
        
        length_by_state <- plyr::rbind.fill(length_by_state, lines_in_subtiles)
        
      }, error = function(e) NA)
      
      length_by_state
    }
    
    
    # bind road data of k groups of tiles
    length_by_state_temp <- do.call("rbind", state_subset) %>% st_as_sf()
    
    # bind centroids and tiles
    centroids_tiles_in_state <- 
      cbind(as.data.frame(tiles_in_state), centroids_in_state) %>% st_sf()
    
    # compute length by road type 
    length_by_roadtype_by_state <- st_join(
      centroids_tiles_in_state, length_by_state_temp) %>% 
      group_by(ID, lat, lon, MTFCC_CODE) %>% 
      dplyr::summarise(length = sum(len)) %>% 
      spread(MTFCC_CODE, length) 
    
    
    # remove irrelevant columns
    if("<NA>" %in% colnames(length_by_roadtype_by_state)){
      length_by_roadtype_by_state <- length_by_roadtype_by_state %>% 
        dplyr::select(-"<NA>")
    }
    
    q <- ncol(length_by_roadtype_by_state)
    first_road <- colnames(length_by_roadtype_by_state)[4]
    last_road <- colnames(length_by_roadtype_by_state)[q - 1]
    
    # convert NAs to 0 
    length_by_state <- length_by_roadtype_by_state %>% 
      mutate_at(.vars = vars(first_road:last_road), 
                .funs = funs(ifelse(is.na(.), 0, .)))
    
    length_by_state = as.data.frame(length_by_state)
    length_by_state$geometry = NULL
    
    #Overwrite output? 
    if(saveOutput = T) {
      write.csv(length_by_state, file = file.path(data_dir,"int/applications/roads", paste0("length_by_state_", sampling_type), paste0("roadLength_in_", state_list[i], ".csv")), row.names = F)
    }
  } #end if
  
  
} # end for state



# ------------------------------ #
#  consolidate state level data  #
# ------------------------------ #
## Consolidate 49 state-specific csv files into one file. 

# obtain file names of the csv files we want to consolidate
csv_list <- list.files(
  path = file.path(data_dir,"int/applications/roads", paste0("length_by_state_", sampling_type)),
  pattern = "*.csv")

road_types <- c("S1100", "S1200", "S1400", "S1500", "S1630", "S1640", "S1740")
variables <- c("lat", "lon", road_types, "state")
num_variables <- c("lat", "lon", road_types)


# define a function that reads in csv files
readCSV <- function(filename){
  # read csv file
  data <- read.csv(
    file.path(data_dir,"int/applications/roads", paste0("length_by_state_",
           sampling_type), filename), header = F, stringsAsFactors = F)
  
  # define column names
  names(data) <- data[1, ]
  data <- data[-1, ]
  names(data)[names(data) == ""] <- "NA"

  # add state column
  data$state <- str_extract(filename, "[A-Z][A-Z]")
  
  # if the data consists of our variables of interest, then select those variables
  # if not, create a corresponding column with 0's
  if(sum(as.numeric(road_types %in% colnames(data))) == length(road_types)){
    # select variables of interest
    data <- data[, c("ID", variables)]
  }else{
    # identify which road type was not present in the data
    ind <- which(!road_types %in% colnames(data))
    
    # for each of those missing road type, append a 0 vector column
    for(i in 1:length(ind)){
      missing_road_type <- road_types[ind[i]]
      data[[missing_road_type]] <- 0 
    }
  }
  
  # select variables of interest
  data <- data[, c("ID", variables)]
  
  # convert length columns to numeric
  data[, num_variables] <- lapply(
    num_variables, function(x) as.numeric(data[[x]]))
  
  return(data)
}


# read and bind csv files from all states
road_in_USA_temp <- map_df(csv_list, readCSV)
road_length_variables <- paste0("length_", road_types)

# for each tile, obtain the length of each road type and sum them 
road_in_USA <- road_in_USA_temp %>%
  #dplyr::filter(ID %in% subgrid$ID) %>%
  group_by(ID, lon, lat) %>%
  dplyr::summarise(
    length_S1100 = sum(S1100, na.rm = T),
    length_S1200 = sum(S1200, na.rm = T),
    length_S1400 = sum(S1400, na.rm = T),
    length_S1500 = sum(S1500, na.rm = T),
    length_S1630 = sum(S1630, na.rm = T),
    length_S1640 = sum(S1640, na.rm = T),
    length_S1740 = sum(S1740, na.rm = T))

road_in_USA$length <- rowSums(road_in_USA[, road_length_variables])


# save the resulting data frame as a csv file
if(saveOutput == T) {
  write.csv(road_in_USA,
            file = file.path(data_dir,"int/applications/roads", paste0("outcomes_sampled_roads_", filename, ".csv")),
            row.names = F)
}



