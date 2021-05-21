############################################################

# This script saves the grid lat and lon files and generates a subsample of grid cells (i.e. "subgrid") from within the U.S. or from across the world
# It does this for both UAR and POP sampling in the US. In the global sample we only do UAR sampling. 

############################################################

rm(list=ls())

## Packages
library(RcppCNPy)
library(reticulate)
library(raster)
library(rgdal)
library(maps)
library(mapdata)
library(rgeos)
library(doParallel)
library(here)

## Import config.R to set filepaths
mosaiks_code <- Sys.getenv("MOSAIKS_CODE")
if (mosaiks_code=="") {
    mosaiks_code = here("code")
}
source(file.path(mosaiks_code,"mosaiks","config.R"))

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

########### INPUTS REQUIRED ############
zoom <- 16 # Choose your zoom level
pixels <- 640 # Pixels per grid cell
multiplier <- 3.25 # This is how many extra draws you want to take from the grid to make sure in the end you get N
seed <- 0

# Number of cores when parallelizing:
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
    no_cores = 4
} else {
    no_cores = args[2]
}

########################################

########### Make the full grid ############
for (extent in c("global", "USA"))
{
    # Set bounding box to global or USA
    if (extent=="USA") {
      latmin <- 25
      latmax <- 50
      lonmin <- -125
      lonmax <- -66
      N <- 100000
    } else if (extent == "global") {
      latmin <- -63
      latmax <- 80
      lonmin <- -180
      lonmax <- 180
      N <- 1000000
    }

    # Get vectors of lat and lon grid cell center values, which when crossed together, define the entire grid. 
    gridvals <- makegrid(zoom, pixels, lonmin, lonmax, latmin, latmax)
    latVals <- gridvals[[2]] 
    lonVals <- gridvals[[1]]

    ###### Save just the grid values for future use. 
    np <- import("numpy")
    if (extent=="USA") {
      filename = "grid_CONTUS_16_640"
    } else if (extent=="global") {
      filename = "grid_WORLD_16_640"
    }

    np$savez(file.path(data_dir, "int/grids", paste0(filename, ".npz")), lon = lonVals, lat = latVals, zoom = zoom, pixels = pixels)
    for (sampling in c("UAR", "POP"))
    {
        
        # don't need pop sample globally
        if (extent=="global" && sampling=="POP") {
            next
        }
        # Note that the original random samples were created prior to setting a RNG 
        # seed, so the random sample that is generated below will not contain the same 
        # grid points as those used in our published analysis.
        set.seed(seed)

        ########### Sample obs ############

        if (sampling=="UAR") {

          subgridN2 <- as.data.frame(matrix(NaN, nrow=N*multiplier, ncol=2 ))
          colnames(subgridN2) <- c("lon", "lat")
          subgridN2$lon <- sample(lonVals, size = N*multiplier, replace = TRUE)
          subgridN2$lat <- sample(latVals, size = N*multiplier, replace = TRUE)

          # drop any duplicate entries
          subgridN2 <- subgridN2[!duplicated(subgridN2), ]

        } else if (sampling=="POP") { # POPULATION WEIGHTED SAMPLING
          weightR = raster(file.path(data_dir, "raw/applications/population/gpw_v4_population_density_rev10_2015_30_sec.tif"))
          weightR = crop(weightR, extent(lonmin, lonmax, latmin, latmax))
          weightRPts = rasterToPoints(weightR)

          vals = weightRPts[,3]
          index = 1:length(vals)

          starttime = Sys.time()
          selectedInd = sample(x = index, size = N*multiplier, prob = vals, replace = T)
          endtime = Sys.time()
          endtime - starttime 

          ## for each of the indicies, find the grid coordinates within the raster cell and randomly choose one of those. 
          selected = weightRPts[selectedInd,]

          resRaster = xres(weightR)

          registerDoParallel(no_cores)

          starttime = Sys.time()
          selectedLatLons = foreach (i = 1:nrow(selected),
                                     .combine = rbind,
                                     .export = c("selected","resRaster","lonVals","latVals"),
                                     .packages = c("raster")) %dopar% 
            {
              lon = selected[i,1]
              lat = selected[i,2]

              goodLons = lonVals[lonVals < (lon + resRaster) & lonVals > (lon - resRaster)]
              goodLats = latVals[latVals < (lat + resRaster) & latVals > (lat - resRaster)]

              if(min(length(goodLons),length(goodLats)) == 0) {
                toReturn = (c(NA,NA))
              } else {

                if(length(goodLons) == 1) {
                  goodLons = c(goodLons,goodLons)
                }
                if(length(goodLats) == 1) {
                  goodLats = c(goodLats,goodLats)
                }

                toReturn = (c(sample(x = goodLons, size = 1), sample(x = goodLats, size = 1) ) ) 
              }
              toReturn

              return(toReturn)
            }
          endtime = Sys.time()
          endtime - starttime 

          stopImplicitCluster()

          # drop any duplicate entries
          selectedLatLons <- selectedLatLons[!is.na(selectedLatLons[,1]), ]
          selectedLatLons <- selectedLatLons[!is.na(selectedLatLons[,2]), ]
          selectedLatLons <- selectedLatLons[!duplicated(selectedLatLons), ]

          subgridN2 = as.data.frame(selectedLatLons)
          colnames(subgridN2) <- c("lon", "lat")

        }

        ########### Crop observations to appropriate land areas (USA or global land area)  ############
        if (extent=="USA") {
          subgrid <- subsetToLand(subgridN2, lonmin, lonmax, latmin, latmax, file.path(data_dir, "raw/shapefiles/gadm36_USA_shp/gadm36_USA_0.shp"))
        } else if (extent=="global") {
          subgrid <- subsetToLand(subgridN2, lonmin, lonmax, latmin, latmax, file.path(data_dir, "raw/shapefiles/world/land_polygons.shp"))
        }

        # Only keep the first N instances
        if (dim(subgrid)[1] > N) {
          subgrid <- subgrid[1:N,]
          print("Excess draws dropped -- N observsations now saving...")
        } else if (dim(subgrid)[1] < N) {
          print("STOP: Too few observations -- need to increase the multiplier!")
        } else {
          print("Exactly N observations drawn and matched!")
        }

        ############ Identify each obs as i,j ############

        # find the i(lat), j(lon) location of each subgrid
        S <- dim(subgrid)[1]

        starttime <- Sys.time()
        for (n in 1:S) { # This should take about 2 minutes
          subgrid$i[n] <- which(latVals==subgrid$lat[n])
          subgrid$j[n] <- which(lonVals==subgrid$lon[n])
        }
        endtime <- Sys.time()
        endtime - starttime

        subgrid$ID <- paste(subgrid$i, subgrid$j, sep=",")

        ########### Save ############
        if (extent=="USA") {
          filename <- paste0("CONTUS_", as.character(zoom), "_", as.character(pixels), "_", 
                             sampling, "_", format(N, scientific=FALSE), "_", as.character(seed)) 
        } else if (extent=="global") {
          filename <- paste0("WORLD_", as.character(zoom), "_", as.character(pixels), "_", 
                             sampling, "_", format(N, scientific=FALSE), "_", as.character(seed))
        }

        # Export to npz
        np <- import("numpy")
        np$savez(file.path(data_dir, "int/grids", paste0(filename, ".npz")), lon = subgrid$lon, lat = subgrid$lat, 
                 ID = subgrid$ID, zoom = zoom, pixels = pixels)

        print("-------- Saved! -----------")
    }
}
