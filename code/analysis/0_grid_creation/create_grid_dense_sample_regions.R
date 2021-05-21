############################################################

# This script samples the global MOSAIKS grid for the set of zoomed
# plots in Figure 4. It defines a sub-region of each zoom for fully 
# dense sampling (every single grid) and then executes dense (but not fully dense) sampling
# for the remainder of each zoom region.

# Each sub-region ("zoom") is designed to have approximately the same number of samples.

# It is run for four different tasks: forest cover, elevation, nighttime lights, and 
# population density.
############################################################

rm(list=ls())

library(rgdal)
library(ggplot2)
library(sp)
library(raster)
library(maps)
library(mapdata)
library(maptools)
library(cowplot)
library(here)

## Import config.R to set filepaths
mosaiks_code <- Sys.getenv("MOSAIKS_CODE")
if (mosaiks_code=="") {
    mosaiks_code = here("code")
}
source(file.path(mosaiks_code,"mosaiks", "config.R"))

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

# Zoom level and no. of pixels defined our grid (see Supplementary Materials Sec. S.2.1 for details)
zoom <- 16
pixels <- 640
seed <- 0
sampling <- "regular" # This implies that a dense sample for a given bounding box is generated using regular intervals between samples

# These form the bounding box we use for the global grid (here and throughout the project)
# and ensure that each zoom region aligns exactly with the global grid
latmin <- -63
latmax <- 80
lonmin <- -180
lonmax <- 180

############################################################
# Settings specific to each task's zoom region #
############################################################

# initial number of samples to draw
N <- 20000

for (domain in c("elevation", "nightlights", "population", "treecover"))
{
    if (domain == "elevation") {
      # Zoom extent
      ellatmin <- 17; ellatmax <- 37; ellonmin <- 72; ellonmax <- 90
      e = extent(ellonmin, ellonmax, ellatmin, ellatmax)
      ext <- as.vector(e)
      # fully dense sampling subregion extent
      np = as.vector(extent(84,85.5,27,28.5))
      # sampling intensity (varies by domain slightly)
      multiplier <- 1.5 # This is how many extra draws you want to take from the grid to make sure in the end you get N
    } else if (domain == "nightlights") {
      # Zoom extent
      ellatmin <- 28; ellatmax <- 48; ellonmin <- 118; ellonmax <- 135
      e = extent(ellonmin, ellonmax, ellatmin, ellatmax)
      ext <- as.vector(e)
      # fully dense sampling subregion extent
      np = as.vector(extent(126.5,128,37,38.5))
      # sampling intensity (varies by domain slightly)
      multiplier <- 2 # This is how many extra draws you want to take from the grid to make sure in the end you get N
    } else if (domain == "population") {
      # Zoom extent  30,47,-5,15
      ellatmin <- -5; ellatmax <- 15; ellonmin <- 30; ellonmax <- 47
      e = extent(ellonmin, ellonmax, ellatmin, ellatmax)
      ext <- as.vector(e)
      # fully dense sampling subregion extent
      np = as.vector(extent(38.5,40,8,9.5))
      # sampling intensity (varies by domain slightly)
      multiplier <- 1.5 # This is how many extra draws you want to take from the grid to make sure in the end you get N
    } else if (domain == "treecover") {
      # Zoom extent  
      ellatmin <- -20; ellatmax <- 0; ellonmin <- -82; ellonmax <- -65
      e = extent(ellonmin, ellonmax, ellatmin, ellatmax)
      ext <- as.vector(e)
      # fully dense sampling subregion extent
      np = as.vector(extent(-76.25,-74.75,-11.5,-10))
      # sampling intensity (varies by domain slightly)
      multiplier <- 1.8 # This is how many extra draws you want to take from the grid to make sure in the end you get N
    } else {
      print(' ------------- THIS DOMAIN IS NOT AVAILABLE FOR ZOOM SAMPLING -------------')
    }

    ############################################################
    # Country boundaries (to ensure samples are within country borders) #
    ############################################################

    # Call country boundaries
    boundaries <- map('worldHires',
                      xlim=ext[1:2], ylim=ext[3:4],
                      plot=FALSE)
    boundaries <- map2SpatialLines(boundaries, proj4string = CRS("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
    boundariesdf = sp::SpatialLinesDataFrame(boundaries, data.frame(ID = c(1:length(boundaries))), match.ID = F)
    boundariesdf <- crop(boundariesdf, e)

    ############################################################
    # Define the sub-sub-region that will be fully sampled #
    ############################################################

    # Makegrid makes the global grid, subset to the zoom extent
    gridvals <- makegrid(zoom, pixels, lonmin,lonmax,latmin,latmax)
    latVals <- subset(gridvals[[2]], gridvals[[2]] > ext[3] & gridvals[[2]] < ext[4])
    lonVals <- subset(gridvals[[1]],  gridvals[[1]] > ext[1] & gridvals[[1]] < ext[2])

    # Dense sampling for subregion of the zoom (e.g. Kathmandu area for elevation)
    latnp = subset(latVals, latVals>np[3] & latVals<np[4])
    lonnp = subset(lonVals, lonVals>np[1] & lonVals<np[2])

    # initialize a list of spatial polygons for all grids in the extent
    mygrids = list()
    subregdf = as.data.frame(matrix(NA, nrow=length(latnp)*length(lonnp), ncol=2 ))
    colnames(subregdf) = c("lon","lat")
    cc = 0
    for(i in 1:length(latnp)) {
      for(j in 1:length(lonnp)) {
        cc = cc +1
        mygrids[[cc]] = centroidsToTiles(latnp[i],lonnp[j], zoom, pixels)
        subregdf$lat[cc] = latnp[i]
        subregdf$lon[cc] = lonnp[j]
      }
    }

    # merge all the grids into one shapefile
    library(purrr)
    subregion = list(mygrids, makeUniqueIDs = T) %>% 
      flatten() %>% 
      do.call(rbind, .)
    print(paste0("------ No. of images in densely sampled subregion: ", length(subregion), " -----------"))

    ############################################################
    # Regularly sample throughout the rest of the zoom area #
    ############################################################

    # sort lat and lon vecs
    latVals = sort(latVals)
    lonVals = sort(lonVals)

    # store in samples df
    samples = as.data.frame(matrix(NA, nrow = N*multiplier, ncol=2))
    colnames(samples) <- c("lon", "lat")

    # make the full grid, order by lat-lon, only take every Kth obs (K = total N in zoom / desired N in zoom)
    K = round((length(latVals)*length(lonVals))/(N*multiplier))
    counter = 0
    for(ly in 1:length(latVals)){
      for(lx in 1:length(lonVals)) {
        counter = counter +1 
        if(round(counter/K)==counter/K & counter/K <= N*multiplier) {
          samples$lat[counter/K] = latVals[ly]
          samples$lon[counter/K] = lonVals[lx]
        }
      }
    }

    # because of rounding, we may have some extras
    samples = samples[complete.cases(samples),]

    # subet to land -- NOTE: takes a few mins to run
    samples <- subsetToLand(samples, ellonmin, ellonmax, ellatmin, ellatmax, file.path(data_dir, "raw/shapefiles/world/land_polygons.shp"))

    # Get rid of observations also inside the densely sampled box so we don't duplicate effort
    todrop = which(samples$lat %in% latnp & samples$lon %in% lonnp)
    samples = samples[-todrop,]

    # Make polygons of all lat-lon grids in samples
    mysamples = list()
    for(i in 1:dim(samples)[1]) {
      mysamples[[i]] = centroidsToTiles(samples$lat[i],samples$lon[i], zoom, pixels)
    }
    samprecs = list(mysamples, makeUniqueIDs = T) %>% 
      flatten() %>% 
      do.call(rbind, .)
    print(paste0("------ No. of images in rest of zoom: ", length(samprecs), " -----------"))

    ############################################################
    # Plot: Countries, fully dense sample, regularly dense sample
    ############################################################

    ##  Plot country outlines plus subregion dense sample
      g = ggplot(data=boundariesdf, aes(y=Latitude, x=Longitude)) +
        geom_path(data = boundariesdf, mapping = aes(x=long, y=lat, group=group), color = "black") +
        geom_polygon(data = samprecs, mapping = aes(x=long, y=lat, group=group), color = "red", size = .5) +
        geom_polygon(data = subregion, mapping = aes(x=long, y=lat, group=group), color = "red", size = .01) +
        labs(title= paste0(domain, " zoom"),
             x="Longitude", y= "Latitude",color = "")

    allsamples = rbind(samples[,1:2], subregdf)
    S <- dim(allsamples)[1]

    savedir <- file.path(res_dir, "figures/Fig4/zoom")
    dir.create(savedir, showWarnings = FALSE, recursive = TRUE)
    outfile <- file.path(savedir, paste0("dense_sample_", domain, "_", S, "_", sampling,".pdf"))
    pdf(outfile)
    g
    dev.off()

    ############################################################
    # Save .npz file for featurization
    ############################################################

    ############ Identify each obs as i,j using our full grid ID system ############

    # full lat and lon vectors for global grid
    latValsfull <- gridvals[[2]]
    lonValsfull <- gridvals[[1]]

    starttime <- Sys.time()
    for (n in 1:S) { # This should take about 2 minutes
      allsamples$i[n] <- which(latValsfull==allsamples$lat[n])
      allsamples$j[n] <- which(lonValsfull==allsamples$lon[n])
    }
    endtime <- Sys.time()
    endtime - starttime

    allsamples$ID <- paste(allsamples$i, allsamples$j, sep=",")

    ########### Export to npz 
    filename <- paste0("DenseSample_", domain, "_", as.character(zoom), "_", as.character(pixels), "_", 
                         sampling, "_", format(dim(allsamples)[1], scientific=FALSE))
    library(reticulate)
    np <- import("numpy")
    np$savez(file.path(grid_dir, paste0(filename, ".npz")), lon = allsamples$lon, lat = allsamples$lat, 
             ID = allsamples$ID, zoom = zoom, pixels = pixels)

    print("-------- Saved .npz! -----------")
}
