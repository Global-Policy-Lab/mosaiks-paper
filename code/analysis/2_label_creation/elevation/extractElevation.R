####################################################
# This script uses the R package called elevatr, which has a function get_aws_terrain() to get elevation data.
# https://www.rdocumentation.org/packages/elevatr/versions/0.1.4/topics/get_aws_terrain. 
# It creates elvation labels for all grid cells within a subgrid. 
# Run this script for the US or for the dense sample shown in Figure 4; Run extractElevation_global for the global labels (it's paralellized).
####################################################

library(raster)
library(foreach)
library(doParallel)
library(reticulate)
library(elevatr)
library(ggplot2)
library(sp)
library(here)

rm(list=ls())

## Import config.R to set filepaths
mosaiks_code <- Sys.getenv("MOSAIKS_CODE")
if (mosaiks_code=="") {
    mosaiks_code = here("code")
}
source(file.path(mosaiks_code,"mosaiks","config.R"))

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

# Choose sampling
sampling <- "UAR" #"POP"  

# Are you running a random sample of the US, or a regular dense sample for Fig 4?
scope =  "dense" #"USA" # 

# Number of Cores
no_cores = 8

# Filename/path for subgrid
if (scope == "USA") {
  gridFile <- file.path(data_dir,"int/grids", paste0("CONTUS_16_640_", sampling, "_100000_0.npz"))
} else {
  gridFile <- file.path(data_dir,"int/grids", "DenseSample_elevation_16_640_regular_40931.npz")
}

np <- import("numpy")
npz1 <- np$load(gridFile)
npz1$files
sampLat = c(npz1$f[["lat"]])
sampLon = c(npz1$f[["lon"]])
zoom = npz1$f[["zoom"]]
pixels = npz1$f[["pixels"]]
ID = c(npz1$f[["ID"]])

####################################################
registerDoParallel(no_cores)

elev = foreach (i = 1:length(sampLat),
                     .combine = rbind,
                     .export = c("sampLat","sampLon", "ID"),
                     .packages = c("raster","elevatr")) %dopar% 
                     {
                       
                       #Source again so that each core has the necessary functions
                       source(file.path(utils_dir, "R_utils.R"))
                       
                       # If lon is too close to 180 or -180, the API will break
                       if(abs(round(sampLon[i],1))==180) {
                         
                         print(paste0(" ------- Too close to 180 degrees! API will break, enter NaN for iteration ", k, " obs ", i, " ----- "))
                         out = cbind(ID[i], sampLon[i], sampLat[i], NaN)
                         
                       } else {
                         
                         #Create 
                         recs = centroidsToTiles(lat = sampLat[i], lon = sampLon[i], zoom = zoom, numPix = pixels)
                         
                         # Throwing this in here to stop the weird "different resolution" error from the AWS API
                         outmean <- tryCatch({
                           mybbx = bbox(recs)
                           DEM = get_aws_terrain(mybbx, z = 8, prj = as.character(recs@proj4string))
                           DEM = projectRaster(from = DEM, crs = crs(recs))
                           
                           evals = extract(x = DEM, y = recs)[[1]]
                           
                           outmean = mean(evals, na.rm=T)
                         }, error=function(x){ 
                           message(paste0("----- Different resolution error, skipping observation ", i, "--------")) 
                           message("Here's the original error message:")
                           message(x)
                           return(NaN)}
                         )
                         
                       out = cbind(ID[i], sampLon[i], sampLat[i], outmean)
                       
                       # Print so we can track progress
                       if(round(i/1000) == i/1000) {
                         print(paste0("-------- Done with sample ", i, " of ", length(sampLat), " ---------"))
                       }
                       
                       rm(recs,DEM)
                       }
                       
                       return(out)

                     }

stopImplicitCluster()

elev = data.frame(elev, stringsAsFactors = F)
colnames(elev) = c("ID","lon","lat","elevation")
  
print('--------- Done extracting elevation ----------')

########################################################
# SAVE CSV OF LABELS
########################################################

if (scope == "USA") {
  fn = file.path(data_dir,"int/applications/elevation", paste0("outcomes_sampled_elevation_CONTUS_16_640_",sampling,"_100000_0.csv"))
} else {
  S = length(sampLat)
  fn = file.path(data_dir,"int/applications/elevation", paste0("outcomes_sampled_elevation_dense_16_640_regular_", S, ".csv"))
}
write.csv(x = elev, file = fn)

print('--------- Saved csv of labels ----------')

