####################################################
# This script uses the R package called elevatr, which has a function get_aws_terrain() to get elevation data.
# https://www.rdocumentation.org/packages/elevatr/versions/0.1.4/topics/get_aws_terrain. 
# It creates elvation labels for all grid cells within a subgrid. 
# Run this script for the global labels.
####################################################

library(raster)
library(foreach)
library(doParallel)
library(reticulate)
library(elevatr)
library(ggplot2)
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

# Number of Cores
no_cores = 8

# Filename/path for subgrid
gridFile <- file.path(data_dir,"int/grids", paste0("CONTUS_16_640_", sampling, "_100000_0.npz"))

np <- import("numpy")
npz1 <- np$load(gridFile)
npz1$files
sampLat = c(npz1$f[["lat"]])
sampLon = c(npz1$f[["lon"]])
zoom = npz1$f[["zoom"]]
pixels = npz1$f[["pixels"]]
ID = c(npz1$f[["ID"]])

##############################################################################
# Loop over 10k groupings of data, save intermediate files, parallelize WITHIN each  10k
##############################################################################

registerDoParallel(no_cores)

# This vector indexes each 10k obs in the 1e6 set of lat-lons
kk <- rep(1:100, each=length(sampLat)/100)

# Divide into 100 pieces = 1e6/10k
for(k in 1:100) {
  
  sampLatk <- sampLat[kk==k]
  sampLonk <- sampLon[kk==k]
  IDk <- ID[kk==k]
  
  elev = foreach (i = 1:length(sampLatk),
                .combine = rbind,
                .export = c("sampLatk","sampLonk", "IDk"),
                .errorhandling = "remove",
                .packages = c("raster","elevatr")) %dopar% 
                {
                  #Source again so that each core has the necessary functions
                  source(file.path(utils_dir, "R_utils.R"))
                  
                  # If lon is too close to 180 or -180, the API will break
                  if(abs(round(sampLonk[i],1))==180) {
                    
                    print(paste0(" ------- Too close to 180 degrees! API will break, enter NaN for iteration ", k, " obs ", i, " ----- "))
                    out = cbind(IDk[i], sampLonk[i], sampLatk[i], NaN)
                  
                  } else {
                  
                  #Create 
                  recs = centroidsToTiles(lat = sampLatk[i], lon = sampLonk[i], zoom = zoom, numPix = pixels)
                  
                  # Throwing this in here to stop the weird "different resolution" error from the AWS API
                  outmean <- tryCatch({
                    mybbx = bbox(recs)
                    DEM = get_aws_terrain(mybbx, z = 8, prj = as.character(recs@proj4string))
                    DEM = projectRaster(from = DEM, crs = crs(recs))
                  
                    evals = extract(x = DEM, y = recs)[[1]]
                  
                    outmean = mean(evals, na.rm=T)
                  }, error=function(x){ 
                    message(paste0("----- Different resolution error, skipping iteration ", k, " observation ", i, "--------")) 
                    message("Here's the original error message:")
                    message(x)
                    return(NaN)}
                  )
                  
                  out = cbind(IDk[i], sampLonk[i], sampLatk[i], outmean)
                  
                  # Print so we can track progress
                  if(round(i/1000) == i/1000) {
                    print(paste0("-------- Done with sample ", i, " of ", length(sampLatk), " ---------"))
                  }
                  
                  }
                  return(out)
                }
  
  #rbind all the 10k obs together
  elev = data.frame(elev, stringsAsFactors = F)
  colnames(elev) = c("ID","lon","lat","elevation")

  # Save Rdata of labels
  fn = file.path(data_dir,"int/applications/elevation/extracted_by_10k", paste0("iteration_", k ,"_outcomes_sampled_elevation_WORLD_16_640_",sampling,"_1000000_0.RData"))
  print(paste0("------- Saving iteration ", k, " of 100 ---------"))
  print(fn)
  save(file = fn, list = c("elev"))
  print("---------- Saved ------------")
  
}
  
  stopImplicitCluster()
  print("--------- DONE EXTRACTING BY 10K ITERATIONS ----------")
  
########################################################
# SAVE CSV OF LABELS 
########################################################

savefn = file.path(data_dir,"int/applications/elevation", paste0("outcomes_sampled_elevation_WORLD_16_640_",sampling,"_1000000_0.csv"))
fileend <- paste0("outcomes_sampled_elevation_WORLD_16_640_",sampling,"_1000000_0.RData")

# Loop over and load all 100 iterations already saved
  print(" -------- Merging and saving full dataset ----------")
  
  library(data.table)
  
  RLoad = function(fn) {
    return(get(load(fn)))
  }
  
  fl = list.files(file.path(data_dir,"int/applications/elevation/extracted_by_10k"), full.names = T)
  fl = fl[grepl(x = fl, pattern = fileend)]
  print(fl)
  df <- do.call(rbind,lapply(fl,RLoad))
  dt = data.table(df)
  colnames(dt) = c("ID","lon","lat","elevation")
  
  write.csv(x = dt, file = savefn)

  print(" -------- FULL DATASET SAVED ----------")
