############################################################

# This script takes in a set of lat-lons (representing the subgrid we sample)
# and assigns a nightlights value from the VIIRS nightlights dataset from 2015 to each of the 
# lat-lons. 

# Note: VIIRS value is the average day-night band (DNB) radiance (avg_rade9h)
# Note: This script does not provide code to download the raw nighttime lights rasters, which are now stored at
#       the Colorado School of Mines and are kept behind an access wall. Formerly, these data were publicly available
#       at NOAA. To download the raw imagery, a user simply needs to register for a free account at https://eogdata.mines.edu/products/vnl/.
#       Each of the 6 global tiles can be manually downloaded, or you can obtain programmatic access once you are 
#       registered for an account. For details on programmatic access in Python and R, see here: 
#       https://payneinstitute.mines.edu/eog-2/transition-to-secured-data-access/.
#       This code will run when raw .tif files are saved in the data/raw/applications/nightlihghts/ directory. 

############################################################

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

## User choices: Sampling, global/USA, raw data download
# Sampling
sampling <- "UAR" #"POP"  
# Is your scope global, within the US, or a dense sample for Figure 4? 
scope <- "dense" # "global" "USA"
# Number of Cores
no_cores = 8

# Filename/path for subgrid
if(scope=="USA") {
  subgridpath <- file.path(data_dir,"int/grids", paste0("CONTUS_16_640_", sampling, "_100000_0.npz"))
} else if(scope=="global") {
  subgridpath <- file.path(data_dir, "int/grids", paste0("WORLD_16_640_", sampling, "_1000000_0.npz"))
} else {
  subgridpath <- file.path(data_dir, "int/grids","DenseSample_nightlights_16_640_regular_36367.npz")
}

# Packages
library(raster)
library(foreach)
library(doParallel)
library(reticulate)
library(elevatr)
library(ggplot2)

############## Pull in lat-lon samples 

np <- import("numpy")
npz1 <- np$load(subgridpath)
npz1$files
npz1$f[["pixels"]]
class(npz1$f[["lat"]])
head(npz1$f[["ID"]])

# Add in the hashes to the subgrid data frame
lon <- npz1$f[["lon"]]
lat <- npz1$f[["lat"]]
ID <- npz1$f[["ID"]]
zoom = npz1$f[["zoom"]]
pixels = npz1$f[["pixels"]]
subgrid <- data.frame(lon=c(lon), lat=c(lat), ID=c(ID), stringsAsFactors=FALSE)

############## USA LOOP #########################

if(scope == "USA" ) {
  
  print("NOTE: code will only run once raw data from 2015 is already downloaded by the user after registering for a free account!")
  
  ### Load raw night lights data (Tile 1 from VIIRS only needed for continental US) 
  nltile <- raster(file.path(data_dir, "raw/applications/nightlights/SVDNB_npp_20150101-20151231_75N180W_vcm-ntl_v10_c201701311200.avg_rade9.tif"))
  
  ### Turn the subgrids into recs (i.e. shapes)
  crs = CRS(as.character("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
  recs = centroidsToTiles(lat = subgrid$lat, lon = subgrid$lon, zoom = zoom, numPix = pixels)
  
  print("--------- Created image shapes -----------")
  
  ### Extract: Average the nightlights value from the cells overlapping the image
  nlmatch <- extract(nltile, recs, fun = mean, df=TRUE)
  colnames(nlmatch) <- c("sampleid", "luminosity")
  
  print("--------- Extraction of nightlights over images done -----------")
  
  # Merge back in with the subgrid (same order, hasn't changed, so can just append it)
  subgrid_nl <- cbind(subgrid, nlmatch[,2])
  subgrid_nl <- subgrid_nl[c(3,1,2,4)]
  colnames(subgrid_nl) <- c("ID", "lon", "lat", "y")
  
  # Save output
  write.csv(subgrid_nl, file=file.path(data_dir, "int/applications/nightlights", paste0("outcomes_sampled_nightlights_CONTUS_16_640_", sampling, "_100000_0.csv")), row.names = FALSE)
  
  print("--------- Output saved: DONE DONE DONE -----------")
}

############## GLOBAL LOOP #########################

if(scope == "global" ) {
  
  # we will loop over all tiles, since the raw data tile up the earth
  tiles <- c("00N060E", "00N060W","00N180W"	, "75N060E","75N060W"	, "75N180W")

  print("NOTE: code will only run once raw data from 2015 is already downloaded by the user after registering for a free account!")
  
  ### For each tile: load data, extract, save matched to a data.frame
  registerDoParallel(no_cores)
  
  for(tt in tiles) {
    
    # Load raw night lights for this tile
    nltile <- raster(file.path(data_dir, "raw/applications/nightlights", paste0("SVDNB_npp_20150101-20151231_", tt,"_vcm-ntl_v10_c201701311200.avg_rade9.tif")))
    ext <- extent(nltile)
    
    # First restrict subgrid to only values within the extent of the tile
    subgridTile <- subset(subgrid, subgrid$lon>ext[1] & subgrid$lon<ext[2] & subgrid$lat>ext[3] & subgrid$lat<ext[4])
  
    lights = foreach (i = 1:length(subgridTile$lon),
                      .combine = rbind,
                      .export = c("lat","lon", "ID"),
                      .packages = c("raster")) %dopar% 
                      {
                        
                        #Source again so that each core has the necessary functions
                        source(file.path(utils_dir, "R_utils.R"))
                        
                        ### Turn the subgrids into recs (i.e. shapes)
                        crs = CRS(as.character("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
                        recs = centroidsToTiles(lat = subgridTile$lat[i], lon = subgridTile$lon[i], zoom = zoom, numPix = pixels)
                        
                        ### Extract: Average the nightlights value from the cells overlapping the image
                        nlmatch <- extract(nltile, recs, fun = mean, df=TRUE)
                        colnames(nlmatch) <- c("sampleid", "luminosity")
                        
                        out = cbind(subgridTile$ID[i], subgridTile$lon[i], subgridTile$lat[i], nlmatch$luminosity)
                        
                        # Print so we can track progress
                        if(round(i/1000) == i/1000) {
                          print(paste0("-------- Done with sample ", i, " of ", length(subgridTile$lon), " ---------"))
                        }
                        
                        return(out)
                      }
    
    #rbind all the obs within a single nightlights tile together
    lights = data.frame(lights, stringsAsFactors = F)
    colnames(lights) = c("ID","lon","lat","luminosity")
    lights$lon <- as.numeric(lights$lon)
    lights$lat <- as.numeric(lights$lat)
    lights$luminosity <- as.numeric(lights$luminosity)
    
    # Save Rdata of labels
    fn = file.path(data_dir,"int/applications/nightlights/extracted_by_tile", paste0("tile", tt ,"_outcomes_sampled_nightlights_WORLD_16_640_",sampling,"_1000000_0.RData"))
    print(paste0("------- Saving iteration ", tt, " of 8 ---------"))
    print(fn)
    save(file = fn, list = c("lights"))
    print("---------- Saved ------------")
    
  }
  
  stopImplicitCluster()
  print("--------- DONE EXTRACTING BY TILE ----------")
  
  
  ########################################################
  # SAVE CSV OF LABELS 
  ########################################################
  
  savefn = file.path(data_dir,"int/applications/nightlights", paste0("outcomes_sampled_nightlights_WORLD_16_640_",sampling,"_1000000_0.csv"))
  fileend <- paste0("outcomes_sampled_nightlights_WORLD_16_640_",sampling,"_1000000_0.RData")
  
  # Loop over and load all 100 iterations already saved
  print(" -------- Merging and saving full dataset ----------")
  
  library(data.table)
  
  RLoad = function(fn) {
    return(get(load(fn)))
  }
  
  fl = list.files(file.path(data_dir,"int/applications/nightlights/extracted_by_tile"), full.names = T)
  fl = fl[grepl(x = fl, pattern = fileend)]
  print(fl)
  df <- do.call(rbind,lapply(fl,RLoad))
  dt = data.table(df)
  colnames(dt) = c("ID","lon","lat","luminosity")
  
  write.csv(x = dt, file = savefn)
  
  print(" -------- FULL DATASET SAVED ----------")
  
}


############## DENSE SAMPLE LOOP #########################

if(scope == "dense" ) {

    #note: extent of the dense sample is: 
    extsubgrid = cbind(min(subgrid$lon), max(subgrid$lon), min(subgrid$lat), max(subgrid$lat))
    
    # this means we just need the "75N060E" tile from VIIRS
    tiles <- c("00N060E", "00N060W","00N180W"	, "75N060E","75N060W"	, "75N180W")
    tt = tiles[4]
    nltile <- raster(file.path(data_dir, "raw/applications/nightlights", paste0("SVDNB_npp_20150101-20151231_", tt,"_vcm-ntl_v10_c201701311200.avg_rade9.tif")))
    ext <- extent(nltile)
    ext
    
    registerDoParallel(no_cores)

    lights = foreach (i = 1:length(subgrid$lon),
                        .combine = rbind,
                        .export = c("lat","lon", "ID"),
                        .packages = c("raster")) %dopar% 
                        {
                          
                          #Source again so that each core has the necessary functions
                          source(file.path(utils_dir, "R_utils.R"))
                          
                          ### Turn the subgrids into recs (i.e. shapes)
                          crs = CRS(as.character("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
                          recs = centroidsToTiles(lat = subgrid$lat[i], lon = subgrid$lon[i], zoom = zoom, numPix = pixels)
                          
                          ### Extract: Average the nightlights value from the cells overlapping the image
                          nlmatch <- extract(nltile, recs, fun = mean, df=TRUE)
                          colnames(nlmatch) <- c("sampleid", "luminosity")
                          
                          out = cbind(subgrid$ID[i], subgrid$lon[i], subgrid$lat[i], nlmatch$luminosity)
                          
                          # Print so we can track progress
                          if(round(i/1000) == i/1000) {
                            print(paste0("-------- Done with sample ", i, " of ", length(subgrid$lon), " ---------"))
                          }
                          
                          return(out)
                        }
      
    stopImplicitCluster()
    print("--------- DONE EXTRACTING SINGLE TILE WE NEED FOR ZOOM ----------")
    
    # clean up data frame
    lights = data.frame(lights, stringsAsFactors = F)
    colnames(lights) = c("ID","lon","lat","luminosity")
    lights$lon <- as.numeric(lights$lon)
    lights$lat <- as.numeric(lights$lat)
    lights$luminosity <- as.numeric(lights$luminosity)
      
    # save
    S = dim(subgrid)[1]
    savefn = file.path(data_dir,"int/applications/nightlights", paste0("outcomes_sampled_nightlights_dense_16_640_regular_", S, ".csv"))
    
    write.csv(x = lights, file = savefn)
    
    print(" -------- DENSE SAMPLE DATASET SAVED ----------")
}