############################################################

# This script takes in a set of lat-lons (representing the GLOBAL subgrid we sample)
# and assigns a 2010 forest cover value at those points, which is the average forest cover 
# across the grid cell (using Hansen et al.)

# This part "extractTreecover_1" creates separate files for each treecover tile 
# and then "extractTreecover_2" merges them all together
############################################################

library(raster)
library(foreach)
library(doParallel)
library(reticulate)
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

# Choose sampling
sampling <- "UAR" # or "POP"

# Are you running a random sample of the entire world, or a regular dense sample for Fig 4?
scope =  "dense" # "world" 

no_cores = 6
registerDoParallel(no_cores)

if (scope=="world") {
  fileEnding = paste0("WORLD_16_640_",sampling,"_1000000_0")
  gridFile = file.path(data_dir,"int/grids", paste0(fileEnding,".npz"))
} else {
  fileEnding = "dense_16_640_regular_37674"
  gridFile = file.path(data_dir,"int/grids/DenseSample_treecover_16_640_regular_37674.npz")
}

np <- import("numpy")
npz1 <- np$load(gridFile)
npz1$files
sampLat = c(npz1$f[["lat"]])
sampLon = c(npz1$f[["lon"]])
zoom = npz1$f[["zoom"]]
pixels = npz1$f[["pixels"]]
ID = c(npz1$f[["ID"]])

### Given these lat and lon, extract data from the correct tile. 
#For each of these tiles extract the points that are within that tile and save their associated values
#the name of the file gives the top left corner of the tile box
if (scope == "world"){
  tileLatsWorld = seq(80,-50,-10)
  tileLonsWorld = seq(-180,170,10)
} else {
  tileLatsWorld = seq(0,-20,-10)
  tileLonsWorld = seq(-90,-60,10)
}

tileLats = tileLatsWorld
tileLons = tileLonsWorld

tilesll = expand.grid(tileLats,tileLons)


treecover = foreach (r = 1:nrow(tilesll),
         .combine = rbind,
         .export = c("tilesll","sampLat","sampLon", "ID", "fileEnding"),
         .packages = c("raster")) %dopar% 
  {
  print(r)
    source(file.path(utils_dir, "R_utils.R"))
    
  ###load the correct tile
  clat = tilesll[r,1]
  clon = tilesll[r,2]
  
  if(clat<0) {
    clat = paste0(abs(clat),"S")
  } else {
    clat = paste0(clat,"N")
  }
  
  if(clon<0) {
    clon = paste0(abs(clon),"W")
  } else {
    clon = paste0(clon,"E")
  }
  
  if(nchar(clon)<4) {
    clon = paste0(0,clon)
  }
  if(nchar(clat)<3) {
    clat = paste0(0,clat)
  }
  #Add an extra zero for this special case
  if(clon == "00E") {
    clon = paste0(0,clon)
  }
  
  fn = file.path(data_dir,"raw/applications/treecover/treecover2010_v3", paste0(clat,"_",clon,"_treecover2010_v3.tif"))
  saveFn = file.path(data_dir,"int/applications/treecover/extracted_by_tile", paste0("tile_",clat,"_",clon,"_treecover2010_v3_",fileEnding,".RData"))
  
  ### If you want to actually re-compute the files you'll need to delete them first.
  if(file.exists(saveFn)){
    print("this file already extracted and saved:")
    print(saveFn)
    next
  }
  
  print(fn)
  s = stack(fn)
  
  ### of all the points, limit to those that are in the tile
  clat = tilesll[r,1]
  clon = tilesll[r,2]
  
  goodLat = sampLat<=clat & sampLat>(clat-10)
  goodLon = sampLon>=clon & sampLon<(clon+10)
  tileSampLat = sampLat[ goodLat & goodLon ]
  tileSampLon = sampLon[ goodLat & goodLon ]
  
  crs = CRS(as.character("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
  
  #Create 
  if(length(tileSampLat)>0) {
    recs = centroidsToTiles(lat = tileSampLat, lon = tileSampLon, zoom = zoom, numPix = pixels)
    
    print('Extracting...')
    out = extract(x = s, y = recs, fun = mean)
    print("Done w tile")
    
    goodInd = goodLat & goodLon
    
    out = cbind(ID[goodInd], sampLon[goodInd], sampLat[goodInd], out)

    print("Saving....")
    print(saveFn)
    save(file = saveFn, list = c("out"))
    print("Saved")
  } else {
    print("This tile is empty; skipping it!")
  }
  
  return(TRUE)
}

stopImplicitCluster()
print("DONE EXTRACTING!!!!")

