############################################################

# This script extracts raw high resolution (30x30 meter) treecover data over the US UAR subgrid for use in the 
# super resolution experiments. 

# For each grid cell it returns a treecover label for each of the 256x256 image pixel locations within the grid cell. 

# Note, running this entire script in full requires a lot of computation and storage.
# Thus, we advise users to run a subsample (by uncommenting the lines enabling subsampling in the code below) 
############################################################

## Load packages
library(raster)
library(foreach)
library(doParallel)
library(reticulate)
library(ggplot2)
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

fileEndings = c("CONTUS_16_640_UAR_100000_0")

fileEnding = fileEndings[1]
for(fileEnding in fileEndings) {
  print("fileEnding:")
  print(fileEnding)
  
  no_cores = 3
  registerDoParallel(no_cores)
  
  gridFile = file.path(data_dir,"int/grids", paste0(fileEnding,".npz"))
  
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
  tileLats = seq(50,30,-10)
  tileLons = seq(-130,-50,10)
  
  tilesll = expand.grid(tileLats,tileLons)
  
  cfun <- function(a, b) NULL
  
  treecover = foreach (r = 1:nrow(tilesll),
                       .combine = 'cfun',
                       .export = c("tilesll","sampLat","sampLon", "ID", "fileEnding"),
                       .packages = c("raster")) %dopar% 
                       {

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
                         
                         fn = file.path(data_dir,"raw/applications/treecover/treecover2010_v3", paste0(clat,"_",clon,"_treecover2010_v3.tif"))
                         saveFn = file.path(data_dir,"int/applications/treecover/extracted_by_tile", paste0("tile_",clat,"_",clon,"_treecover2010_v3_",fileEnding,".RData"))
                         
                         print(fn)
                         s = stack(fn)
                         
                         ### of all the points, limit to those that are in the tile
                         clat = tilesll[r,1]
                         clon = tilesll[r,2]
                         goodLat = sampLat<=clat & sampLat>(clat-10)
                         goodLon = sampLon>=clon & sampLon<(clon+10)
                         tileSampLat = sampLat[ goodLat & goodLon ]
                         tileSampLon = sampLon[ goodLat & goodLon ]
                        
                         #Create recs from centroids
                         crs = CRS(as.character("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
                         
                         if(length(tileSampLat)>0) {
                           recs = centroidsToTiles(lat = tileSampLat, lon = tileSampLon, zoom = zoom, numPix = pixels)
                           
                           print("Num Recs:")
                           print(length(recs))
                           
                           # for each tile, extract a 256 x 256 grid:
                           for (ri in 1:length(recs)) {
                             print(ri)
                             rec = recs[ri]
                             
                             #Within each rec create a 256 x 256 grid
                             #The extent is the same as the shape for the 
                             sr = raster(nrows = 256, ncol = 256)
                             extent(sr) = extent(rec)
                             #Get the centroids of that grid
                             cr = data.frame(coordinates(sr))
                             
                             #Extract those centroids over the treecover data
                             vals = extract(s, cr)
                             
                             #Rename colnames
                             goodInd = goodLat & goodLon
                             recID = ID[goodInd][ri]
                             df = cbind(recID,cr, vals)
                             colnames(df) = c("ID","lon","lat","treecover")
                             
                             #### Save
                             write.csv(x = df, file = file.path(data_dir,"int/applications/treecover/superRes/tile256Data", paste0("tileID_",recID,"_data.csv")))
                             
                             print("saved!")
                             
                          } #end for each rec in recs
                           return(TRUE)
                           
                         } #end if length(tileSampLat > 0)
                         
                       } #end parfor each tile
  
  stopImplicitCluster()
}

