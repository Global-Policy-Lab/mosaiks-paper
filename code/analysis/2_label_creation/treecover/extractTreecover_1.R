############################################################

# This script takes in a set of lat-lons (representing the subgrid we sample)
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
source(file.path(mosaiks_code,"mosaiks" ,"config.R"))

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

###--- Set any constants ---###
sampling = "UAR" # Choose UAR or POP

### Change this depending on whether you want to make the labels for the UAR or POP subgrid. 
fileEnding = paste0("CONTUS_16_640_", sampling, "_100000_0")

no_cores = 20
registerDoParallel(no_cores)

### Load the subgrid
gridFile = file.path(data_dir,"int/grids", paste0(fileEnding,".npz"))

np <- import("numpy")
npz1 <- np$load(gridFile)
npz1$files
sampLat = c(npz1$f[["lat"]])
sampLon = c(npz1$f[["lon"]])
zoom = npz1$f[["zoom"]]
pixels = npz1$f[["pixels"]]
ID = c(npz1$f[["ID"]])

### Given these lat and lon, extract data from the correct treecover tile. 
tileLats = seq(50,30,-10)
tileLons = seq(-130,-50,10)
tilesll = expand.grid(tileLats,tileLons)
#For each of these tiles extract the points that are within that tile and save their associated values
#the name of the file gives the top left corner of the tile box
treecover = foreach (r = 1:nrow(tilesll),
                     .combine = rbind,
                     .export = c("tilesll","sampLat","sampLon", "ID", "fileEnding"),
                     .packages = c("raster")) %dopar% 
                     {
                       
                       print(paste0("r is: ",r))
                       
                       #Source again so that each core has the necessary functions
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
                       
                       crs = CRS(as.character("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
                       
                       #Create 
                       if(length(tileSampLat)>0) {
                         recs = centroidsToTiles(lat = tileSampLat, lon = tileSampLon, zoom = zoom, numPix = pixels)
                         
                         print('Extracting...')
                         out = extract(x = s, y = recs, fun = mean)
                         print("Done w tile")
                         
                         goodInd = goodLat & goodLon
                         
                         out = cbind(ID[goodInd], sampLon[goodInd], sampLat[goodInd], out)
                         #print(out)
                         
                         print("Saving....")
                         print(saveFn)
                         save(file = saveFn, list = c("out"))
                         print("Saved")
                       }
                       
                       return(r)
                     }

stopImplicitCluster()

print("DONE EXTRACTING!!!!")



