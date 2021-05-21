
############################################################

# This script takes in a set of lat-lons (representing the subgrid we sample)
# and assigns an ACS value for many outcomes by taking the average of the 
# value over each grid cell 

############################################################

## Load packages
library(maps)
library(maptools)
library(rgdal)
library(raster)
library(sp)
library(parallel)
library(foreach)
library(doParallel)
library(reticulate)
library(data.table)
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

###### Helper Functions ##########
read.csv2Me = function(x) {
  return(read.csv2(x,stringsAsFactors = F))
}

padZeros = function(num, dd) {
  #num is the input number
  #dd is the number of desired digits
  num = as.character(num)
  
  if(nchar(num)>dd){print("too long already!"); return(num)}
  
  while(nchar(num)<dd) {
    num = paste0("0",num)
  }
  return(num)
}

########################

ACStableNumbers = c("B08303",
                    "B15003",
                    "B19013",
                    "B19301",
                    "C17002",
                    "B22010",
                    "B25071",
                    "B25001",
                    "B25002",
                    "B25035",
                    "B25017",
                    "B25077")
ACSvariableNames = c("MinToWork",
                     "PctBachDeg",
                     "MedHHIncome",
                     "MedPerCapIncome",
                     "PctBelowPov",
                     "PctFoodStamp",
                     "PctIncomeRent",
                     "NumHouseUnits",
                     "PctVacant",
                     "YrBuilt",
                     "NumHouseRooms",
                     "MedHouseValue"
)

#For each table download all the ACS data
j = 1
for(tableNumber in ACStableNumbers) {
  print(ACStableNumbers[j])
  print(ACSvariableNames[j])
  fileEnding = "CONTUS_16_640_UAR_100000_0"
  #Extract one, then the other:
  fileEndings = c("CONTUS_16_640_UAR_100000_0","CONTUS_16_640_POP_100000_0")
  ### For each subgrid:
  for(fileEnding in fileEndings) {
    ### Load the subgrid:
    print("File:")
    print(fileEnding)
    gridFile = paste0(data_dir,"int/grids/",fileEnding,".npz")
    np <- import("numpy")
    npz1 <- np$load(gridFile)
    npz1$files
    sampLat = c(npz1$f[["lat"]])
    sampLon = c(npz1$f[["lon"]])
    zoom = npz1$f[["zoom"]]
    pixels = npz1$f[["pixels"]]
    ID = c(npz1$f[["ID"]])
    
    ### Get everything that we're gonna want to loop over:
    statefips = unique(state.fips$fips)
    #take only the last three parts of the county fips
    tmp = county.fips
    tmp = tmp$fips
    tmp = as.character(tmp)
    countyfips = as.numeric(substr(x = tmp, start = (nchar(tmp) - 2), stop = nchar(tmp)))
    
    #make the state fips as the first part of the county fips
    statefips = tmp
    statefips[nchar(tmp)==4] = substr(x = statefips[nchar(tmp)==4], start = 1, stop = 1)
    statefips[nchar(tmp)==5] = substr(x = statefips[nchar(tmp)==5], start = 1, stop = 2)
    statefips = as.numeric(statefips)
    
    state = 44
    county = 007
    endYear = 2015
    
    ### For each state:
    no_cores = 10
    cl = makeCluster(no_cores)
    registerDoParallel(cl)
    numbers = foreach (state = unique(statefips), .combine = 'c',
                       .export = c(),
                       .packages=c("maps","maptools","rgdal","raster","sp","reticulate","data.table")) %dopar% {

      print("State:")
      print(state)
      fn = file.path(data_dir,"int/applications/ACS",tableNumber,"extracted_by_state", paste0("ACSByTile_",fileEnding,"_",state,".RData"))
      if(file.exists(fn)) {
        print("already made")
      } else {
        ## Load the data:
        # Load all the income data for that state
        fl = list.files(file.path(data_dir,"raw/applications/ACS/data_by_county",tableNumber), full.names = T)
        fl = fl[grepl(pattern = paste0("/",state,"_"), x = fl, fixed = T)]
        
        dtList = lapply(as.list(fl), function(x) {read.csv2Me(x)})
        dfACS = do.call(rbind, dtList)
        
        # Load the blockgroup shapefile for that state
        if(state<10) {
          shpFolder = file.path(data_dir,"raw/applications/income/blockgroup_shps", paste0("cb_2017_0",state,"_bg_500k"))
          shpName = paste0("cb_2017_0",state,"_bg_500k")
          
        } else {
          shpFolder = file.path(data_dir,"raw/applications/income/blockgroup_shps", paste0("cb_2017_",state,"_bg_500k"))
          shpName = paste0("cb_2017_",state,"_bg_500k")
          
        }
        shp = readOGR(dsn = shpFolder, layer = shpName, stringsAsFactors = F)
        
        # Load all the tiles within that state (or within the extent/bounding box of the state)
        shpExtent = extent(shp)
        goodInds = sampLat>shpExtent@ymin & sampLat<shpExtent@ymax & sampLon<shpExtent@xmax & sampLon>shpExtent@xmin
        
        stateLat = sampLat[goodInds]
        stateLon = sampLon[goodInds]
        stateID = ID[goodInds]
        
        ## Make tiles
        crs = CRS(as.character("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
        
        #Create recs
        recs = centroidsToTiles(lat = stateLat, lon = stateLon, zoom = zoom, numPix = pixels)
        
        #add attribute data to the recs
        recs = SpatialPolygonsDataFrame(recs, data.frame(ID=stateID), match.ID=F)
        
        #make the shp the same crs as the recs
        shp = spTransform(shp, crs(recs))
        
        ### Connect the income data to the shapes
        #Make the GEOID from the ACS data to merge onto the shpfile
        dfACS$GEOID = paste0(unlist(lapply(X = dfACS$state, FUN = padZeros, dd=2)),
                             unlist(lapply(X = dfACS$county, FUN = padZeros, dd=3)),
                             unlist(lapply(X = dfACS$tract, FUN = padZeros, dd=6)),
                             as.character(dfACS$blockgroup))
        
        shp = sp::merge(x = shp, y = dfACS, by = "GEOID", all.x=T)
        
        ## Determine which shapes connect to which tile (ideally the area of each block group in each tile)
        inter = intersect(x = recs, y = shp)
        inter$area <- area(inter) / 1000000
        
        ## For each tile in the state, calculate a weighted average of the income data
        ## where the weight is the area of the subgroup in the tile. 
        dt = data.table(inter@data)
        out = dt[,list(Val = weighted.mean(Val,area)),by=ID]
        
        ## Save this in a new folder
        dir.create(path = file.path(data_dir,"int/applications/ACS",tableNumber,"extracted_by_state"), recursive = T)
        fn = file.path(data_dir,"int/applications/ACS",tableNumber,"extracted_by_state", paste0("ACSByTile_",fileEnding,"_",state,".RData"))
        print(fn)
        save(x = out, file = fn)
        print("saved")
      }#end if file exists
      return(1)
    } #end foreach state
    parallel::stopCluster(cl)
  } #end fileEndings
  j = j + 1
} #End for tableNumber


print("DONE DONE DONE")
