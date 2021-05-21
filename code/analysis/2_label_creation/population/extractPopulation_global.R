
############################################################

# This script takes in a set of lat-lons (representing the subgrid we sample GLOBALLY)
# and assigns a population density value by taking the average of the 
# Gridded Population of the World (GPW) dataset over each grid cell 

############################################################

# Packages
library(raster)
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
source(file.path(mosaiks_code,"mosaiks", "config.R"))

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

# Sampling
sampling <- "UAR" #"POP"  

# Filename/path for subgrid
fileEnding = "CONTUS_16_640_POP_100000_0"
gridFile = file.path(data_dir,"int/grids", paste0(fileEnding,".npz"))

############## Pull in lat-lon samples 
np <- import("numpy")
npz1 <- np$load(gridFile)
npz1$files
sampLat = c(npz1$f[["lat"]])
sampLon = c(npz1$f[["lon"]])
zoom = npz1$f[["zoom"]]
pixels = npz1$f[["pixels"]]
ID = c(npz1$f[["ID"]])

#Load Raster
pop = raster(file.path(data_dir,"raw/applications/population/gpw_v4_population_density_rev10_2015_30_sec.tif"))
          
crs = CRS(as.character("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))

#Create 
recs = centroidsToTiles(lat = sampLat, lon = sampLon, zoom = zoom, numPix = pixels)

#crop the pop raster to the recs
e = extent(recs)
delta = .1
e@xmin = e@xmin - delta
e@xmax = e@xmax + delta
e@ymin = e@ymin - delta
e@ymax = e@ymax + delta
pop = crop(pop, e)


print('Extracting...')
out = extract(x = pop, y = recs, fun = mean)

df = cbind(ID, sampLon, sampLat, out)
colnames(df) = c("ID","lon","lat","population")
dt = data.table(df)

fn = file.path(data_dir,"int/applications/population", paste0("outcomes_sampled_population_",fileEnding,".csv"))
write.csv(x = df, file = fn)

print("DONE DONE DONE")




