############################################################

# This script takes in a set of lat-lons (representing the GLOBAL subgrid we sample)
# and assigns a 2010 forest cover value at those points, which is the average forest cover 
# across the grid cell (using Hansen et al.)

# This part "extractTreecover_2" merges and saves the grid cell labels from each treecover tile. 
############################################################

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

###--- Set any constants ---###
sampling = "UAR" # Choose UAR or POP

# Are you running a random sample of the entire world, or a regular dense sample for Fig 4?
scope =  "dense" # "world" 

print("mergeing and saving...")

#Set the grid that we're extracting over
if (scope=="world"){
  fileEnding = paste0("WORLD_16_640_",sampling,"_1000000_0")
} else {
  fileEnding = "dense_16_640_regular_37674"
}
RLoad = function(fn) {
  return(get(load(fn)))
  
}

### List all the grid cell labels for each tile, then merge them all together
fl = list.files(file.path(data_dir,"int/applications/treecover/extracted_by_tile"), full.names = T)
fl = fl[grepl(x = fl, pattern = fileEnding)]
print(fl)
df <- do.call(rbind,lapply(fl,RLoad))
dt = data.table(df)
colnames(dt) = c("ID","lon","lat","treecover")

# Change format in case there are any issues
dt$treecover = as.numeric(dt$treecover)
dt$lon = as.numeric(dt$lon)
dt$lat = as.numeric(dt$lat)

fn = file.path(data_dir,"int/applications/treecover", paste0("outcomes_sampled_treecover_",fileEnding,".csv"))

write.csv(x = dt, file = fn)