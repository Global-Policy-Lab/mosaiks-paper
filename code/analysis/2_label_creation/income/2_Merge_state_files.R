
### This script merges the files that have subgrid income labels by state together into one file
### It averages the values of duplicated ids. 

library(data.table)
library(reticulate)
library(here)

rm(list=ls())

## Import config.R to set filepaths
mosaiks_code <- Sys.getenv("MOSAIKS_CODE")
if (mosaiks_code=="") {
    mosaiks_code = here("code")
}
source(file.path(mosaiks_code,"mosaiks/config.R"))

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

fileEndings = c("CONTUS_16_640_UAR_100000_0","CONTUS_16_640_POP_100000_0")
for(fileEnding in fileEndings) {
  fl = list.files(file.path(data_dir,"int/applications/income/extracted_by_state/"), full.names = T)
  fl = fl[grepl(pattern = fileEnding, x = fl, fixed = T)]
  
  dtList = lapply(as.list(fl), function(x) {get(load(x))})
  dtFull = do.call(rbind, dtList)
  
  namean = function(x){return(mean(x,na.rm=T))}
  
  out = dtFull[,list(income = namean(income)),by=ID]
  
  out$ID = as.character(out$ID)
  
  ### Drop NAs
  out = out[!is.na(out$income),]
  
  ### Add back all missing tile IDs so that things merge, and just fill income with -999:
  gridFile = file.path(data_dir,"int/grids/",fileEnding,".npz")
  np <- import("numpy")
  npz1 <- np$load(gridFile)
  npz1$files
  sampLat = c(npz1$f[["lat"]])
  sampLon = c(npz1$f[["lon"]])
  IDs = c(npz1$f[["ID"]])
  
  IDsToAdd = IDs[! (IDs %in% out$ID)]
  toAdd = data.frame(ID = IDsToAdd)
  toAdd$income = -999
  out = rbind(out,toAdd)
  
  fn = file.path(data_dir,"int/applications/income", paste0("outcomes_sampled_income_",fileEnding,".csv"))
  write.csv(x = out, file = fn)
}


