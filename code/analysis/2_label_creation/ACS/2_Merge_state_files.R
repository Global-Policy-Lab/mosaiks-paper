
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
source(file.path(mosaiks_code,"mosaiks","config.R"))

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

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
                    "B25077"
                    )
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
tableNumber = ACStableNumbers[j]
for(tableNumber in ACStableNumbers) {
  print(tableNumber)
  print(ACSvariableNames[j])
  fileEndings = c("CONTUS_16_640_UAR_100000_0","CONTUS_16_640_POP_100000_0")
  fileEnding = fileEndings[1]
  for(fileEnding in fileEndings) {
    print(fileEnding)
    fl = list.files(file.path(data_dir,"int/applications/ACS",tableNumber,"extracted_by_state"), full.names = T)
    fl = fl[grepl(pattern = fileEnding, x = fl, fixed = T)]
    
    dtList = lapply(as.list(fl), function(x) {get(load(x))})
    dtFull = do.call(rbind, dtList)
    
    namean = function(x){return(mean(x,na.rm=T))}
    
    out = dtFull[,list(Val = namean(Val)),by=ID]
    
    out$ID = as.character(out$ID)
    
    ### Do any data control; set all bad values to -999, which will then
    #   be dropped later in the pipeline. 

    #set NAs to -999
    out$Val[is.na(out$Val)] = -999
    #set negative values to -999
    out$Val[out$Val < 0] = -999
    
    ### Add back all missing tile IDs so that things merge
    gridFile = file.path(data_dir,"int/grids", paste0(fileEnding,".npz"))
    np <- import("numpy")
    npz1 <- np$load(gridFile)
    npz1$files
    sampLat = c(npz1$f[["lat"]])
    sampLon = c(npz1$f[["lon"]])
    IDs = c(npz1$f[["ID"]])
    
    IDsToAdd = IDs[! (IDs %in% out$ID)]
    toAdd = data.frame(ID = IDsToAdd)
    toAdd$Val = -999
    out = rbind(out,toAdd)
    
    fn = file.path(data_dir,"int/applications/ACS",tableNumber, paste0("outcomes_sampled_",tableNumber,"_",fileEnding,".csv"))
    print(fn)
    write.csv(x = out, file = fn)
    print("saved")
  } #end for fileEndings
  j = j + 1
} #end for tableNumber



