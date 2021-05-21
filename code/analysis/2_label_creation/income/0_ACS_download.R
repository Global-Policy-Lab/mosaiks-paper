###################################################
### This script downloads the ACS median income data at the blockgroup level
### For each county in the US, it saves a file holding the income data for that county. 
###################################################

###---###
library(acs)
### Get in your own key here: https://api.census.gov/data/key_signup.html
keyString = "your key here"
if(keyString == "your key here"){print("update key"); break}
api.key.install(key=keyString) 
acs.tables.install()
###---###

library(maps)
library(acs)
library(parallel)
library(foreach)
library(doParallel)
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

# Set the number of cores: 
no_cores = 20 

### We can do this to get all the block groups within a single county. 
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

state = 06
county = 019
tableNumber = "B19013"
endYear = 2015

### Now I'm going to download and save a file for each county to a single folder:
### Then I can load that folder later. 
cl = makeCluster(no_cores)
registerDoParallel(cl)
numbers = foreach (i = 1:length(countyfips), .combine = 'c',
                               .export = c(),
                               .packages=c("acs") ) %dopar% {


  print(i)  
  print("for county:")
  print(paste(statefips[i],countyfips[i]))
  
  blockGroupForCounty = acs.fetch(geography = geo.make(state = statefips[i],  county = countyfips[i], tract = "*", block.group = "*"), table.number = tableNumber, endyear = endYear)
  df = data.frame(estimate(blockGroupForCounty))
  df$state = statefips[i]
  df$county = countyfips[i]
  df$tract = blockGroupForCounty@geography$tract
  df$blockgroup = blockGroupForCounty@geography$blockgroup
  
  fn = file.path(data_dir, "raw/applications/income/acs_blockgroup_income_by_county/",statefips[i],"_",countyfips[i],".csv")
  print(fn)
  write.csv2(x = df, file = fn)
  print("saved")
  return(i)
}

print("DONE DONE DONE")
