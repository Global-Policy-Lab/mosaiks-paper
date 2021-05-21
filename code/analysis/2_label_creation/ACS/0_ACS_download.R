###################################################
### This script downloads the ACS data for many variables at the blockgroup level using the ACS R package.
### For each county in the US, it saves a file holding the income data for that county. 
### Information on the ACS variables can be found here: https://www.socialexplorer.com/data/ACS2015_5yr/metadata/?ds=ACS15_5yr

### Variables:
# #Minutes to get to work
# B08303 Travel Time to Work
# 
# #Percent Bachelor's Degree
# B15003 Educational Attainment for the Population 25 Years and Over [25]
# 
# #Median Household Income (same as before)
# B19013 Median Household Income in the Past 12 Months (In 2015 Inflation-Adjusted Dollars) [1]
# 
# #Per Capita Income 
# B19301 Per Capita Income in the Past 12 Months (In 2015 Inflation-Adjusted Dollars) [1]
# 
# #Percent below poverty level
# C17002 Ratio of Income to Poverty Level in the Past 12 Months [8]
# 
# #Percent households recieving foodstamps
# B22010 Receipt of Food Stamps/Snap in the Past 12 Months by Disability Status for Households [7]
# 
# #Rent as a percentage of income
# B25071 Median Gross Rent as a Percentage of Household Income in the Past 12 Months (Dollars) [1]
# 
# #Number of Housing Units
# B25001 Housing Units [1]
# 
# #Percent housing units vacant
# B25002 Occupancy Status [3] --are houses occupied
# 
# # Year Structure built
# B25035 Median Year Structure Built [1]
# 
# # Number of Rooms 
# B25017 Rooms [10]
# 
# #Median Value
# B25077 Median Value (Dollars) [1]

###################################################

rm(list=ls())

library(maps)
library(acs)
library(parallel)
library(foreach)
library(doParallel)
library(here)

## Import config.R to set filepaths
mosaiks_code <- Sys.getenv("MOSAIKS_CODE")
if (mosaiks_code=="") {
    mosaiks_code = here("code")
}
source(file.path(mosaiks_code,"mosaiks","config.R"))

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

###---###
library(acs)
### Get your own key here: https://api.census.gov/data/key_signup.html
keyString = "your key here"
if(keyString == "your key here"){print("update key"); break}
api.key.install(key=keyString) 
acs.tables.install()
###---###



# Set the number of cores: 
no_cores = 15 

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
tableNumber = "B19013" #Income

endYear = 2015


ACStableNumbers = c(
  "B08303",
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
ACSvariableNames = c(
  "MinToWork",
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
  "MedHouseValue")

#For each table download all the ACS data
j = 1
for(tableNumber in ACStableNumbers) {
  print(tableNumber)
  print(ACSvariableNames[j])
  ### Now I'm going to download and save a file for each county to a single folder:
  ### Then I can load that folder later. 
  cl = makeCluster(no_cores)
  registerDoParallel(cl)
  #numbers = foreach (i = 1:1, .combine = 'c',
  numbers = foreach (i = 1:length(countyfips), .combine = 'c',
                     .export = c(),
                     .packages=c("acs"), .errorhandling = 'pass') %dopar% {
  #for(i in 2388:2390) {                     
                       
                       
                       print(i)  
                       print("for county:")
                       print(paste(statefips[i],countyfips[i]))                  
                       fn = file.path(data_dir, "raw/applications/ACS/data_by_county",tableNumber, paste0(statefips[i],"_",countyfips[i],".csv"))
                       if(!file.exists(fn)) {
                         ### Download the data
                         out = tryCatch(
                           {
                           blockGroupForCounty = acs.fetch(geography = geo.make(state = statefips[i],  county = countyfips[i], tract = "*", block.group = "*"), table.number = tableNumber, endyear = endYear)
                           df = data.frame(estimate(blockGroupForCounty))
                           head(df)
                           
                           ### Do any simple pre-processing
                           ### This will end up with a data frame with a single variable column named "Val"
                           if(tableNumber == "B08303") {
                             #MinToWork
                             #Raw data is given as number of people who fall into bins of travel time to work.
                             #To create a single average from these binned values I assume that each person in a bin
                             #took the average of the min and max of the bin -- e.g. people in the 5-9 min bin got a value of (5+9)/2
                             #I sum up all the travel time, 
                             #and then divide by the number of people to get the average travel time for the county. 
                             df2 = data.frame(2.5*df$B08303_002 + (5+9)/2*df$B08303_003 +  (10+14)/2*df$B08303_004 +
                                                (15+19)/2*df$B08303_005 +
                                                (10+24)/2*df$B08303_006 +
                                                (25+29)/2*df$B08303_007 +
                                                (30+34)/2*df$B08303_008 +
                                                (35+39)/2*df$B08303_009 +
                                                (40+44)/2*df$B08303_010 +
                                                (45+59)/2*df$B08303_011 +
                                                (60+89)/2*df$B08303_012 + 
                                                (90+90)/2*df$B08303_013) / df$B08303_001
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B15003") {
                             #PctBachDeg
                             df2 = data.frame(df$B15003_022 / df$B15003_001) * 100
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B19013") {
                             #MedHHIncome
                             df2 = data.frame(df$B19013_001)
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B19301") {
                             #MedPerCapIncome
                             df2 = data.frame(df$B19301_001)
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "C17002") {
                             #PctBelowPov
                             df2 = data.frame( (df$C17002_003 + df$C17002_002)  / df$C17002_001) * 100
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B22010") {
                             #PctFoodStamp
                             df2 = data.frame( df$B22010_002  / df$B22010_001) * 100
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B25071") {
                             #PctIncomeRent
                             df2 = data.frame(df$B25071_001)
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B25001") {
                             #NumHouseUnits
                             df2 = data.frame(df$B25001_001)
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B25002") {
                             #PctVacant
                             df2 = data.frame(df$B25002_003 / df$B25002_001) * 100
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B25035") {
                             #YrBuilt
                             df2 = data.frame(df$B25035_001)
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B25017") {
                             #NumHouseRooms
                             df2 = data.frame( (1*df$B25017_002 +
                                                  2*df$B25017_003 +
                                                  3*df$B25017_004 +
                                                  4*df$B25017_005 +
                                                  5*df$B25017_006 +
                                                  6*df$B25017_007 +
                                                  7*df$B25017_008 +
                                                  8*df$B25017_009 +
                                                  9*df$B25017_010) / df$B25017_001)
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           } else if(tableNumber == "B25077") {
                             #MedHouseValue
                             df2 = data.frame(df$B25077_001)
                             rownames(df2) = rownames(df)
                             colnames(df2) = "Val"
                             df = df2
                             
                           }
                           
                           
                           df$state = statefips[i]
                           df$county = countyfips[i]
                           df$tract = blockGroupForCounty@geography$tract
                           df$blockgroup = blockGroupForCounty@geography$blockgroup
                           
                           dir.create(path = file.path(data_dir, "raw/applications/ACS/data_by_county",tableNumber))
                           
                           fn = file.path(data_dir, "raw/applications/ACS/data_by_county",tableNumber,paste0(statefips[i],"_",countyfips[i],".csv"))
                           print(fn)
                           write.csv2(x = df, file = fn)
                           print("saved")},
                           error=function(cond){
                             message(cond)
                             message(paste0(" with i = ",i))
                             return(NA)},
                           #warning = function(cond){
                            # message(cond)
                             #return(NA)},
                           finally = {
                             print("worked!")
                             
                           }
                         ) #end TryCatch
                       } #end if file exists
                       
                       
                       
                       return(i)
  }
  
  j = j + 1
}



print("DONE DONE DONE")

