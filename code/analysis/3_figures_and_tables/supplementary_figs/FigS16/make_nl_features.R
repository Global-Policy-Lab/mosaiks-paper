####----
### This script makes nighttime lights features for the comparison of MOSAIKS to Head et al (2017).
### Run once for each country.
### Specifically, it creates, for each observation a 10km by 10km box around the given location of the observation and then calculates
### the min, max and mean of NL values as well as bin features which are the percent of the area covered by nl values within 19 different 
### log spaced bins from a value of .1 to 500. 

library(raster)
library(data.table)
library(emdbook)
library(here)

rm(list=ls())

# Identify which countries you would like to create NL features for
countries_to_run <- c("haiti", "rwanda", "nepal")

## Import config.R to set filepaths
mosaiks_code <- Sys.getenv("MOSAIKS_CODE")
if (mosaiks_code=="") {
    mosaiks_code = here("code")
}
source(file.path(mosaiks_code,"mosaiks","config.R"))

basename = "DHS_wealth"

for (country_name in countries_to_run)
{
    #Read in the records that we want to match the NL data onto
    csv_file = file.path(data_dir, "raw/head_rep/All_DHS", paste0(country_name , "_" , basename , ".csv"))
    records = read.csv(csv_file)
    records = records[records$LATNUM !=0,]


    latmin = min(records$LATNUM,na.rm = T)
    latmax = max(records$LATNUM,na.rm = T)
    lonmin = min(records$LONGNUM,na.rm=T)
    lonmax = max(records$LONGNUM,na.rm=T)

    #Read in the NL raster
    #Check here for which panel to use:
    #https://www.ngdc.noaa.gov/eog/viirs/download_ut_mos.html
    nl_data_dir <- file.path(data_dir, "raw", "applications", "nightlights")
    if(country_name == "haiti") {
      r = raster(file.path(nl_data_dir,"SVDNB_npp_20150101-20151231_75N180W_vcm-ntl_v10_c201701311200.avg_rade9.tif"))
    } else if(country_name == "rwanda") {
      r = raster(file.path(nl_data_dir,"SVDNB_npp_20150101-20151231_00N060W_vcm-ntl_v10_c201701311200.avg_rade9.tif"))
    } else if(country_name == "nepal") {
      r = raster(file.path(nl_data_dir,"SVDNB_npp_20150101-20151231_75N060E_vcm-ntl_v10_c201701311200.avg_rade9.tif"))
    }

    #Extract the values of the NL over the raster.
    #We want to take a 5km radius square around the cluster center, so use that to define a box and then pass that box as a polygon to 
    #the extract function. Then we can have the extract function calculate bins of the raw values and aggregate them up. 
    first = T
    for (i in 1:nrow(records)) {
      print(i)
      #Generate a polygon from the 
      lat = records$LATNUM[i]
      lon = records$LONGNUM[i]
      #Look 5km in each direction: 
      rad = 0.05
      p = as(extent(c((lon-rad),(lon + rad), (lat - rad), (lat + rad))), 'SpatialPolygons')
      values = extract(x = r, y = p)

      feat = c()
      feat = c(feat,mean(values[[1]],na.rm = T),min(values[[1]],na.rm=T),max(values[[1]],na.rm=T))
      #Get bin values
      breaks = lseq(.1,500,20)
      breaks = c(0,breaks,99999999)
      for(j in 1:(length(breaks)-1)) {
        feat = c(feat, mean( (values[[1]]>breaks[j]) & (values[[1]]<breaks[j+1]), na.rm=T))
      }

      if(first) {
        NLfeat = feat
        first = F
      } else {
        NLfeat = rbind(NLfeat, feat)
      }
    }

    ### combine the features back onto the records: 
    out = cbind(records, NLfeat)
    colnames(out) = c(colnames(out)[1:9],"NLmean","NLmin","NLmax",paste0("NLbin",colnames(out)[13:33]))

    fn = file.path(data_dir,"raw/head_rep/All_DHS", paste0(country_name , "_" , basename , "_withNL.csv"))
    print(fn)
    write.csv(file = fn, x = out)
}
