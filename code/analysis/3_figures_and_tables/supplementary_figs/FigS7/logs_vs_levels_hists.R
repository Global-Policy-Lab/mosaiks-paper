############################################################

# This script makes Figure S7, a set of simple histograms 
# of the log and level of each task outcome

############################################################

## Packages
library(ggplot2)
library(cowplot)
library(ggExtra)
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

figdir <- file.path(res_dir,"figures/FigS7")
dir.create(figdir, showWarnings=FALSE, recursive=TRUE)

###############################################################
#Load data
###############################################################

# File paths
outcomes = c("treecover", "elevation", "log_population", "log_nightlights", "income", "log_ppsqft", "roads")

# Note that we call each task with the grid sampling used in the main analysis (UAR vs. POP)
ypaths <- c(file.path(data_dir, "int/applications/treecover/outcomes_sampled_treecover_CONTUS_16_640_UAR_100000_0.csv"),
            file.path(data_dir, "int/applications/elevation/outcomes_sampled_elevation_CONTUS_16_640_UAR_100000_0.csv"),
            file.path(data_dir, "int/applications/population/outcomes_sampled_population_CONTUS_16_640_UAR_100000_0.csv"),
            file.path(data_dir, "int/applications/nightlights/outcomes_sampled_nightlights_CONTUS_16_640_POP_100000_0.csv"),
            file.path(data_dir, "int/applications/income/outcomes_sampled_income_CONTUS_16_640_POP_100000_0.csv"),
            file.path(data_dir, "int/applications/housing/outcomes_sampled_housing_CONTUS_16_640_POP_100000_0.csv"),
            file.path(data_dir, "int/applications/roads/outcomes_sampled_roads_CONTUS_16_640_POP_100000_0.csv"))

# load data for each outcome
dfList = list()
for(i in 1:length(outcomes)) {
  print(outcomes[i])
  print(ypaths[i])
  
  dfList[[i]] = load_Y(outcome = outcomes[i],
                       ypath = ypaths[i])
}

######################################################################
# Plotting function
######################################################################

LogLevHist = function(data, mycol){
  
  #my var name
  varname = colnames(data)[mycol]
  
  # generate logs
  if(varname=="price_per_sqft") {
    data$log <- log(data[,mycol])
  } else {
    data$log <- log(data[,mycol]+1)
  }
  
  # reorder for simplicity
  data <- data[c("ID", varname, "log")]
  colnames(data) <- c("ID", varname, paste0("log_",varname))
  
  # labels
  xlablev = colnames(data)[2]
  xlablog = colnames(data)[3]
  
  # two hists
  gglevel <- ggplot(data, aes(x=data[,2])) + 
              geom_histogram(aes(y=..density..),     
                   bins=50,
                   colour="black", fill="lightcyan2")  + xlab(xlablev)  +
              theme(text = element_text(size=8),
                    axis.text.y = element_blank(),  
                    axis.title.y = element_blank(),  
                    axis.text.x = element_text(size = 8),
                    axis.ticks.y=element_blank())
  gglog <- ggplot(data, aes(x=data[,3])) + 
    geom_histogram(aes(y=..density..),      
                   bins=50,
                   colour="black", fill="seagreen")  + xlab(xlablog) +
    theme(text = element_text(size=8),
          axis.text.y = element_blank(), 
          axis.title.y = element_blank(), 
          axis.text.x = element_text(size = 8),
          axis.ticks.y=element_blank())
  gg <- plot_grid(gglevel, gglog, nrow = 1)
  return(gg)
}

######################################################################
# Plot histograms
######################################################################

# Clean outcome names
outcomes_clean = c("treecover", "elevation", "population", "nightlights", "income", "price_per_sqft", "roads")

figList <- list() 

for(i in 1:length(outcomes_clean)){
  
  data = dfList[[i]]
  mycol <- which(colnames(data) == outcomes_clean[i]) 
  
  figList[[i]] <- LogLevHist(data, mycol = mycol)
  
}

combined <- plot_grid(figList[[1]], figList[[2]], figList[[3]], figList[[4]], figList[[5]], figList[[6]], 
                      figList[[7]], nrow = 7)

fn = file.path(figdir,"log_vs_levels_hists.pdf")
ggsave (filename = fn, plot = combined, width = 6, height = 12)


