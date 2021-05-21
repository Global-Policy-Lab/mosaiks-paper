
############################################################

# This script makes Figure S2, a matrix of scatterplots of each task against the other.

############################################################

## Packages
library(rgdal)
library(rgeos)
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

figdir <- file.path(res_dir,"figures/FigS2")
dir.create(figdir, showWarnings=FALSE, recursive=TRUE)

# Which sampling do you want to use?
samp <- "POP" # choices: POP or UAR (POP used in text)

###############################################################
        #Load data
###############################################################

# File paths
outcomes = c("treecover", "elevation", "log_population", "log_nightlights", "income", "log_ppsqft", "roads")


ypaths <- c(file.path(data_dir,"int/applications/treecover",paste0("outcomes_sampled_treecover_CONTUS_16_640_", samp, "_100000_0.csv")),
            file.path(data_dir, "int/applications/elevation",paste0("outcomes_sampled_elevation_CONTUS_16_640_", samp, "_100000_0.csv")),
            file.path(data_dir, "int/applications/population",paste0("outcomes_sampled_population_CONTUS_16_640_", samp, "_100000_0.csv")),
            file.path(data_dir, "int/applications/nightlights",paste0("outcomes_sampled_nightlights_CONTUS_16_640_", samp, "_100000_0.csv")),
            file.path(data_dir, "int/applications/income",paste0("outcomes_sampled_income_CONTUS_16_640_", samp, "_100000_0.csv")),
            file.path(data_dir, "int/applications/housing",paste0("outcomes_sampled_housing_CONTUS_16_640_POP_100000_0.csv")),
                      file.path(data_dir, "int/applications/roads",paste0("outcomes_sampled_roads_CONTUS_16_640_", samp, "_100000_0.csv")))

# load data for each outcome
dfList = list()
for(i in 1:length(outcomes)) {
  print(outcomes[i])
  print(ypaths[i])
  
  dfList[[i]] = load_Y(outcome = outcomes[i],
                                 ypath = ypaths[i])
}

######################################################################
# Plotting function: Used only in this figure
######################################################################

plotMyCorr = function(data,colx,coly){
  
  # calculate R2
  r2 <- round(cor(data[,colx], data[,coly], use = "complete.obs")^2,2)
  xloc <- floor(0.6*max(data[,colx], na.rm=T))
  yloc <- floor(max(data[,coly], na.rm=T))
  
  # get variable names
  myxlab = colnames(data)[colx]
  myylab = colnames(data)[coly]
  
  gg = ggplot() + geom_point(data=data,aes(data[,colx], data[,coly]),color = "gray48", alpha=.2, na.rm=T, size = .1) + xlab(myxlab) + ylab(myylab) +
  theme(axis.text.x = element_blank(), #element_text(size = 8,hjust = .5, vjust = .5, face = "plain"),
          axis.text.y = element_blank(), #element_text(size = 8, angle = 0, hjust = 1, vjust = 0, face = "plain"),
          axis.title.x = element_blank(), #element_text(size = 6, angle = 0, hjust = .5, vjust = 0, face = "plain"),
          axis.title.y = element_blank(), #element_text(size = 6, angle = 90, hjust = .5, vjust = .5, face = "plain"),
          axis.ticks.x=element_blank(),
          axis.ticks.y=element_blank(),
          aspect.ratio=1)
  return(gg)
}

######################################################################
# Plot cross-correlations and histograms
######################################################################

merged = Reduce(function(x, y) merge(x, y, all=TRUE, by=c("ID")), list(dfList[[1]], dfList[[2]], dfList[[3]],
                                                                       dfList[[4]], dfList[[5]], dfList[[6]], dfList[[7]]))
mycols <- which(colnames(merged) %in% outcomes)

### This is a canned routine that shows all R2s alongside figures
lower.panel<-function(x, y){
  points(x,y, pch=19, col=rgb(red = .62, green = .57, blue = .57, alpha = 0.2), cex = .5)
  r <- round(cor(x, y, use = 'complete.obs')^2, digits=2)
  txt <- paste0("R2 = ", r)
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  text(0.5, 0.9, txt)
}

p <- pairs(merged[,mycols], upper.panel = NULL,
      lower.panel = lower.panel)

### This is a custom routine to improve aesthetics
iter <- length(outcomes)-1

# Empty list of figures to store all corr plots
figList <- list() 
r2s <- list()

count <- 0

# i indicates your x axis, j indicates your y axis
for( i in 1:iter) {
  s <- i+1
  for(j in s:length(outcomes)) {
    count <- count+1
    print(paste0(outcomes[i], " corr w ", outcomes[j]))
    
    colx = which(colnames(merged) == outcomes[i])   
    coly = which(colnames(merged) == outcomes[j])
    
    figList[[count]] = plotMyCorr(data=merged, colx,coly)
    r2s[[count]] = cor(merged[,colx], merged[,coly], use = 'complete.obs')^2
  }
}

combined <- plot_grid(figList[[1]],  NULL, NULL, NULL,NULL, NULL,   
                      figList[[2]],  figList[[7]],NULL, NULL,NULL, NULL,     
                      figList[[3]], figList[[8]], figList[[12]],   NULL,NULL, NULL, 
                      figList[[4]], figList[[9]],  figList[[13]], figList[[16]],   NULL, NULL, 
                      figList[[5]],figList[[10]], figList[[14]], figList[[17]], figList[[19]], NULL,
                      figList[[6]], figList[[11]], figList[[15]],figList[[18]], figList[[20]], figList[[21]],   nrow = 6)

#combined
fn = file.path(figdir, paste0("corr_matrix_", samp, ".jpg"))
ggsave (filename = fn, plot = combined, width = 12, height = 12)

