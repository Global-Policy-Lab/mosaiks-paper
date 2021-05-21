
############################################################

# This script loads labels and predictions for the four tasks
# we predict at global scale (forest cover, elevation, nighttime lights
# and population density) in four densely sampled zoomed-in regions of
# the world shown in Figure 4. It then plots labels and predictions 
# within each task's own subregion of the world. These zoom plots 
# are shown in Figure 4 for each subregion.

############################################################

## Load necessary packages
library(raster)
library(ggplot2)
library(maps)
library(mapdata)
library(maptools)
library(cowplot)
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

## Figure output path
figdir <- file.path(res_dir,"figures", "Fig4", "zoom")
dir.create(figdir, showWarnings = FALSE, recursive = TRUE)

## task list -- subset if you want only certain tasks
tasks = c("treecover","elevation","population","nightlights")

## This resolution parameter determines the display resolution of predictions (larger value = higher resolution)
## You may want to vary the parameter by task/zoom
res = c(8,6.25,7,8)

######################################################################
# Load labels
######################################################################

# Load labeled data from all tasks
datList = list()

for (i in 1:length(tasks)) {
  fl = list.files(file.path(data_dir,"int/applications", tasks[i]))
  fl = fl[grepl(x = fl, pattern = "dense")]
  datList[[i]] = read.csv(file.path(data_dir, "int/applications", tasks[i], fl))
  
  # clean - task-specific
  if (tasks[i] == "nightlights") {
    datList[[i]]$nightlights = log(1+datList[[i]]$luminosity)
  } else if (tasks[i] == "population") {
    datList[[i]]$population = log(1+datList[[i]]$population)
  }
  
  # clean - only need lon, lat, and label
  mycols = c("lon", "lat",tasks[i])
  datList[[i]] <- datList[[i]][,mycols]

}

######################################################################
# Load predicted values
######################################################################

predList = list()

for (i in 1:length(tasks)) {
  fl = list.files(file.path(out_dir, "world"))
  fl = fl[grepl(x = fl, pattern = tasks[i])]
  predList[[i]] = read.csv(file.path(out_dir, "world", fl))
  
  # clean - only need lon, lat, and label
  mycols = c("lon", "lat",tasks[i])
  predList[[i]] <- predList[[i]][,mycols]
  
  # the population labeled data are missing in Lake Victoria, but we have predictions there. Remove these! 
  if (tasks[i] == "population") {
    missingrows = is.na(datList[[i]]$population)
    predList[[i]]$population[missingrows] = NA
  }
}

######################################################################
# Rasterize & Plot
######################################################################

for (i in 1:length(tasks)) {
  e <- extent(min(datList[[i]]$lon), max(datList[[i]]$lon), min(datList[[i]]$lat), max(datList[[i]]$lat))

  rast <- raster()
  extent(rast) = e

  # Rasterize: depends on resolution parameter chosen at top of script  
  ncols <- (e@xmax-e@xmin)*res[i]
  nrows <- (e@ymax-e@ymin)*res[i]
  ncol(rast) <- ncols
  nrow(rast) <- nrows
  mycol = which(colnames(datList[[i]])==tasks[i])
  mycolpred = which(colnames(predList[[i]])==tasks[i])
  
  # clipping color bounds for display only -- both predictions and truth
  if(tasks[i] == "treecover") {
    clipmax <- 100
    clipmin <- 0
  } else if (tasks[i] == "elevation") {
    clipmax <- 5000
    clipmin <- 0
    } else if (tasks[i] == "nightlights") {
    clipmax <- 1.5
    clipmin <- 0
  } else if (tasks[i] == "population") {
    clipmax <- 5.5
    clipmin <- 0
  } 

  datList[[i]][,mycol] <- ifelse(datList[[i]][,mycol]>clipmax & is.na(datList[[i]][,mycol])==0, clipmax,datList[[i]][,mycol])
  datList[[i]][,mycol] <- ifelse(datList[[i]][,mycol]<clipmin & is.na(datList[[i]][,mycol])==0, clipmin,datList[[i]][,mycol])
  predList[[i]][,mycolpred] <- ifelse(predList[[i]][,mycolpred]>clipmax & is.na(predList[[i]][,mycolpred])==0, clipmax,predList[[i]][,mycolpred])
  predList[[i]][,mycolpred] <- ifelse(predList[[i]][,mycolpred]<clipmin & is.na(predList[[i]][,mycolpred])==0, clipmin,predList[[i]][,mycolpred])

  # Rasterize
  rast_true <- rasterize(datList[[i]][,1:2], rast, datList[[i]][,mycol], fun=mean, na.rm = T)
  rast_pred <- rasterize(predList[[i]][,1:2], rast, predList[[i]][,mycolpred], fun=mean, na.rm = T)

  # Call country boundaries
  ext <- as.vector(e)
  boundaries <- map('worldHires',
                  xlim=ext[1:2], ylim = ext[3:4],
                  plot=FALSE)
  boundaries <- map2SpatialLines(boundaries,
                               proj4string=CRS(projection(rast_true)))
  boundariesdf = sp::SpatialLinesDataFrame(boundaries, data.frame(ID = c(1:length(boundaries))), match.ID = F)
  boundariesdf <- crop(boundariesdf, e)

  # Convert raster to spatial points for plotting 
  map.true <- rasterToPoints(rast_true)
  map.pred <- rasterToPoints(rast_pred)

  # Make the points a dataframe for ggplot
  dftrue <- data.frame(map.true)
  dfpred <- data.frame(map.pred)
  #Make appropriate column headings
  colnames(dftrue) <- c("Longitude", "Latitude", colnames(datList[[i]])[mycol])
  colnames(dfpred) <- colnames(dftrue)

  ## Map, with task-specific aesthetics

    if (tasks[i] == "treecover") {
      mypal = "Greens"
      
      # True
      g = ggplot(data=dftrue, aes(y=Latitude, x=Longitude)) +
        geom_raster(aes(fill=dftrue[,3])) +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_rect(fill = "grey80", colour = NA),title= element_text(size = 10,hjust = 0.5,vjust = 1,face= c("bold"))) +
        coord_equal() +
        geom_path(data = boundariesdf, mapping = aes(x=long, y=lat, group=group), color = "black") +
        scale_fill_distiller(palette = mypal, direction=1) +
        labs(title= "Sampled labels",
         x="Longitude", y= "Latitude",color = "% forest cover")
      
      filename = file.path(figdir, paste0("Fig4_zoom_dense_", tasks[i], ".pdf"))
      ggsave(g, file = filename, width = 8, height = 10)
      
      # Predictions
      g = ggplot(data=dfpred, aes(y=Latitude, x=Longitude)) +
        geom_raster(aes(fill=dfpred[,3])) +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
              panel.background = element_rect(fill = "grey80", colour = NA),title= element_text(size = 10,hjust = 0.5,vjust = 1,face= c("bold"))) +
        coord_equal() +
        geom_path(data = boundariesdf, mapping = aes(x=long, y=lat, group=group), color = "black") +
        scale_fill_distiller(palette = mypal, direction=1) +
        labs(title= "Sampled predictions",
             x="Longitude", y= "Latitude",color = "% forest cover")
      
      filename = file.path(figdir, paste0("Fig4_zoom_dense_", tasks[i], "_predictions.pdf"))
      ggsave(g, file = filename, width = 8, height = 10)
      
    }  else if (tasks[i] == "elevation") {
      
      lowcol <- "antiquewhite"
      highcol <- "chocolate4"
      linecol <- "black"
      
      # True
      g = ggplot(data=dftrue, aes(y=Latitude, x=Longitude)) +
        geom_raster(aes(fill=dftrue[,3])) +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
              panel.background = element_rect(fill = "grey80", colour = NA),title= element_text(size = 10,hjust = 0.5,vjust = 1,face= c("bold"))) +
        coord_equal() +
        geom_path(data = boundariesdf, mapping = aes(x=long, y=lat, group=group), color = linecol) +
        scale_fill_gradient2(low = lowcol, mid = lowcol, high = highcol, midpoint = 0.1, na.value = "grey92", guide = "colourbar") +
        labs(title= "Sampled labels",
             x="Longitude", y= "Latitude",color = "meters")
      
      filename = file.path(figdir, paste0("Fig4_zoom_dense_", tasks[i], ".pdf"))
      ggsave(g, file = filename, width = 8, height = 10)
      
      # Predictions
      g = ggplot(data=dfpred, aes(y=Latitude, x=Longitude)) +
        geom_raster(aes(fill=dfpred[,3])) +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
              panel.background = element_rect(fill = "grey80", colour = NA),title= element_text(size = 10,hjust = 0.5,vjust = 1,face= c("bold"))) +
        coord_equal() +
        geom_path(data = boundariesdf, mapping = aes(x=long, y=lat, group=group), color = linecol) +
        scale_fill_gradient2(low = lowcol, mid = lowcol, high = highcol, midpoint = 0.1, na.value = "grey92", guide = "colourbar") +
        labs(title= "Sampled predictions",
             x="Longitude", y= "Latitude",color = "meters")
      
      filename = file.path(figdir, paste0("Fig4_zoom_dense_", tasks[i], "_predictions.pdf"))
      ggsave(g, file = filename, width = 8, height = 10)
      
    } else if (tasks[i] == "population") {
      mypal = "Blues"
      
      # True
      g = ggplot(data=dftrue, aes(y=Latitude, x=Longitude)) +
        geom_raster(aes(fill=dftrue[,3])) +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
              panel.background = element_rect(fill = "grey80", colour = NA),title= element_text(size = 10,hjust = 0.5,vjust = 1,face= c("bold"))) +
        coord_equal() +
        geom_path(data = boundariesdf, mapping = aes(x=long, y=lat, group=group), color = "black") +
        scale_fill_distiller(palette = mypal, direction=1) +
        labs(title= "Sampled labels",
             x="Longitude", y= "Latitude",color = "log(1+pop)")
      
      filename = file.path(figdir, paste0("Fig4_zoom_dense_", tasks[i], ".pdf"))
      ggsave(g, file = filename, width = 8, height = 10)
      
      # Predictions
      g = ggplot(data=dfpred, aes(y=Latitude, x=Longitude)) +
        geom_raster(aes(fill=dfpred[,3])) +
        theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
              panel.background = element_rect(fill = "grey80", colour = NA),title= element_text(size = 10,hjust = 0.5,vjust = 1,face= c("bold"))) +
        coord_equal() +
        geom_path(data = boundariesdf, mapping = aes(x=long, y=lat, group=group), color = "black") +
        scale_fill_distiller(palette = mypal, direction=1) +
        labs(title= "Sampled predictions",
             x="Longitude", y= "Latitude",color = "log(1+pop)")
      
      filename = file.path(figdir, paste0("Fig4_zoom_dense_", tasks[i], "_predictions.pdf"))
      ggsave(g, file = filename, width = 8, height = 10)
  
  } else if (tasks[i]=="nightlights") {
  
    lowcol <- "floralwhite" #"cornsilk"
    highcol <- "darkgoldenrod1"
    linecol <- "black"
    
    # True
    g = ggplot(data=dftrue, aes(y=Latitude, x=Longitude)) +
      geom_raster(aes(fill=dftrue[,3])) +
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_rect(fill = "grey80", colour = NA),title= element_text(size = 10,hjust = 0.5,vjust = 1,face= c("bold"))) +
      coord_equal() +
      geom_path(data = boundariesdf, mapping = aes(x=long, y=lat, group=group), color = linecol) +
      scale_fill_gradient2(low = lowcol, mid = lowcol, high = highcol, midpoint = 0.1, na.value = "grey92", guide = "colourbar") +
      labs(title= "Sampled labels",
         x="Longitude", y= "Latitude",color = "log(1+radiance)")
    
    filename = paste0(figdir, "/Fig4_zoom_dense_", tasks[i], ".pdf")
    ggsave(g, file = filename, width = 8, height = 10)
    
    # Predictions
    g = ggplot(data=dfpred, aes(y=Latitude, x=Longitude)) +
      geom_raster(aes(fill=dfpred[,3])) +
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_rect(fill = "grey80", colour = NA),title= element_text(size = 10,hjust = 0.5,vjust = 1,face= c("bold"))) +
      coord_equal() +
      geom_path(data = boundariesdf, mapping = aes(x=long, y=lat, group=group), color = linecol) +
      scale_fill_gradient2(low = lowcol, mid = lowcol, high = highcol, midpoint = 0.1, na.value = "grey92", guide = "colourbar") +
      labs(title= "Sampled predictions",
           x="Longitude", y= "Latitude",color = "log(1+radiance)")
    
    filename = file.path(figdir, paste0("Fig4_zoom_dense_", tasks[i], "_predictions.pdf"))
    ggsave(g, file = filename, width = 8, height = 10)
    
    }
  rm(dftrue, dfpred, map.true, map.pred, rast_true, rast_pred, ext, e)
}
  