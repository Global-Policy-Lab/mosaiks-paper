
#########################################
# This script holds helper functions that are called throughout the R scripts used in the MOSAIKS repo. 
#########################################

library(rgdal)

#image with in pixels
imageWidthPixels = 640

#Meters per degree:
#110.567 #km/degree at equator
#111.699 #km/degree at pole (diff because it's an ellipsoid not sphere)
m_degree = 111 * 1000

### Useful functions: 
#Latitude in degrees, zoom level from API
meters_per_pixel = function(lat, zoom) {
  156543.03392 * cos(lat * pi / 180) / (2^zoom)
}

### This function intakes a vector of lats, lons, radii, and a crs
### It returns spatialpolygons object of the same length. 
pointsToSquares = function(lat, lon, dX, dY, crs) {
  yPlus <- lat+dY
  xPlus <- lon+dX
  yMinus <- lat-dY
  xMinus <- lon-dX
  
  # calculate polygon coordinates for each plot centroid. 
  square=cbind(xMinus,yPlus,  # NW corner
               xPlus, yPlus,  # NE corner
               xPlus,yMinus,  # SE corner
               xMinus,yMinus, # SW corner
               xMinus,yPlus)  # NW corner again - close ploygon
  
  ID = 1:nrow(square)
  
  
  polys <- SpatialPolygons(mapply(function(poly, id) 
  {
    xy <- matrix(poly, ncol=2, byrow=TRUE)
    Polygons(list(Polygon(xy)), ID=id)
  }, 
  split(square, row(square)), ID),
  proj4string=crs)
  
  return(polys)
}

#This is a wrapper function for the more general function pointsToSquares
#It intakes lat and lon centroids and returns the according tile shapefile. 
centroidsToTiles = function(lat,lon,zoom,numPix) {
  
  #Tiles are square in meter space, which means that they're equal width (in degrees) but not equal height (in degrees) everywhere. 
  
  #Get tile width in m as a function of zoom and latitude
  tileWidthMeters = meters_per_pixel(lat = lat,zoom) * imageWidthPixels
  
  #Get dX and dY (in degrees)
  #in lat the meters per degree is always the same, so we can get dY in degrees by just dividing by m/degree
  dY = tileWidthMeters / m_degree / 2
  #in lon the meters per degree is a function of latitude, so we have to account for that when converting meters to degree.
  #note that dX in degrees, however, is the same at all latitudes. AKA "tileWidthMeters / m_degree / cos(lat*pi/180) / 2" is a constant: 0.006886219, given zoom and numPix. 
  dX = tileWidthMeters / m_degree / cos(lat*pi/180) / 2
  
  #Call pointsToSquares
  return(pointsToSquares(lat = lat, lon = lon, dX = dX, dY = dY, crs = CRS(as.character("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0"))
  ))
  
}

#This is a function that replicates a lot of "centroidsToTiles" and "pointsToSquares" but 
#instead of returning the shapefile it just returns the verticies of the tile in degrees
centroidsToSquareVerticies = function(lat,lon,zoom,numPix) {
  
  ###Get tile width in m as a function of zoom; this is the same as tile height.
  tileWidthMeters = meters_per_pixel(lat = lat, zoom) * imageWidthPixels
  
  ###Get dX and dY
  #in lat it's always the same 
  dY = rep(tileWidthMeters / m_degree, length(lat)) / 2
  #in lon it's a function of latitude
  dX = tileWidthMeters / m_degree / cos(lat*pi/180) / 2
  
  ### calculate rectangle coordinates for each plot centroid. 
  #x and y values of edges
  yPlus <- lat+dY
  xPlus <- lon+dX
  yMinus <- lat-dY
  xMinus <- lon-dX
  #x and y values of coordinates
  rectangle=cbind(xMinus,yPlus,  # NW corner
               xPlus, yPlus,  # NE corner
               xPlus,yMinus,  # SE corner
               xMinus,yMinus, # SW corner
               xMinus,yPlus)  # NW corner again - close ploygon
  
  print("NW corner, NE corner, SE corner, SW corner, NW corner again to close")
  return(rectangle)
  
}

namean = function(x) {
  return(mean(x, na.rm=T))
}

##### makegrid.R function #####
# This script defines the grid for the remote sensing project using Google imagery
# The grid is defined by two vectors which represent the latitudes and longitudes of the grid centers. 
# In angle space, the centers are equally spaced in longitude and unequally spaced in latitude (closer together as you get to the poles)
# In meter-space, the centers are equally spaced at a given latitude but closer together as you go higher in latitude; 
# the centers are unequally spaced vertically and are closer to gether vertically as you approach the poles. 
# Returns a list of lonvals, latvals
# Note: uses some constants defined at the top of this script, which will be loaded when the function is loaded.
makegrid <- function(zoom=16, pixels = 640, lonmin = -125, lonmax = -66, latmin = 25, latmax = 50) {
  
  #image with in pixels
  imageWidthPixels = pixels
  
  ###Start in the upper left corner and assign the entire space a value in lat lon space. 
  imageWidthMeters_atEquator = meters_per_pixel(lat = 0,zoom) * imageWidthPixels
  imageWidthDegrees = imageWidthMeters_atEquator * 1/(m_degree) #the same as image width degrees.
  
  #Get lonVals first, easy because they are equally spaced.
  lonVals = seq(from = lonmin, to = lonmax, by = imageWidthDegrees)
  
  #Get latVals second, harder because the images get shorter as you get higher latitude. 
  latVals = c()
  currentLat = latmax
  i = 1
  while(currentLat > latmin) {
    latVals[i] = currentLat
    i = i + 1
    currentLat = currentLat - imageWidthDegrees * cos(currentLat * pi / 180)
  }
  
  return(list(lonVals, latVals))
}

###### Crop a matrix of lon-lats to the continental US ###### 
subsetToLand = function(lonlatmtx, lonmin = -125, lonmax = -66, latmin = 25, latmax = 50, shapefilepath = "../data/raw/shapefiles/gadm36_USA_shp/gadm36_USA_0.shp") {
  print("original length:")
  print(nrow(lonlatmtx))
  
  ### load shapefile
  landshp <- readOGR(shapefilepath)
  land <- crop(landshp, extent(lonmin, lonmax, latmin, latmax)) #This should take about 15 seconds
  
  ## Turn the subgrid into spatial points
  subgridN2sp <- SpatialPoints(lonlatmtx[,1:2])
  proj4string(subgridN2sp) <- proj4string(land)
  
  starttime <- Sys.time()
  matched <- over(subgridN2sp, as(land, "SpatialPolygons"))
  endtime <- Sys.time()
  endtime - starttime 
  
  subgridmatch <- cbind(lonlatmtx, matched)
  subgrid <- subset(subgridmatch, !is.na(matched))
  print("final length")
  print(nrow(subgrid))
  return(subgrid)
}

#############################
  # Function: Load all outcomes and append, including any task-specific cleaning or transformation
#############################

# function for loading
load_Y = function(outcome, ypath) {
  ### Load labels
  yy <- read.csv(ypath)
  yy$lat = NULL
  yy$lon = NULL
  
  ### Do any outcome-specific cleaning:
  if(outcome == "log_ppsqft") {
    # Log price per sq ft
    yy$log_ppsqft = log(yy$price_per_sqft)
    # Drop any bad data 
    yy <- yy[is.na(yy$log_ppsqft)==FALSE,]  
    
  } else if(outcome == "income") {
    # Drop missing data, which is flagged as -999
    yy = yy[yy$income != -999,]
    
  } else if (outcome == "log_nightlights") {
    yy$log_nightlights = log(yy$y+1)
    yy$nightlights = yy$y
  } else if (outcome == "log_population") {
    yy = yy[yy$population != -999,]
    yy$log_population = log(yy$population+1)
  } else if (outcome == "roads") {
    yy$roads = yy$length
  }
  
  #remove unnecessary merge variables
  drop <- c("X","V1", "V1.1")
  yy = yy[,!(names(yy) %in% drop)]
  
  return(yy)
}

