
############################################################

# This script takes in a netCDF file of results for the replication
# of Head et al. (2017), shown in Figure S17, and exports a table
# of R^2 performance metrics.

############################################################

library(here)

rm(list=ls())

## Import config.R to set filepaths
mosaiks_code = here("code")

source(file.path(mosaiks_code,"mosaiks","config.R"))

out_dir <- file.path(res_dir, "tables", "TableS6")
dir.create(out_dir, showWarnings=FALSE, recursive=TRUE)

## Source the necessary helper files
source(file.path(utils_dir, "R_utils.R"))

library(ncdf4)
library(reshape2)
library(dplyr)
library(knitr)
library(kableExtra)

nc_data <- nc_open(file.path(data_dir, "output", "head_rep", "head_rep_performance.nc"))
print(nc_data)

# load R2s for each method

# dimensions: fold, outcome, intercept, country [NOTE: use only intercept 2nd dimension which includes the intercept]
r2_hyb <- melt(ncvar_get(nc_data, "best_r2_hyb")[,,2,])
colnames(r2_hyb) = c("fold", "outcome", "country", "r2" )

# dimensions: outcome, fold, intercept, country [NOTE: use only intercept 2nd dimension which includes the intercept]
r2_rcf <- melt(ncvar_get(nc_data, "best_r2_rcf")[,,2,])
r2_nl <- melt(ncvar_get(nc_data, "best_r2_nl")[,,2,])
colnames(r2_rcf) = c("outcome","fold","country","r2")
colnames(r2_nl) = colnames(r2_rcf)

# dimensions: outcome, fold, country
r2_head <- melt(ncvar_get(nc_data, "r2_head"))
colnames(r2_head) = c("outcome", "fold","country","r2")

# combine
dfs <- list(r2_rcf, r2_nl,r2_hyb,r2_head)

# compute mean R2 across folds
dfsmn = list()
namevec = c("rcf", "nl", "hyb", "head")
for (i in 1:4) {
  nm = paste0("r2", namevec[i])
  dfsmn[[i]] = dfs[[i]] %>% group_by(outcome, country) %>% summarize(mnr2 = mean(r2, na.rm = TRUE))
  colnames(dfsmn[[i]]) = c("outcome","country",nm)
}

# merge
df =  left_join(dfsmn[[1]], dfsmn[[2]], by=c("outcome","country")) %>%
      left_join(., dfsmn[[3]], by=c("outcome","country")) %>%
      left_join(., dfsmn[[4]], by=c("outcome","country"))

# intuitive labels
df$country = factor(as.numeric(df$country),
                              levels = c(
                                '1', 
                                '2', 
                                '3'), 
                              labels = c(
                                'Rwanda', 
                                'Haiti', 
                                'Nepal')) 

df$outcome = factor(as.numeric(df$outcome),
                  levels = c(
                    '1', 
                    '2', 
                    '3', '4', '5', '6', '7', '8','9','10','11'), 
                  labels = c(
                    "Wealth",
                    "Electricity",
                    "Mobile Phone Ownership",
                    "Education",
                    "Bed net count",
                    "Female BMI",
                    "Water access",
                    "Child height %ile",
                    "Child weight %ile",
                    "Hemoglobin level",
                    "Child weight / height %ile"))


# sort and export
df = df %>% arrange(., country, outcome)

# output
write.csv(df, file.path(res_dir, "tables", "TableS6", "head_etal_comparison_table.csv"))

# The below snippet outputs a well-formatted PDF but for some reason hangs when running
# in "headless" mode on CodeOcean. It should run find when executed in an interactive
# environment
# df %>% 
#   select(country, outcome, `r2rcf`:`r2head`) %>% 
#   kable(digits = 2, align = "llcccc", col.names = c("Country", "Task", "RCF", "Nightlights", "RCF+NL", "Head et al.")) %>% 
#   collapse_rows(columns = 1, valign = "top") %>%
#   add_header_above(header = c(" " = 2, "Method" = 4)) %>%
#   kable_styling(font_size = 12, full_width=F) %>%
#   save_kable(file.path(out_dir, "head_etal_comparison_table.pdf"))
