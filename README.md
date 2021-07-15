# MOSAIKS

This repository provides the code required to produce the figures appearing in the main text and Supplementary Materials of:

E. Rolf, J. Proctor, T. Carleton, I. Bolliger, V. Shankar, M. Ishihara, B. Recht, and S. Hsiang. ["A Generalizable and Accessible Approach to Machine Learning with Global Satellite Imagery,"](https://doi.org/10.1038/s41467-021-24638-z) *Nature Communications*, 2021.

This repository also serves provides examples of the generalizability of the MOSAIKS approach, which enables users with access to limited computing resources and without expertise in machine learning or remote sensing to generate observations of new variables in new locations. While this repository reproduces the results from the paper titled above, it also describes how users could generate observations of new variables in new research and policy contexts, with only minimal changes to the existing code base.

If you are viewing this repository on Github, please check out our [Code Ocean capsule](https://doi.org/10.24433/CO.8021636.v2), where you will find a mirror of this repository along with data and a computing environment set up to run all of the analyses in our paper. You may interact with the code via this platform or simply download the data for use on your own platform.

Additional material related to this repository and the corresponding paper are available at [http://www.globalpolicy.science/mosaiks](http://www.globalpolicy.science/mosaiks).

## 1. Structure of the repository

### 1.1. Code

The code folder is organized into an analysis pipeline and a package containing tools necessary to enable that pipeline.

- [analysis/](code/analysis): This folder contains scripts and notebooks (both R and Python) used to train and test the MOSAIKS prediction model for all tasks shown in the paper, as well as to generate figures and tables from the main text and Supplementary Materials. This folder also contains an example Python notebook that shows how to use the MOSAIKS approach to predict population density using MOSAIKS. The notebook also demonstrates the generalizability of MOSAIKS by describing the few changes needed to predict the other tasks studied in the paper (or a completely new task).

- [mosaiks/](code/mosaiks): This package contains all custom tools and functions that are used by the scripts/notebooks in the analysis folder. Importantly, it includes configuration files, [config.py](code/mosaiks/config.py) and [config.R](code/mosaiks/config.R), which contain settings that control parameters used in both analysis and plotting of results. All default settings in these files are set to match the results shown in the paper, but users may adjust them as desired to conduct new analyses or tests.

- [run.sh](code/run.sh): This script contains a (mostly) end-to-end replication of our analysis, including production of figures included in the paper, and is what gets executed when clicking "Reproducible Run" on Code Ocean. All raw data downloading/preprocessing steps, as well as some other long-running steps (e.g. training a CNN for comparison to MOSAIKS) are excluded from this pipeline. Users may choose to run this full replication script for a generic replication or to extract pieces that they would like to modify and/or execute interactively. All intermediate data should be available such that users may begin executing their code from any stage of the pipeline.

### 1.2. Data

Data is hosted within our [Code Ocean capsule](https://doi.org/10.24433/CO.8021636.v2). Due to a variety of license and data use agreement restrictions across data sources, as well as storage size constraints, we cannot host the raw label data nor the imagery used in our analysis. Instead, we provide detailed instructions on how to obtain and preprocess this data in the [Installation](#-installation) section. Preprocessing scripts are available in [code/analysis/1_feature_extraction](code/analysis/1_feature_extraction) (for image feature extraction) and [code/analysis/2_label_creation](code/analysis/2_label_creation) (for label creation). For data sources that can be downloaded programmatically, scripts in the latter folder also contain code that initiates a download of the corresponding labels. Note that over time, URLs and access instructions may change. If you are not able to obtain any of this data, please file an issue in our GitHub repository and we will do our best to update the instructions accordingly.

The data folder is organized as follows:

- `raw/`: The destination for downloads of all raw data. Locations for specific data files (e.g. raw nighttime lights values) within this directory will be defined in the corresponding scripts. This directory is not provided as part of the hosted data (see [Obtaining Raw Data](#obtaining-raw-data)).

- `int/`: Contains all intermediate data necessary to reproduce the results in our article, including extracted features and aggregated/transformed label data.

- `output/`: Contains outputs of regressions and various end-stage products.

#### Obtaining Raw Data

All data used in this analysis is from free, publicly available sources and is available for download other than the house price data (see below). While all other data is freely available, the source data is under a variety of licenses that limit re-distribution of data in various ways. To accommodate these agreements, we do not host any of the raw data that falls under these restrictions. Instructions are provided below for obtaining each of the input datasets.

##### Imagery

The `.npz` grid files in `data/int/grids` (produced by [Step 0](code/analysis/0_grid_creation) of our analysis pipeline) provide the lat/lon centroids of image tiles used in our analysis. You will need to acquire ~1x1km 256x256 pixel RGB images centered at these locations. The function [centroidstoSquareVertices](code/mosaiks/utils/R_utils.R#L75) can help by mapping these centroids to exact grid cell boundaries (use `zoom=16, numPix=640` arguments to achieve the appropriately sized boundaries).

Downloading imagery may be the most difficult piece of reproducing out analysis. Thus, in order to facilitate reproducibility of our article's results, we provide all of the pre-featurized images necessary to conduct our analysis within `data/int/feature_matrices`.

##### Labels

- **American Community Survey Data (Including Income)**: All data from the ACS, including our income label, can be programmatically downloaded from the US Census Bureau's website. Users will first need to obtain an API key [here](https://api.census.gov/data/key_signup.html). Then, you should only need to run the two ACS download scripts [here](code/analysis/2_label_creation/income/0_ACS_download.R) and [here](code/analysis/2_label_creation/ACS/0_ACS_download.R) to download income data and all other ACS variables, respectively.
- **Forest Cover**: The Global Land Analysis and Discovery Group at the University of Maryland currently hosts this data, which is free to use and disseminate. However, the raw data used for this analysis exceeds 600GB, so we do not host it directly. The scripts in [2_label_creation/treecover](code/analysis/2_label_creation/treecover) will provide a rough template to download this data, though modification will be needed as the data structure has changed slightly from when it was originally downloaded from an earlier USGS source.
- **Elevation**: Elevation data comes from from Mapzen elevation tiles hosted on AWS. The `elevatr` R package downloads this data directly and thus no source data is stored in `data/raw`.
- **Population**: Population data comes from from the GPW dataset, which is licensed under a CC BY 4.0 license. We host this dataset but users may re-download it from [https://sedac.ciesin.columbia.edu/downloads/data/gpw-v4/gpw-v4-population-density-rev10/gpw-v4-population-density-rev10_2015_30_sec_tif.zip](https://sedac.ciesin.columbia.edu/downloads/data/gpw-v4/gpw-v4-population-density-rev10/gpw-v4-population-density-rev10_2015_30_sec_tif.zip)
- **Nighttime Lights**: The Earth Observation Group at the Colorado School of Mines hosts this data. They are freely available, but users are required to register [here](https://eogdata.mines.edu/eog/EOG_sensitive_contents). Once registered, download all of the files associated with the v1 VIIRS annual composites for 2015 (link [here](https://eogdata.mines.edu/nighttime_light/annual/v10/2015/)) and save them to `data/raw/applications/nightlights/`.
- **Road Length**: Road length data is extracted from the USGS National Transportation Dataset, which is provided freely in the public domain. A selection of several states from the raw road length data is available in the Code Ocean capsule for use in replicating some imagery; however, the full collection can be downloaded using the [extractRoadLength.R](code/analysis/2_label_creation/roads/extractRoadLength.R) script.
- **Housing Price**: House price data is provided by Zillow through the Zlllow Transaction and Assessment Dataset (ZTRAX). More information on accessing the data can be found at <http://www.zillow.com/ztrax>. Because this dataset does not exist in the public domain, we cannot share any of the raw data files. Furthermore, due to privacy concerns, we can only release our aggregated, gridded housing price dataset for grid cells containing >=30 recent property sales. Because of this, results in replication code will demonstrate better performance than those quoted in the paper, which utilize the full dataset (see Supplementary Materials Section S.1.1). The results and opinions regarding housing price described in this repository and the associated publication are those of the authors and do not reflect the position of Zlllow Group.
- **Replication of [Head et al. 2017](https://people.eecs.berkeley.edu/~andrewhead/pdf/satellites.pdf)**: Several data sources are required for this portion of our paper, which produces Supplementary Figure 16.
  - *DHS data*: Acquiring these data requires permission from the DHS. See the [task-specific readme](code/analysis/3_figures_and_tables/supplementary_figs/FigS16/README.md) for more details on how to acquire these data, which we are not able to be disseminate.
  - *DMSP Nighttime Lights data*: We do not use the actual values from this earlier generation "nightlights" product but the GeoTiff file provided in this directory is necessary to execute code from the original *Head et al.* analysis. The data is free to distribute and can be re-downloaded from [https://ngdc.noaa.gov/eog/data/web_data/v4composites/F182010.v4.tar](https://ngdc.noaa.gov/eog/data/web_data/v4composites/F182010.v4.tar). Image and data processing by NOAA's National Geophysical Data Center. DMSP data collected by US Air Force Weather Agency. Additional files with the *_TL* suffix are aggregated versions of the nighttime lights data, processed by the authors of *Head et al. 2017* and redisseminated with their permission.

### 1.3 Results

Results, such as paper figures and tables, are produced within a `results/` folder at the top level of this repository. On Code Ocean, these are archived after each "Reproducible Run".

## 2. Installation

The easiest way to replicate and/or interact with our analysis is to use this Code Ocean capsule, which provides a cloud platform containing our code, data, and computing environment together. An alternative approach is to separately obtain these three items for use on an alternative computing platform. Both approaches are described below. In either case, if you wish to begin the analysis/replication from raw data (rather than preprocessed data), you must obtain additional data directly from various data providers (see [Obtaining Raw Data](#obtaining-raw-data)).

### 2.1. Using Code Ocean

To work with the interactive notebooks we have written to (a) walk through a typical user's experience with MOSAIKS, or (b) reproduce the analyses in the associated paper, you will likely want to use the `Launch Cloud Workstation-->JupyterLab` functionality of Code Ocean. After doing so, you will want to run the following code snippet to install the package needed to execute our analyses:

```bash
pip install -e code
```

**Note 1**: The Code Ocean capsule contains ~50 GB of data, which takes ~10 minutes to load when launching a cloud workstation or running the non-interactive `Reproducible Run` script. The `Reproducible Run` itself takes ~10 hours.

**Note 2**: In both the `Reproducible Run` and `Interactive Cloud Workstation` environments, our code is configured to use the GPU provided on Code Ocean's workstations. For some figures and tables, slightly better performance may be observed upon replication with a GPU than that which is presented in our manuscript. This is because the package we use to solve the ridge regression on a CPU, where the majority of our analysis was run, raises a warning when the $X^TX + \lambda I$ matrix used in a ridge regression is ill-conditioned. We throw these runs out when performing hyperparameter selection. When run on a GPU, a different solver is used (from `cupy` rather than `scipy`) which does not raise these warnings and thus those hyperparameters are not ignored. In rare cases, a hyperparameter that raises an "ill-conditioned" warning may give better out-of-sample performance and will be selected when replicating our analysis with a GPU.

### 2.2. Using an Alternative Computing Platform

If you choose option two, you will separately need to obtain three things: code, data, and an appropriate computing environment:

#### 2.2.1. Code

You should clone this repository, which is mirrored on [Github](https://github.com/Global-Policy-Lab/mosaiks) and Code Ocean. Either source is appropriate to clone as they contain the same code

#### 2.2.2. Data

When viewing our Code Ocean capsule, hover over `data` and click the caret that appears. You will see an option to download this folder. Place this downloaded `data` folder in the root directory of this repository (i.e. at the same level as the `code/` folder). Alternatively, you may place a symlink at that location that points to this data folder.

#### 2.3. Computing Environment

You will need to install and activate our `mosaiks-env` [conda](https://docs.conda.io/en/latest/miniconda.html) environment. Once you have conda installed on your machine, from the root directory of this repository, run:

```bash
conda env create -f environment/environment.yml
conda activate mosaiks-env
```

Note that depending on the operating system and GPU capability of your machine, you may need to change `pytorch`, `torchvision`, and/or `cudatoolkit` packages in [environment.yml](environment.yml). Please see <https://pytorch.org/get-started/locally/> for instructions. Additionally, to reduce the size of the environment you may wish to comment out some of the "infrastructure" packages (e.g. `jupyterlab`) *if* you already have these installed and are able to access the R and python kernels that get installed in the `mosaiks-env` environment. If you're not sure, do not comment these out.

Finally, you will also need to install our `mosaiks` package. From the root directory of the repo, call

```bash
pip install -e code
```

### 2.4. Alternative locations for code, data, and results folders

Using the above instructions, our code should correctly identify the locations of the `code`, `data`, and `results` folders as residing within the root directory of this repository. If, for whatever reason, you need to direct our scripts to look in different locations, you can set the following environment variables

- `MOSAIKS_HOME`: Sets a new root directory
- `MOSAIKS_CODE`: Sets a new directory to use as the top-level code directory (the folder named `code` within this repo)
- `MOSAIKS_DATA`: Sets a new data directory.
- `MOSIAKS_RESULTS`: Sets a new results directory

`MOSAIKS_HOME` is overridden by the other variables, such that if you had `MOSAIKS_HOME=path1/path2` and `MOSAIKS_DATA=path3`, then our code would search for a code folder at `path1/path2/code`, a results folder at `path1/path2/results`, and a data folder at `path3`.

## 3. Details on the contents of each subfolder within code

### 3.1. analysis/

We recommend that new users begin with the notebook MOSAIKS_example.ipynb, which can be found in the `code/analysis/` folder. If users want to replicate the entire MOSAIKS pipeline, the scripts and subfolders contained in the analysis folder are named according to their stage in the pipeline. Thus, if users wish to begin at the beginning of the pipeline, they should start with 0_grid_creation, proceed to `1_feature_extraction/`, then `2_label_creation/`, and finally to `3_figures_and_tables/`. A key component of the MOSAIKS framework is that feature extraction only needs to be performed once per image, no matter how many tasks are being predicted. If a user wanted to predict a new task, for example, they would skip straight to step 2. See the accompanying manuscript for further details.

- [MOSAIKS_example.ipynb](code/analysis/MOSAIKS_example.ipynb): While this notebook is not designed to directly replicate any exact figure or table in the paper, it is an example implementation of the MOSAIKS pipeline that can be adapted to any new prediction problem a user may face. This notebook shows how a user can load and merge labeled data with satellite image-based MOSAIKS features, train a model using ridge regression, and generate and map predictions from the trained model. It conducts these three steps for the case of predicting population density in the United States. This notebook can easily be run on a personal computer. It relies on intermediate data in `data/int/`, and it calls on settings in [config.py](code/mosaiks/config.py) (any of which can be adapted as desired for the other tasks in this paper or for a new prediction problem faced by a future user).

- [0_grid_creation/](code/analysis/0_grid_creation): This directory contains scripts to construct the standardized grids that are used to connect satellite image-based features to labeled data, and to create our uniform-at-random and population-weighted samples from these grids. A detailed description of this grid can be found in Supplementary Materials Section S.2.1.

- [1_feature_extraction/](code/analysis/1_feature_extraction): This folder contains scripts to transform raw satellite images into MOSAIKS random convolutional features, as well as features generated from a pretrained CNN (ResNet-152). While this is a relatively computationally expensive step in the MOSAIKS pipeline, we note that it is not necessary for users to conduct this step in order to train and test new tasks. This is because the same set of features ("X") can be used to predict all tasks ("Y"). Thus, the features can be computed only once and then shared to predict all known and future tasks.

- [2_label_creation/](code/analysis/2_label_creation): This folder contains code necessary to project raw labeled data for each of the seven tasks in the paper onto the standardized grid generated in `0_grid_creation`. The steps involved in conducting the spatial aggregation from raw data to grid cell vary depending on the structure of the raw data (see Supplementary Materials Section S.3.2). For some tasks, the raw data are large (> 100 G); users can choose to conduct replication of tables and figures in the paper using the standardized label data (i.e. aggregated to grid cells) stored in `data/int/` if they do not wish to interact with the raw labels (in this case, users should proceed to scripts and notebooks in [3_figures_and_tables/](code/analysis/3_figures_and_tables)). The scripts in [2_label_creation](code/analysis/2_label_creation) provide users with examples of how to aggregate any future labeled data used in a new task, as they include the aggregation of lat-lon point data (e.g. housing prices), polyline data (e.g. road length), polygon data (e.g. income), and raster data (e.g. nighttime lights). In each task, aggregation is constructed over two sets of random samples of size N=100,000 sampled from the full grid. The first random sample is sampled uniform-at-random (indicated in scripts and in [config.py](code/mosaiks/config.py) as “UAR”) and the second is sampled with population weights (indicated as “POP”). This folder is structured with task-specific subfolders `[taskname]/`, each containing the aggregation script used for that task.

- [3_figures_and_tables/](code/analysis/3_figures_and_tables): This folder contains all code necessary to produce the figures and tables in the paper. Each subfolder applies to an individual figure or table from the paper, and is self-contained, in that all training of the prediction model and plotting of results are contained within the subfolder. These scripts rely on data stored in `data/int/` and call on settings in [config.py](code/mosaiks/config.py) and [config.R](code/mosaiks/config.R). Users can choose to enter the pipeline at this stage, relying on the output from the cleaning and standardization of both labeled data (“Y”) and features (“X”) occurring in scripts in steps 0 through 2, as all output is provided for all intermediate steps in `data/int/`.

**Note**: Some of the analysis subdirectories containing multiple scripts and/or notebooks contain an additional README providing further instructions.

### mosaiks/

This package contains all functions called by the analysis scripts in analysis/ and subfolders therein.

- [diagnostics/](code/mosaiks/diagnostics): Functions required to conduct the experiments shown in Figure 3, as well as some supplementary figures.
- [plotting/](code/mosaiks/plotting): Functions required for generating figures shown in the main text and Supplementary Materials.
- [solve/](code/mosaiks/solve): Functions required for training prediction models using ridge regression and k-fold cross-validation.
- [utils/](code/mosaiks/utils): Basic utility tools and functions used throughout various steps in the analysis.
- [featurization.py](code/mosaiks/featurization.py): Functions required for featurizing satellite imagery.
- [transforms.py](code/mosaiks/transforms.py): Helper functions that apply some simple QA/QC on our label data prior to regression.
- [config.py](code/mosaiks/config.py) and [config.R](code/mosaiks/config.R): Configuration files. This module contains settings that apply to the entire repository (e.g. file paths), settings that apply to each stage of analysis (e.g. parameters determining the spatial scale of the experiments shown in Figure 3B and C), and settings that apply to each task (e.g. whether to log the outcome variable or not). One exists for Python scripts and one for R scripts.

## Use of code and data

Our code can be used, modified, and distributed freely for educational, research, and not-for-profit uses. For all other cases, please contact us. Further details are available in the [code license](code/LICENSE). All data products created through our work that are not covered under upstream licensing agreements are available via a CC BY 4.0 license (see the [data license](data/LICENSE) available within the Code Ocean capsule). All upstream data use restrictions take precedence over this license.
