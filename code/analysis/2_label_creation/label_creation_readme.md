The scripts in this folder calculate the grid cell labels from *ground-truth* measurements.

The raw data used to produce these grid cell labels is available as described in the main repository readme.

The grid cell labels for each application are saved as: `outcomes_sampled_[application]_[location]_16_640_[sampling type]_[N]_0.csv`
For example: `outcomes_sampled_treecover_CONTUS_16_640_UAR_100000_0.csv`

Scripts are numbered in the order they should be run (e.g. `extractTreecover_1` should be run before `extractTreecover_2`). To create all the files some scripts must be run twice, once for uniform-at-random sampling (UAR) and again for population-weighted (POP) sampling.

For variables included in the global or super-resolution analyses, respective scripts are labeled `_global` and `_superres`, although some extraction scripts produce both continental US and global samples in a single script (e.g. nighttime lights).

The scripts in the `ACS/` folder construct the variables used to demonstrate the rapid generalizability of MOSAIKS in Fig. 4B.