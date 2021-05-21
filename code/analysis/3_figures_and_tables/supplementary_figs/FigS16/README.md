To replicate this analysis fully, you will need to do the following:

1. Acquire imagery and extract RCF features. Acquire imagery following [Head et al. 2017](https://people.eecs.berkeley.edu/~andrewhead/pdf/satellites.pdf) and place into `data/raw/imagery/head_rep/[countryname]`. When collected in this, run [1_create_rcf_features.py](../../1_feature_extraction/1_create_rcf_feature_matrices.py).

2. Acquire and process DHS labels. This can be done in two steps. First, users must get written consent from the DHS program for us to share this data. Details on how to access DHS data can be found here: https://dhsprogram.com/data/Access-Instructions.cfm. Second, once consent has been acquired, contact us with proof and we will share the data. Alternatively, you can download the data directly from DHS and process it as described in [Head et al. 2017](https://people.eecs.berkeley.edu/~andrewhead/pdf/satellites.pdf). Once acquired, place the data in the data directory under: raw/head_rep/All_DHS/. Each processed CSV file in this directory should correspond to one country-outcome pair and each row should correspond to a cluster. The files should have the following columns: id, xcoord, ycoord, cluster, `name_of_outcome`. `id` and `cluster` should be equivalent.

3. Create nightime lights features:
```bash
Rscript make_nl_features.R
```

4. Run Ridge regressions using nighttime lights, transfer-learning features (from Head et al.), and RCF features
```bash
papermill head_replication.ipynb path/to/output/notebook.ipynb
```
