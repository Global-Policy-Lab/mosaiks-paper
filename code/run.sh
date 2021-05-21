#!/usr/bin/env bash
set -e


#######
# setup
#######

# set wd to root of repo
cd "$(dirname "$0")/.."

# make sure conda is initialized
CONDA_PREFIX_1=${CONDA_PREFIX_1:-/opt/conda}
source $CONDA_PREFIX_1/etc/profile.d/conda.sh

# take in passed conda environment name as argument
# if none passed, use current conda env
ENVNAME=${1:-$CONDA_DEFAULT_ENV}
conda activate $ENVNAME

# install kernel if needed, otherwise simply correct the kernel name
if [[ -z $ENVNAME || "$ENVNAME" = "base" ]]
    then
        ENVNAME="python3"
    else
        python -m ipykernel install --user --name $ENVNAME
fi

export MOSAIKS_OVERWRITE=True

# ensure trailing slash for consistency in filepaths
MOSAIKS_HOME=${MOSAIKS_HOME:-$(pwd)}
export MOSAIKS_HOME=${MOSAIKS_HOME%/}/
export MOSAIKS_RESULTS=${MOSAIKS_RESULTS:-${MOSAIKS_HOME}results}/
export MOSAIKS_DATA=${MOSAIKS_DATA:-${MOSAIKS_HOME}data}/
export MOSAIKS_CODE=${MOSAIKS_CODE:-${MOSAIKS_HOME}code}/

# define env vars
GRID_DIR=${MOSAIKS_CODE}analysis/0_grid_creation
FEAT_DIR=${MOSAIKS_CODE}analysis/1_feature_extraction
LAB_DIR=${MOSAIKS_CODE}analysis/2_label_creation
FIG_DIR=${MOSAIKS_CODE}analysis/3_figures_and_tables

# install our utilities
pip install -e $MOSAIKS_CODE

# define alias to run nb's from command line
run_nb() {
    NBPATH=$1
    OUTPATH=${NBPATH/"code/analysis"/"${MOSAIKS_RESULTS}/nb_logs"}
    mkdir -p $(dirname "$OUTPATH")
    shift
    papermill $NBPATH $OUTPATH \
        -k $ENVNAME \
        -p overwrite True \
        -p fixed_lambda True \
        --cwd $(dirname "$NBPATH") \
        "$@"
}


#########################################
# Grid, feature, and label pre-processing
#########################################

echo "The pre-processing steps of grid cell sampling, feature extraction, and ground \
truth label extraction are not included in this replication script due to \
computational time. However, they may be run individually via the associated \
scripts..."

# create grids
#time Rscript ${GRID_DIR}/create_grid_CONTUS_and_WORLD.R
#time Rscript ${GRID_DIR}/create_grid_dense_sample_regions.R

# extract features
#time python ${FEAT_DIR}/1_create_rcf_feature_matrices.py
#time python ${FEAT_DIR}/2_featurize_models_deep_pretrained

# extract labels: See various scripts in code/analysis/2_label_creation


########################################
# replicate all figures and tables
########################################

echo "Replicating the creation of all figures... (NOTE: Some time-intensive steps such \
as CNN training are not included in this replication but may be run individually via \
the associated scripts.)"

# Fig 2
echo "Running Fig 2: Regressions..."
time run_nb ${FIG_DIR}/Fig2_randomCV/1_run_regressions.ipynb
echo "Running Fig 2: Plots..."
time run_nb ${FIG_DIR}/Fig2_randomCV/2_make_fig.ipynb


# Fig 3
echo "Running Fig3: Training CNN (skipped for time)..."
# for LABEL in treecover elevation population nightlights income roads housing; do
#     time python ${FIG_DIR}/Fig3_diagnostics/train_CNN.py \
#         ${MOSAIKS_DATA}/output/cnn_comaprison/resnet18_${LABEL} --outcome $LABEL
# done
echo "Running Fig 3: Sensitivity to N and K..."
time run_nb ${FIG_DIR}/Fig3_diagnostics/model_sensitivity.ipynb
echo "Running Fig 3: Spatial cross-validation..."
time run_nb ${FIG_DIR}/Fig3_diagnostics/spatial_analyses_jitter_gammaLoop_interpolation.ipynb
echo "Running Fig 3: Transfer Learning comparison..."
time run_nb ${FIG_DIR}/Fig3_diagnostics/run_pretrained_resnet_regressions.ipynb
echo "Running Fig 3: Plotting..."
time run_nb ${FIG_DIR}/Fig3_diagnostics/fig_3.ipynb


# Fig 4
echo "Running Fig 4: Global regressions..."
time run_nb ${FIG_DIR}/Fig4_global_ACS_and_SR/world_model.ipynb

echo "Running Fig 4: Super-resolution (model training skipped due to need for raw imagery...)"
# time run_nb ${FIG_DIR}/Fig4_global_ACS_and_SR/superres/1_choose_sr_lambda.ipynb
# time python ${FIG_DIR}/Fig4_global_ACS_and_SR/superres/2_compute_superres.py 16000
time run_nb ${FIG_DIR}/Fig4_global_ACS_and_SR/superres/3_sr_figs.ipynb

echo "Running Fig 4: Zoomed-in dense predictions..."
time Rscript ${FIG_DIR}/Fig4_global_ACS_and_SR/plot_labels_preds_dense_zoom_regions.R

echo "Running Fig 4: ACS regressions..."
time run_nb ${FIG_DIR}/Fig4_global_ACS_and_SR/ACS/1_run_regressions.ipynb
echo "Running Fig 4: ACS plot..."
time run_nb ${FIG_DIR}/Fig4_global_ACS_and_SR/ACS/2_make_fig.ipynb


# Fig S1
echo "Running Fig S1: Panel A..."
time Rscript ${FIG_DIR}/supplementary_figs/FigS1/FigS1_A.R
echo "Running Fig S1: Panel B..."
time python ${FIG_DIR}/supplementary_figs/FigS1/FigS1_B_1.py
time Rscript ${FIG_DIR}/supplementary_figs/FigS1/FigS1_B_2.R


# Fig S2
echo "Running Fig S2..."
time Rscript ${FIG_DIR}/supplementary_figs/FigS2/correlation_across_tasks.R


# Fig S3 cannot be generated with publicly available data
# Fig S4 generated manually
# Fig S5 generated manually


# Fig S6
echo "Running Fig S6 (feature extraction skipped for time)..."
# See notebook for instructions on how to extract and save features using each patch size
time run_nb ${FIG_DIR}/supplementary_figs/FigS6/patch_size_regressions.ipynb


# Fig S7
echo "Running Fig S7..."
time Rscript ${FIG_DIR}/supplementary_figs/FigS7/logs_vs_levels_hists.R


# Fig S8
echo "Running Fig S8..."
time run_nb ${FIG_DIR}/supplementary_figs/FigS8/beta_correlations.ipynb


# Fig S9
echo "Running Fig S9..."
time run_nb ${FIG_DIR}/supplementary_figs/FigS9/plot_FigS9.ipynb


# Fig S10 generated manually
# Fig S11 generated manually
# Fig S12 generated by Fig4_global_ACS_and_SR/superres/3_sr_figs.ipynb
# Fig S13 generated by Fig4_global_ACS_and_SR/superres/3_sr_figs.ipynb
# Fig S14 generated by Fig4_global_ACS_and_SR/world_model.ipynb
# Fig S15 from previous work


# Fig S16 (skipped due to need for DHS data)
# time Rscript ${FIG_DIR}/supplementary_figs/FigS16/make_nl_features.R
# time run_nb ${FIG_DIR}/supplementary_figs/FigS16/head_replication.ipynb


# Fig S17
echo "Running Fig S17..."
time run_nb ${FIG_DIR}/supplementary_figs/FigS17/cnn_vs_MOSAIKS_scatter.ipynb


# Table S1 generated manually
# Table S2 generated by Fig2_randomCV/1_run_regressions.ipynb


# Table S3
echo "Running Table S3..."
time run_nb ${FIG_DIR}/supplementary_tables/TableS3/logs_vs_levels_model_selection.ipynb


# Table S4 generated manually


# Table S5
echo "Running Table S5/7..."
time run_nb ${FIG_DIR}/supplementary_tables/TableS5/mosaiks_comparisons.ipynb


# Table S6
echo "Running Table S6..."
Rscript ${FIG_DIR}/supplementary_tables/TableS6/head_etal_comparison_table.R


#Table S7 generated by supplementary_tables/TableS5/cnn_mosaiks_combo.ipynb
