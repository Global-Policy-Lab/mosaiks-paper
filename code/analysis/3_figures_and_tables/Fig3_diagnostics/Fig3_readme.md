To create this figure, 3 analyses need to be run, followed by the plotting script:

1. `model_sensitivity.ipynb`: Test MOSIAKS model sensitivity to changing training set size and feature vector length.
2. `spatial_analyses_jitter_gammaLoop_interpolation.ipynb`: Test MOSIAKS model sensitivity to changing geographic distance between train and test sets.
3. `train_CNN.py`: Train a RESNET-18 model for prediction with satellite imagery. In order to run the script, run the following command once for each outcome you wish to process (computational costs to run this script on a GPU are provided in Table S5) - `python train_CNN.py ../../../../data/output/cnn_comparison/resnet18_housing.pickle ${MOSAIKS_DATA}/output/cnn_comaprison/resnet18_OUTCOMENAME --outcome OUTCOMENAME`. You will need to have previously downloaded imagery.
4. `run_pretrained_resnet_regressions.ipynb`: Use features from a pre-trained RESNET-152 model within a Ridge Regression framework to predict labels.
5. `fig_3.ipynb`: Plot the full figure.