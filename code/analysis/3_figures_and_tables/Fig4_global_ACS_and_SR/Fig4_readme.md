The scripts and notebooks in this folder generate all results shown in main text Figure 4, panels A, B and C. 

`world_model.ipynb` is a notebook to train, test, and plot the performance of MOSAIKS at global scale. This includes using results from the globally trained model to predict outcomes in four "zoomed" subregions of the globe shown on the right panel of Figure 4A.

`plot_labels_preds_dense_zoom_regions.R` makes plots of the "zoomed" predictions generated in `world_model.ipynb`

`ACS/` has two notebooks which replicate the analysis and plotting of Figure 4B.

`superres/` contains scripts and notebooks used to compute super resolution performance shown in Figure 4C.