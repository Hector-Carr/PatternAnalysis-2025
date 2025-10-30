# 3d Unet For pelvis segmentation
## Algorithm
This algorithum aims to automatically annotate scans of pelvises to segment features for later processing. This will be acived by implimenting a unet based on [1] which can be seen below. 

![Source graph](images/Source_network.png)

This basic strucure was modified due to the data set being used having fewer input channels and a larger resolution. The final structure of the implimented unet can be seen below
![Implimented network graph](images/Network_graph_implimented.png)\
where:\
(->): convolution layers\
(==>): skip connections\
(\/): downsampling\
(^): upsampling\
(n): numer of channels\

## Dependancies

## Data
There was minimal preprocessing of the data the main preprocessing was normalising the scan images and one hot encoding of the label images. The original format of the scan annotatins was/
5: prostate/
4: anal cavity/
3: bladder/
2: bone/
1: miscilanious tissue/
0: background/
So this data was one-hot encoded for the final algorithum./

The train-test-valitation split was 80% training and validation with 20% testing, with a further 90%-10% training-validation split, so a 72-8-20 train-validation-test split. This was chosen to maximise the ammount of trainign data, as the dataset has relitively few samples./

Another noteable action of preprocessing was the removal of case 19, this was done because of a data resolution mismatch, and given that it was the only case with this issue it was chosen to remove it rather than resize it. To reproduce the results this traning it should be removed. In the case that data is being downloaded from [2] then the command below should be run in the PatternAnalysis-2025/recognition/3d_unet_prostate_segmentation_47447600/data/ directory
```sh
rm  semantic_MRs_anon/Case_019* semantic_labels_anon/Case_019*
```
If the data on rangpur is being used it is recomended to copy the data into the project files and then remove case 19, the commands for that are below again to be run from the PatternAnalysis-2025/recognition/3d_unet_prostate_segmentation_47447600/data/ directory
```sh
cp -r /home/groups/comp3710/HipMRI_study_open/semantic_MRs /home/groups/comp3710/HipMRI_study_open/semantic_labels_only .
rm semantic_MRs/K019* semantic_labels_only/K019*
```


## Results
![Graph of results](images/train_vs_val_loss.png)


## References
1. O. Cicek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, “3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation,” in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016, ser. Lecture Notes in Computer Science, S. Ourselin, L. Joskowicz, M. R. Sabuncu, G. Unal, and W. Wells, Eds. Cham: Springer International Publishing, 2016, pp. 424–432.
2. J. Dowling, P. Greer, “Labelled weekly MR images of the male pelvis ,” Csiro.au, 2025. https://data.csiro.au/collection/csiro:51392v2?redirected=true
