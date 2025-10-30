# 3d Unet For pelvis segmentation
## Algorithm
This algorithum aims to automatically annotate scans of pelvises to segment features for later processing. This will be acived by implimenting a unet based on WHATEVER SOURCE which can be seen below. 

![Source graph](images/foo)

This basic strucure was modified due to the data set being used having fewer input channels and a larger resolution. The final structure of the implimented unet can be seen below
1 -> 8 -> 16 =========================================================> 16 + 64 -> 32 -> 32 -> 6
          \/                                                                  ^
          16 -> 32 -> 64 ===============================> 64 + 256 -> 128 -> 64
                      \/                                        ^
                      64 -> 128 -> 256 ====> 256+512 -> 256 -> 256
                                   \/             ^
                                   256 -> 256 -> 512
where:
(->): convolution layers
(==>): skip connections
(\/): downsampling
(^): upsampling


## Results
![Graph of results](images/train_vs_val_loss.png)

## Dependancies

## Data
There was minimal preprocessing of the data the main preprocessing was normalising the scan images and one hot encoding of the label images. The original format of the scan annotatinos was
5: prostate
4: anal cavity
3: bladder
2: bone
1: miscilanious tissue
0: background
So this data was one-hot encoded for the final algorithum.

The train-test-valitation split was 80% training and validation with 20% testing, with a further 90%-10% training-validation split, so a 72-8-20 train-validation-test split. This was chosen to maximise the ammount of trainign data, as the dataset has relitively few samples


