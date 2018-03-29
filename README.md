# Convolutional Sparse Auto-Encoders (CSAEs)

The code is the implementation of our work "Convolutional Sparse Auto-Encoders for Image Classification".

The ConvSparseLearning.m and the ConvSparseLearning2.m are the unsupervised learning modules for the 1st and 2nd layers, respectively. 

For inspecting how the CSAE works, you only need to reconfigure "convsparsetrain = 1" in the csae_caltech101_64.m file and load the proper data. 
We provide two preprocessed data in the data folder: Caltech-101 and Berkeley-500. Caltech-101 contains 500 randomly selected samples.

For evaluating the classification performance, you shuold first download the data (), and unzip the MatConvNet.zip file, cause our implementation is based on an earlier 
version of MatConvNet.To get a better performance, you'd better to retrain the CSAE using the new dataset.
