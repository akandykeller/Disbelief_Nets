# Disbelief_Nets
Experiment to test MLP generalization capabilities as a function of depth and layer size. 

Two main experiments were performed. The first experiment involved training MLP's of a range of depths and sizes on both noise and real CIFAR-10 data, measuring test and training error, and comparing the generalization gap between the two. This can be found in cifar_test.py

The second experiment involved training MLP's on binary labeled randomly generated points in [0,1]^d, and measuring as a function of size & depth the average classification accuracy. The goal was to find the number of points n_H for a given architechture for which the model was no longer able to exactly shatter the points. This can be found in shattering_minibatch.py. 

Implementation uses UCSD DeepX (https://github.com/sharadmv/deepx) framework to allow for quick prototyping.
