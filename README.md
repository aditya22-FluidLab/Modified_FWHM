% Modified FWHM:

This MATLAB code is the modified version of the radon tranform based FWHM algorithm.

Originally written/Edited by Kevin L. Turner, The Pennsylvania State University, Dept. of Biomedical Engineering (https://github.com/KL-Turner) 
and adapted from code written by Dr. Patrick J. Drew (https://github.com/DrewLab).

List of modifications:
1.) Could accept both tiff and .mat files
2.) Provides visual output
3.) 10th percentile approach to deal with various imaging artifacts


% Synthetic data:

img_noise: A 100x100 image array with 91 frames having a NSR of 0.07.
