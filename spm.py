'''
This code was developed as part of the 16-720 A course
by Carnegie Mellon University
'''
import os
import math
import multiprocessing
from os.path import join
from copy import copy,deepcopy

import numpy as np
from PIL import Image
import sklearn
import visual_filters
import time
import scipy

'''
This file contains code for 
(1) Computing histogram of visual words
(2) Implementing a Spatial Pyramid Matching of histograms
'''

'''
Returns a normalized histogram of visual words
'''
def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts: options
    * wordmap: numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    
    hist,_ = np.histogram(wordmap.flatten(),bins = range(K+1),density=True)
    hist = hist.astype(np.float32)
    return hist

'''
Retruns a non-normalized histogram of visual words
this function is used to compute histogram for finest layer
Non-normalization is essential to correctly merge below layer 
in Spatial pyramid for constructing the above layer
'''
def get_feature_from_wordmap_not_norm(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts: options
    * wordmap: numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    
    hist,_ = np.histogram(wordmap.flatten(),bins = range(K+1))
    hist = hist.astype(np.float32)
    return hist


'''
The code for Spatial Pyramid Matching.
'''
def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts: options
    * wordmap: numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    K = opts.K
    L = opts.L
    
    '''
    A weights array is initialized with values.
    L1-norm of weights is 1. They are initialized 
    such that base layer gets maximum weight 
    '''
    weights = np.zeros((L+1)).astype(np.float32)
    for i in range(1,L+1):
    	weights[i] = 2 ** (i-L-1)
    if(len(weights)>1):
    	weights[0] = weights[1]
    else:
    	weights[0] = 2 ** (-L)
    	
    w,h = wordmap.shape
    base_size = 2**L
    w_step,h_step = w//base_size,h//base_size
    print(weights)
    hist_all = np.array([])

    '''
    The algorithm first extracts the histograms for the base layer in SPM
    The image is divided into 2**L x 2**L blocks
    Cropped regions in image are used to create histograms
    base hist stores the histograms in a 1D list
    '''
    base_hist = []
    for i in range(0,base_size):
    	for j in range(0,base_size):
    		finest_hist = get_feature_from_wordmap_not_norm(opts,
                wordmap[j*w_step:(j+1)*w_step,i*h_step:(i+1)*h_step])
    		base_hist.append(finest_hist)
    
    hist_size = base_size
    '''
    The algorithm iteratively builds larger layers of SPM from
    the previous layer. The algorithm uses the previous state (base_hist)
    to compute a higher (combined) hist (curr_hist). The base_hist 
    is then inserted into the final 1D array of hists
    and curr_hist is iteratively used to calculate the higher layer.
    '''
    '''
    base_hist is 1D list and hence we iterate
    in chunks to combine top-left(tl), top-right(tr)
    bottom-left(bl) and bottom-right(br) histograms
    '''
    for l in range(L-1,-1,-1):
    	curr_hist = []
    	for i in range(0,hist_size,2):
    		for j in range(0,hist_size,2):
    			tl = base_hist[i*hist_size+j]
    			tr = base_hist[i*hist_size+j+1]
    			bl = base_hist[(i+1)*hist_size + j]
    			br = base_hist[(i+1)*hist_size + j+1]
    			n_hist = np.add(np.add(tl,tr),np.add(bl,br))
    			curr_hist.append(n_hist)
    	base_hist = np.ravel(base_hist)
    	base_hist /= np.sum(base_hist)
    	base_hist *= weights[l+1]
    	hist_all = np.append(base_hist,hist_all)
    	base_hist = curr_hist
    	hist_size //= 2
    
    base_hist = np.ravel(base_hist)
    base_hist /= np.sum(base_hist)
    base_hist *= weights[0]
    
    hist_all = np.append(base_hist,hist_all)
    
    return hist_all

'''
Function extracts the combined SPM based list of histograms
'''
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts: options
    * img_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255.
    wordmap = visual_filters.get_visual_words(opts,img,dictionary)
    image_histograms = get_feature_from_wordmap_SPM(opts,wordmap)
    return image_histograms

'''
To ensure synchronization, img_label is passed as parameter.
During Multiprocessing, files are not processed in order and hence
we couple the image with it's corresponding label.
This method is used while multiprocessing.
'''
def get_image_feature_parallel(opts, img_path,img_label, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts: options
    * img_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    img_path = join(opts.data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255.
    wordmap = visual_filters.get_visual_words(opts,img,dictionary)
    image_histograms = get_feature_from_wordmap_SPM(opts,wordmap)
    return [image_histograms,img_label]