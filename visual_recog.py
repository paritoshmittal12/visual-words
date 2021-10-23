import os
import math
import multiprocessing
from os.path import join
from copy import copy,deepcopy

import numpy as np
from PIL import Image
import sklearn
import visual_words
import time
import scipy

'''
This file contains code for 
(1) Computing histogram of visual words
(2) Implementing a Spatial Pyramid Matching of histograms
(3) Save the features from training samples as recognition system
(4) Test time code to evaluate the model
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
    # ----- TODO -----
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
    # ----- TODO -----
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
    # ----- TODO -----
    
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

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255.
    wordmap = visual_words.get_visual_words(opts,img,dictionary)
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

    # ----- TODO -----
    img_path = join(opts.data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255.
    wordmap = visual_words.get_visual_words(opts,img,dictionary)
    image_histograms = get_feature_from_wordmap_SPM(opts,wordmap)
    return [image_histograms,img_label]
    
'''
This function uses Multiprocessing to quickly process
the training samples. A collection of histogram based on SPM
is used as feature and saved for future use.
'''
def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from
    all training images.

    [input]
    * opts: options
    * n_worker: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    
    # ----- TODO -----
    '''
    MultiProcessing step. A process pool is created such that
    each instance receives a subset of task to solve
    '''
    n_process_pool = multiprocessing.Pool(processes=n_worker)
    n_process_params = zip([opts]*len(train_files),
        train_files,train_labels,[dictionary]*len(train_files))
    feature_tuple = n_process_pool.starmap(get_image_feature_parallel,
        n_process_params)
    n_process_pool.close()
    n_process_pool.join()
    features = []
    train_labels = []
    for i in range(len(feature_tuple)):
    	features.append(feature_tuple[i][0])
    	train_labels.append(feature_tuple[i][1])
    
    	
    features = np.asarray(features)
    train_labels = np.asarray(train_labels)
    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def distance_to_set(word_hist, histograms):
    '''
    Compute distance between a histogram of visual words with all training
    image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    return np.subtract(1,np.sum(np.minimum(word_hist,histograms),axis=1))


'''
This function is a helper function used to leverage
multiprocessing while evaluation
'''
def evaluate_each_image_parallel(opts,img_path,labels,
    dictionary,trained_features,train_labels):
	img_path = join(opts.data_dir,img_path)
	test_image_histograms = get_image_feature(opts,img_path,dictionary)
	test_distance = distance_to_set(test_image_histograms,trained_features)
	x = train_labels[np.argmin(test_distance)]
	return [x,labels]

'''
Test samples are evaluated based on 1-NN algorithm.
Visual words are extracted for test images. Then SPM based
matching is used to extract histogram based visual features.
The nearest neighbor in training sample is used as predicted label.
A commented code for parallel processing is included.
'''
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the
    confusion matrix.

    [input]
    * opts: options
    * n_worker: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    train_labels = trained_system['labels']
    # using the stored options in the trained system instead of opts.py
    test_opts = deepcopy(opts)
    test_opts.K = int(dictionary.shape[0])
    test_opts.L = int(trained_system['SPM_layer_num'])
    
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----

    
    # The Following code can be used to perform testing using multiprocessing

    # n_process_pool = multiprocessing.Pool(processes=n_worker)
    # n_process_params = zip([test_opts]*len(
    #     test_files),test_files,test_labels,[dictionary]*len(
    #     test_files),[trained_features]*len(
    #     test_files),[train_labels]*len(test_files))
    # test_tuple = n_process_pool.starmap(
    #     evaluate_each_image_parallel,
    #     n_process_params)
    # n_process_pool.close()
    # n_process_pool.join()
    # test_predictions = []
    # test_labels = []
    # for i in range(len(test_tuple)):
    # 	test_predictions.append(test_tuple[i][0])
    # 	test_labels.append(test_tuple[i][1])
    
    
    test_predictions = []
    for i,test_file in enumerate(test_files):
    	img_path = join(data_dir,test_file)
    	test_image_histograms = get_image_feature(test_opts,
            img_path,dictionary)
    	test_distance = distance_to_set(test_image_histograms,
            trained_features)
    	x = train_labels[np.argmin(test_distance)]
    	test_predictions.append(x)
    	
    test_predictions = np.asarray(test_predictions)
    test_labels = np.asarray(test_labels)
    
    '''
    Accuracy and confusion matrix is computed for the evaluation process
    '''
    acc = sklearn.metrics.accuracy_score(test_labels,test_predictions)
    conf_mat = sklearn.metrics.confusion_matrix(test_labels,test_predictions)
    
    return conf_mat,acc

'''
The Implementation of Custom functions are listed below. 
They are mainly copies of the original functions with slight modifications.
'''

'''
The function is modified to use a custom distance function
'''
def get_image_feature_parallel_custom_comparison(opts, img_path,
    img_label, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts: options
    * img_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    img_path = join(opts.data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255.
    
    '''
    A custom distance function is used to calculate 
    the matching between image features and dictionary.
    '''
    wordmap = visual_words.get_visual_words_varied_distances(opts,img,dictionary)
    image_histograms = get_feature_from_wordmap_SPM(opts,wordmap)
    return [image_histograms,img_label]
    
'''
The function is modified to use a custom distance function. 
Hence the trained model is going to be different.
'''
def build_recognition_system_custom_comparison(opts, n_worker=1):

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    
    # ----- TODO -----
    n_process_pool = multiprocessing.Pool(processes=n_worker)
    n_process_params = zip([opts]*len(train_files),train_files,
        train_labels,[dictionary]*len(train_files))
    feature_tuple = n_process_pool.starmap(
        get_image_feature_parallel_custom_comparison,n_process_params)
    n_process_pool.close()
    n_process_pool.join()
    features = []
    train_labels = []
    for i in range(len(feature_tuple)):
        features.append(feature_tuple[i][0])
        train_labels.append(feature_tuple[i][1])
    
        
    features = np.asarray(features)
    train_labels = np.asarray(train_labels)
    # example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
'''
Function uses most common prediction among N Nearest Neighbors
as the predicted output. The modification gives an ~5% boost in 
accuracy.
'''
def evaluate_recognition_system_custom_neighbors(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the
    confusion matrix.

    [input]
    * opts: options
    * n_worker: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    train_labels = trained_system['labels']
    # using the stored options in the trained system instead of opts.py
    test_opts = deepcopy(opts)
    test_opts.K = int(dictionary.shape[0])
    test_opts.L = int(trained_system['SPM_layer_num'])
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----

    
    # The Following code can be used to perform testing using multi-threading

    # n_process_pool = multiprocessing.Pool(processes=n_worker)
    # n_process_params = zip([test_opts]*len(
    #     test_files),test_files,test_labels,[dictionary]*len(
    #     test_files),[trained_features]*len(
    #     test_files),[train_labels]*len(test_files))
    # test_tuple = n_process_pool.starmap(
    #     evaluate_each_image_parallel_consider_neighbors,
    #     n_process_params)
    # n_process_pool.close()
    # n_process_pool.join()
    # test_predictions = []
    # test_labels = []
    # for i in range(len(test_tuple)):
    #     test_predictions.append(test_tuple[i][0])
    #     test_labels.append(test_tuple[i][1])
    
    '''
    nn corresponds to the number of closest training samples
    whose majority vote is used as predicted label
    '''
    nn = 10
    test_predictions = []
    for i,test_file in enumerate(test_files):
      img_path = join(data_dir,test_file)
      test_image_histograms = get_image_feature(test_opts,img_path,dictionary)
      test_distance = distance_to_set(test_image_histograms,trained_features)
      '''
      We partially sort the array and extract indices of nn training
      samples whose features were closest to the test sample. 
      We then use the most common sample among nn as our
      prediction.
      '''
      x = np.argpartition(np.array(test_distance),nn)
      x,_ = scipy.stats.mode(train_labels[x[:nn]])
      test_predictions.append(x)
    test_predictions = np.asarray(test_predictions)
    test_labels = np.asarray(test_labels)
    
    acc = sklearn.metrics.accuracy_score(test_labels,test_predictions)
    conf_mat = sklearn.metrics.confusion_matrix(test_labels,test_predictions)
    
    return conf_mat,acc

'''
Helper function used for multiprocessing while evaluation. 
Function uses most common prediction among N Nearest Neighbors
as the predicted output
'''
def evaluate_each_image_parallel_consider_neighbors(opts,
    img_path,labels,dictionary,trained_features,train_labels):
    
    img_path = join(opts.data_dir,img_path)
    test_image_histograms = get_image_feature(opts,img_path,dictionary)
    test_distance = distance_to_set(test_image_histograms,trained_features)
    
    x = np.argpartition(np.array(test_distance),10)
    x,_ = scipy.stats.mode(train_labels[x[:10]])
    
    return [x,labels]