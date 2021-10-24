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
import spm
import time
import scipy

'''
This code is used for building the visual words based recognition system.
Parts include:
(1) Bulding the recognition system
(2) using Nearest Neighbors for evaluation
'''


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

    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))
    
    
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
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )


def distance_to_set(word_hist, histograms):
    '''
    Compute distance between a histogram of visual words with all training
    image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    '''

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