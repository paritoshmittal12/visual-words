import os
import multiprocessing
from os.path import join, isfile
import time
import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans

'''
This Code is used to Extract Filter Responses
Construct a dictionary of visual words using multiprocessing
And create wordmaps by measuring the similarity between features and dictionary
'''

# Set of global variables that indicate the filters used
gaussian_filter_name = "Gaussian"
LoG_name = "LoG"
DeriOfGX_name = "DeriOfGX"
DeriOfGY_name = "DeriOfGY"
filters = [gaussian_filter_name,LoG_name,DeriOfGX_name,DeriOfGY_name]


'''
This code is used to extract 3*F filter responses for a given image
The code iterates over each scale and for each filter F it extracts 
and stores the response as an additional channel.
'''
def extract_filter_responses(opts, img):
	'''
	Extracts the filter responses for the given image.

	Args:
	* opts: options
	* img: numpy.ndarray of shape (H,W) or (H,W,3)
	'''
	# ----- TODO -----
	filter_scales = opts.filter_scales
	if len(img.shape) < 3:
		img = np.stack([img,img,img],axis=2)
	img = skimage.color.rgb2lab(img)
	filter_responses = img[:,:,0]
	for scale in filter_scales:	
		for F in filters:
			if F == gaussian_filter_name:
				for channel in range(3):
					x = scipy.ndimage.gaussian_filter(
						input=img[:,:,channel],
						sigma=scale)
					filter_responses = np.dstack(
						(filter_responses,x)) 
			elif F == LoG_name:
				for channel in range(3):
					x = scipy.ndimage.gaussian_laplace(
						input=img[:,:,channel],
						sigma=scale)
					filter_responses = np.dstack(
						(filter_responses,x))

			elif F == DeriOfGX_name:
				for channel in range(3):
					x = scipy.ndimage.gaussian_filter(
						input=img[:,:,channel],
						sigma=scale,order=[0,1])
					filter_responses = np.dstack(
						(filter_responses,x))

			elif F == DeriOfGY_name:
				for channel in range(3):
					x = scipy.ndimage.gaussian_filter(
						input=img[:,:,channel],
						sigma=scale,order=[1,0])
					filter_responses = np.dstack(
						(filter_responses,x))
					
	filter_responses = filter_responses[:,:,1:]
	
	return filter_responses

import os
import multiprocessing
from os.path import join, isfile
import time
import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMeans

'''
This Code is used to Extract Filter Responses
Construct a dictionary of visual words using multiprocessing
And create wordmaps by measuring the similarity between features and dictionary
'''

# Set of global variables that indicate the filters used
gaussian_filter_name = "Gaussian"
LoG_name = "LoG"
DeriOfGX_name = "DeriOfGX"
DeriOfGY_name = "DeriOfGY"
filters = [gaussian_filter_name,LoG_name,DeriOfGX_name,DeriOfGY_name]


'''
This code is used to extract 3*F filter responses for a given image
The code iterates over each scale and for each filter F it extracts 
and stores the response as an additional channel.
'''
def extract_filter_responses(opts, img):
	'''
	Extracts the filter responses for the given image.

	Args:
	* opts: options
	* img: numpy.ndarray of shape (H,W) or (H,W,3)
	'''
	# ----- TODO -----
	filter_scales = opts.filter_scales
	if len(img.shape) < 3:
		img = np.stack([img,img,img],axis=2)
	img = skimage.color.rgb2lab(img)
	filter_responses = img[:,:,0]
	for scale in filter_scales:	
		for F in filters:
			if F == gaussian_filter_name:
				for channel in range(3):
					x = scipy.ndimage.gaussian_filter(
						input=img[:,:,channel],
						sigma=scale)
					filter_responses = np.dstack(
						(filter_responses,x)) 
			elif F == LoG_name:
				for channel in range(3):
					x = scipy.ndimage.gaussian_laplace(
						input=img[:,:,channel],
						sigma=scale)
					filter_responses = np.dstack(
						(filter_responses,x))

			elif F == DeriOfGX_name:
				for channel in range(3):
					x = scipy.ndimage.gaussian_filter(
						input=img[:,:,channel],
						sigma=scale,order=[0,1])
					filter_responses = np.dstack(
						(filter_responses,x))

			elif F == DeriOfGY_name:
				for channel in range(3):
					x = scipy.ndimage.gaussian_filter(
						input=img[:,:,channel],
						sigma=scale,order=[1,0])
					filter_responses = np.dstack(
						(filter_responses,x))
					
	filter_responses = filter_responses[:,:,1:]
	
	return filter_responses


'''
The function is used to randomly pick alpha values of filter responses.
The extracted featurs are stored in memory. This helps in synchronization
among multiple threads during multiprocessing.
During multiprocessing, each thread implements a copy of this base function.
'''
def compute_dictionary_one_image(image_path,opts):
	'''
	Extracts a random subset of filter responses of an image and save it to
	disk. This is a worker function called by compute_dictionary.

	Your are free to make your own interface based on how you implement
	compute_dictionary.
	'''

	# ----- TODO -----
	file_path = join(opts.data_dir, image_path)
	img = Image.open(file_path)
	img = np.array(img).astype(np.float32) / 255
	if len(img.shape) < 3:
		img = np.stack([img,img,img],axis=2)

	img_filter_responses = extract_filter_responses(opts,img)
	w,h,f = img_filter_responses.shape
	
	selected_filter_responses = np.empty((opts.alpha,f))
	
	for alpha in range(opts.alpha):
		random_w = np.random.randint(0,w)
		random_h = np.random.randint(0,h)
		xx = img_filter_responses[random_w,random_h,:]
		selected_filter_responses[alpha,:] = xx
	
	'''
	Check if a dir exists, else create the temp dir
	'''
	image_dir = os.path.dirname(image_path)
	if not os.path.exists(opts.feat_dir):
		os.makedirs(opts.feat_dir)

	if not os.path.exists(join(opts.feat_dir,image_dir)):
		os.makedirs(join(opts.feat_dir,image_dir))

	'''
	Extract the name of file from file path
	'''
	split_file_path = image_path.split('/')
	image_name = [x for x in split_file_path if "." in x and ".." not in x]
	image_name = image_name[0].split('.')[0]
	
	np.save(join(join(opts.feat_dir,image_dir),
		image_name+".npy"),selected_filter_responses)

'''
This function uses multiprocessing to compute a dictionary of visual words
Once the features are extracted and stored, the function reads them from memory
and computes K Means clustering
'''
def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts: options
    * n_worker: number of workers to process in parallel

    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''
    
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    
    '''
    MultiProcessing step. A process pool is created such that
    each instance receives a subset of task to solve
    '''
    n_process_pool = multiprocessing.Pool(processes=n_worker)
    n_process_params = zip(train_files,[opts]*len(train_files))
    n_process_pool.starmap(compute_dictionary_one_image,n_process_params)
    n_process_pool.close()
    n_process_pool.join()

    train_filter_responses = np.empty((len(
    	train_files*opts.alpha),3*len(filters)*len(opts.filter_scales)))
    for i,train_file in enumerate(train_files):
    	train_file = train_file.split('.')[0]+'.npy'
    	filter_response = np.load(join(feat_dir,train_file))
    	filter_response = np.array(filter_response)
    	train_filter_responses[i*opts.alpha:(i+1)*opts.alpha,:] = filter_response
    
    kmeans = KMeans(n_clusters=K, n_jobs=n_worker).fit(train_filter_responses)
    dictionary = kmeans.cluster_centers_
    
    if not os.path.exists(out_dir):
    	os.makedirs(out_dir)
    
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

'''
The distance between features and dictionary are used to find the closest
cluster centroid. This centroid number is further used as a visual word for 
that particular pixel.
For optimization, visual features are resized into 2D (from 3D) to avoid looping
'''
def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of
    visual words.

    [input]
    * opts: options
    * img: numpy.ndarray of shape (H,W) or (H,W,3)
    * dictionary: numpy.ndarray of shape (K,3F)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''

    visual_features = extract_filter_responses(opts,img)
    visual_features = np.reshape(visual_features,(
    	-1,visual_features.shape[-1]))
    wordmap = np.argmin(scipy.spatial.distance.cdist(
    	visual_features,dictionary),axis=1)
    wordmap = np.reshape(wordmap,(img.shape[0],img.shape[1]))
    return wordmap


'''
The Implementation of Custom functions are listed below. 
'''

'''
We use this function to use multiple distance metrics to compute the similarity
between the image feature and dictionary feature. 
A linear combination of euclidean and Manhattan is tested here
'''
def get_visual_words_varied_distances(opts, img, dictionary):

    visual_features = extract_filter_responses(opts,img)
    visual_features = np.reshape(visual_features,(-1,visual_features.shape[-1]))
    dist_1 = scipy.spatial.distance.cdist(
    	visual_features,dictionary,metric='cityblock')
    dist_2 = scipy.spatial.distance.cdist(
    	visual_features,dictionary,metric='euclidean')
    wordmap = np.argmin((dist_1+dist_2)/2.,axis=1)
    wordmap = np.reshape(wordmap,(img.shape[0],img.shape[1]))
    return wordmap
