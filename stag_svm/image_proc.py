#!/usr/bin/python3
import sys 
import os
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
import cv2 as cv
import glob
import random

from sklearn.cluster import MiniBatchKMeans
from skimage.feature import greycomatrix

"""
options:

	data path - contains ManipulatedFrames, ManipulatedMasks, and OriginalFrames folders with .png files
	#todo percent of training data
	#todo percent of test data
	filters
		sobel     - 1st order (either horiz or vert)
		gabor
		#todo scharr
		laplacian - 2nd order
"""

# Algo related 
f_choice = "laplacian"
KSIZE = 21 
CMAX = 256
FLT = cv.CV_32F
QUANTIZATION_STEP = 5 	# Range [0, 255]
TRUNC_THRESHOLD = 2		# Cutoff for residual vals
GLCM_STRIDE = [1]       # Pel distance for GLCM
GLCM_DIR = [0,90,180,270]

# Input/output related
tmp_imageRootPath = ""
outfilename = "ImageProc.png"
logfilename = "ImageProc.txt"
outpostfix = "_proc_w_mask"
FRAME_FILE_EXT = ".png"

# Flags
ShouldApplyFaceMask = True 
TEST_MODE = True

# Debug related
NUM_PROGRESS_INTERVALS = 10 

filters = {
"laplacian" : lambda x: cv.Laplacian(x, FLT),
"sobel-x": lambda x: cv.Sobel(x, FLT, 1, 0, ksize = KSIZE),
"sobel-y": lambda x: cv.Sobel(x, FLT, 0, 1, ksize = KSIZE),
"gabor": lambda x: cv.filter2D(x, FLT, cv.getGaborKernel((KSIZE, KSIZE), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=FLT))
}

def IsImageNone(image):
	return isinstance(image, type(None))

def Normalize(image, shouldUseFlt=False):
	return cv.normalize(image, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=FLT)\
	if (shouldUseFlt) else cv.normalize(image, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

def ToFeatureVec(image):
	assert not IsImageNone(image), "Image type is None"
	return image.reshape(image.shape[0] * image.shape[1], image.shape[2])

def ComputeResidualImage(image, name="none"):
	if (IsImageNone(image)):
		print("no image")	
		return None

	res_image = filters[f_choice](image)

	# TODO: see Fridrich et al. - Models for steganalysis of Images  
	# 1) quantize pels - Rq_ij = round( Rij / q_step )
	# NOTE: Temporarily cvt to LAB color space (Lightness, Alpha=[G,M], Beta=[B,Y])
	# to yield ftr vector representing euclidean dists for k-means quantization
	input_ftr = ToFeatureVec(cv.cvtColor(Normalize(res_image), cv.COLOR_BGR2LAB))
	kmc = MiniBatchKMeans(n_clusters = 255 // QUANTIZATION_STEP)
	labels = kmc.fit_predict(input_ftr)
	quantized_image = kmc.cluster_centers_.astype("uint8")[labels]	
	# Cvt back to RGB color space
	quantized_image = quantized_image.reshape(res_image.shape)
	quantized_image = cv.cvtColor(quantized_image, cv.COLOR_LAB2BGR)

	# 2) truncate pels - truncate( Rq_ij, T )

	return quantized_image

def LoadMask(image_path):
	if (not os.path.exists(image_path)):
		print("Failed to load image '{}'".format(image_path))
		return None

	return cv.imread(image_path, 0)
  
def LoadImage(image_path, name="none", shouldDisplay=False):
	if (not os.path.exists(image_path)):
		print("Failed to load image '{}'".format(image_path))
		return None

	image = cv.imread(image_path)
	if (shouldDisplay): 
		cv.imshow(name, image)
		cv.waitKey(0)
		print("shape: {}".format(image.shape))
	return image

# NOTE: for now, compute only for single channel - req by skimage.feature api
def ComputeCoOccurence(res_image):
	#return greycomatrix(res_image[0], GLCM_STRIDE, GLCM_DIR)
	return Normalize(res_image)

def ProcessImage(image, mask, name="", shouldDisplay=False):
	assert not IsImageNone(image), "Image to process is None"
	unmasked_image = image
	if (not IsImageNone(mask)): image = cv.bitwise_and(image, image, mask = mask)

	res = ComputeResidualImage(image, name)
	global_feature = ComputeCoOccurence(res)

	#scaler = MinMaxScaler(feature_range=(0, 1))
	#Normalize The feature vectors...
	#rescaled_features = scaler.fit_transform(global_features)	
	if (shouldDisplay): 
		plt.figure(figsize=(16, 12), dpi=100)
		plt.subplot(2,2,1), plt.imshow((unmasked_image))
		plt.title(name)
		plt.subplot(2,2,2), plt.imshow((image))
		plt.title(name)
		plt.subplot(2,2,3), plt.imshow(Normalize(res))
		plt.title("{0}_{1}".format(name, f_choice))
		plt.subplot(2,2,4), plt.imshow((global_feature))
		plt.title("{0}_{1}".format(name, "co_occur"))
		plt.show()

		print("res stats: \n{0}\n{1}\n{2}".format( \
			res.shape, res.max(), res.min()
		))
		print("ftr stats: \n{0}\n{1}\n{2}".format( \
			global_feature.shape, global_feature.max(), global_feature.min()
		))

	return global_feature

def LoadPreProcFrames(frames_path):
	frame_files = [x for x in os.listdir(frames_path) if FRAME_FILE_EXT in x and outpostfix in x]
	frame_files.sort()
	if TEST_MODE:
		frame_files = [frame_files[0]]# random.sample(frame_files,1)

	preproc_frame_list = []
	for frame_file in frame_files:
		preproc_frame = LoadImage(os.path.join(frames_path, frame_file))
		# Minor validation
		if (IsImageNone(preproc_frame) and preproc_frame.any()):
			print("Failed to load '{}'".format(frame_file))
			return []	
		preproc_frame_list.append(preproc_frame)

	return preproc_frame_list

# REFAC: Preload masks and pass in as dict
def ProcessAllFrames(frames_path, mask_path):
		i = 0
		features_list = []
		frame_files = [x for x in os.listdir(frames_path) if FRAME_FILE_EXT in x and not outpostfix in x]
		frame_files.sort()
		if TEST_MODE:
			frame_files = [frame_files[0]]# random.sample(frame_files,1)
		else:	
			assert len(frame_files) >= NUM_PROGRESS_INTERVALS, "Expected more than {} frames".format(NUM_PROGRESS_INTERVALS)
			progress_interval_len = (len(frame_files) // NUM_PROGRESS_INTERVALS)

		log_file = open(os.path.join(frames_path, logfilename), "w")
		log_file.write("frame file, max, mu, min")
		for frame_file in frame_files:
			frame = LoadImage(os.path.join(frames_path, frame_file)) 
			m_frame = LoadMask(os.path.join(mask_path, frame_file))

			if TEST_MODE:
				debug_name = os.path.join(frames_path.replace(tmp_imageRootPath,""), frame_file)
				features = ProcessImage(frame, m_frame if ShouldApplyFaceMask else None, debug_name, TEST_MODE) 
			else:
				features = ProcessImage(frame, m_frame if ShouldApplyFaceMask else None) 

			#
			## Log some stats and save intermediate results (TO RM)
			if (not IsImageNone(features)):
				log_file.write("{0}, {1}, {2}, {3}".format(frame_file.replace(FRAME_FILE_EXT, ""), features.max(), features.mean(), features.min()))
				cv.imwrite(os.path.join(frames_path, frame_file.replace(FRAME_FILE_EXT, outpostfix + FRAME_FILE_EXT)), features)
			else:
				print("Failed to process real/fake frame '{}'".format(frame))
				return []	

			if not TEST_MODE:
				if (i % progress_interval_len == 0):
					print("{}".format(int(100 * i / len(frame_files))))
				i += 1
			## 
			#

			features_list.append(features)

		return features_list

def Run(imageRootPath, shouldUseCachedFrames):
	fake_data_path = os.path.join(imageRootPath, "ManipulatedFrames")
	masks_path = os.path.join(imageRootPath, "ManipulatedMasks")
	real_data_path = os.path.join(imageRootPath, "OriginalFrames")

	if (not os.path.exists(fake_data_path) 
		and not os.path.exists(masks_path) 
		and not os.path.exists(real_data_path)):
		print("Failed to find expected folders under '{}'".format(imageRootPath))
		return 1

	# enumerate original and manipulated images and corresponding masks
	items_in_real_path = os.listdir(real_data_path)
	items_in_fake_path = os.listdir(fake_data_path)
	items_in_mask_path = os.listdir(masks_path)
	for item_name in items_in_real_path:
		subject_id = ""
		subject_real_frames_path = ""
		if (os.path.isdir(os.path.join(real_data_path, item_name))):
			subject_id = item_name
			subject_real_frames_path = os.path.join(real_data_path, subject_id)
		print("Found subject '{}'".format(subject_id))

		subject_id_fake = ""
		subject_fake_frames_path = ""
		subject_mask_frames_path = ""
		print("Seeking corresponding fake seq...")
		if (subject_id):
			subject_id_fake = [x for x in items_in_fake_path if x.startswith(subject_id)][0]
			subject_fake_frames_path = os.path.join(fake_data_path, subject_id_fake)
			subject_mask_frames_path = os.path.join(masks_path, subject_id_fake)

		if (os.path.exists(subject_fake_frames_path)
			and os.path.exists(subject_mask_frames_path)
			and os.path.exists(subject_real_frames_path)):
			print("Processing frames for each seq...".format(subject_id))

			#for frame in glob.glob(os.path.join(subject_real_frames_path, "*.png")):
			frame_files = []
			r_features_list = [] 
			f_features_list = [] 
			if (not shouldUseCachedFrames):
				r_features_list = ProcessAllFrames(subject_real_frames_path, subject_mask_frames_path)
				f_features_list = ProcessAllFrames(subject_fake_frames_path, subject_mask_frames_path)
			else:
				r_features_list = LoadPreProcFrames(subject_real_frames_path)
				f_features_list = LoadPreProcFrames(subject_fake_frames_path)

			if (len(r_features_list) == 0 or len(f_features_list) == 0):
				print("processed frames count: real={}, fake={}".format(r_features.count(), f_features.count()))
				return 2, []
			else:
				print("Finished.")

				#plt.figure(figsize=(16, 12), dpi=100)
				#plt.subplot(2,1,1), plt.imshow(frame)
				#plt.subplot(2,1,2), plt.plot(fd_histogram(frame))
				#plt.show()

		else:
			print("Failed to find frames for subject '{}':\nfake id: {}\npaths: {},{}".format(\
				subject_id, subject_id_fake, subject_fake_frames_path, subject_mask_frames_path))
			return 1, [] 

	return 0, [("real", r_features_list), ("fake", f_features_list)]

if (__name__ == "__main__"):
	if (len(sys.argv) < 2):
		print("Please provide image data path.")
		sys.exit()

	imageRootPath = sys.argv[1]
	tmp_imageRootPath = imageRootPath
	shouldUseCachedFrames = True if len(sys.argv) == 3 and sys.argv[2] == "-c" else False
	print("Args: \nimage root path={}\nshould use cached frames={}".format(\
		imageRootPath, shouldUseCachedFrames))

	exitCode, processed_images_and_labels = Run(imageRootPath, shouldUseCachedFrames)
	print("Exit with error code {}".format(exitCode))
