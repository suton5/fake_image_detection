#!/usr/bin/python3
import sys 
import os
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import greycomatrix

"""
params:

    data path - contains ManipulatedFrames, ManipulatedMasks, and OriginalFrames
    percent of training data
    percent of test data
    filters
        sobel     - 1st order (either horiz or vert)
        gabor
        scharr
        laplacian - 2nd order
"""

# Algo related 
F_CHOICE = "laplacian"
KSIZE = 21 
CMAX = 256
FLT = cv.CV_32F
MAX_MASK_W_H = [0, 0]           # Values empirically determined 
QUANTIZATION_STEP = 5           # Range [0, 255]
TRUNC_UPPER_THRESHOLD = 200     # Cutoff for residual vals
GLCM_STRIDE = [1]               # Pel distance for GLCM
GLCM_DIRECTIONS = [0, np.pi / 2, np.pi, 2 * np.pi/3]
DATA_SET_NAMES = ["Face2Face"]#"DeepFakes", "FaceSwap", "NeuralTextures", "Face2Face"]
FILTER_TYPES = ["laplacian", "sobel-x", "sobel-y", "gabor"]
FRAME_FILE_EXT = ".png"
NUM_PROGRESS_INTERVALS = 10 

# Flags
TEST_SINGLE_PASS = False 
ALLOW_OVERWRITE = True

filters = {
"laplacian" : lambda x: cv.Laplacian(x, FLT),
"sobel-x": lambda x: cv.Sobel(x, FLT, 1, 0, ksize = KSIZE),
"sobel-y": lambda x: cv.Sobel(x, FLT, 0, 1, ksize = KSIZE),
"gabor": lambda x: cv.filter2D(x, FLT, cv.getGaborKernel((KSIZE, KSIZE), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=FLT))
}

def LoadMask(image_path):
        if (not os.path.exists(image_path)):
                print("Failed to load image '{}'".format(image_path))
                return None

        return cv.imread(image_path, 0)
  
def LoadImage(image_path):
        if (not os.path.exists(image_path)):
                print("Failed to load image '{}'".format(image_path))
                return None

        return cv.imread(image_path)

def IsMatNone(image):
        return isinstance(image, type(None))

def Normalize(image, should_use_flt=False):
        return cv.normalize(image, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=FLT)\
        if (should_use_flt) else cv.normalize(image, None, 0, 255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

def ToFeatureVec(image):
        assert not IsMatNone(image), "Image type is None"
        return image.reshape(image.shape[0] * image.shape[1], image.shape[2])

def ComputeResidualImage(image, filterName, shouldApplyQT=True):
        assert not IsMatNone(image), "no image"

        # 1) quantize pels - Rq_ij = round( Rij / q_step )
        # NOTE: Temporarily cvt to LAB color space (Lightness, Alpha=[G,M], Beta=[B,Y])
        # to yield ftr vector representing euclidean dists for k-means quantization
        res_image = filters[filterName](image)
        if (not shouldApplyQT):
            return res_image 

        input_ftr = ToFeatureVec(cv.cvtColor(Normalize(res_image), cv.COLOR_BGR2LAB))
        kmc = MiniBatchKMeans(n_clusters = 255 // QUANTIZATION_STEP)
        labels = kmc.fit_predict(input_ftr)
        quantized_image = kmc.cluster_centers_.astype("uint8")[labels]  
        # Cvt back to RGB color space
        quantized_image = quantized_image.reshape(res_image.shape)
        quantized_image = cv.cvtColor(quantized_image, cv.COLOR_LAB2BGR)
        assert(not IsMatNone(quantized_image)), "no quantized image"

        # 2) truncate pels - truncate( Rq_ij, T )
        ret,truncated_image = cv.threshold(quantized_image, TRUNC_UPPER_THRESHOLD, CMAX - 1, cv.THRESH_TRUNC)
        return cv.cvtColor(truncated_image, cv.COLOR_BGR2GRAY)

def ProcessAllFrames(frames_path, outpath, mask_path="", ShouldApplyFaceMask=True, ShouldQuantizeTrunc=True):
    i = 0
    frame_files = [x for x in os.listdir(frames_path) if FRAME_FILE_EXT in x]
    frame_files.sort()
    if (TEST_SINGLE_PASS and len(frame_files) > 5):
        frame_files = frame_files[:5]
    assert(len(frame_files) >= NUM_PROGRESS_INTERVALS)

    # output dirs
    if os.path.exists(outpath):
        if not ALLOW_OVERWRITE:
            print("Skipping '{}' since it already exists.".format(outpath))
            return
    else:
        os.makedirs(outpath)


    i = 0
    progress_interval_len = (len(frame_files) // NUM_PROGRESS_INTERVALS)
    for frame_file in frame_files:
        # We don't convert to grey scale image until co-occurances are computed
        frame = LoadImage(os.path.join(frames_path, frame_file))
        assert not IsMatNone(frame), "no frame to process for '{}'".format(frame_file)
        
        if ShouldApplyFaceMask:
            m_frame = LoadMask(os.path.join(mask_path, frame_file))
            frame = cv.bitwise_and(frame, frame, mask = m_frame)
            frame = frame[np.ix_(m_frame.any(1), m_frame.any(0))]

            # apply mask via CV subtract instead of bitwise op
            #ret, bin_mask = cv.threshold(m_frame, 127, 1, cv.THRESH_BINARY)
            # create 3 channel mask
            #m_frame = cv.cvtColor(m_frame,cv.COLOR_GRAY2BGR)
            #frame = cv.subtract(m_frame,frame)
            #for channel in frame:
            #    frame = frame[np.ix_(bin_mask.any(1), bin_mask.any(0))]

        for filter_name in FILTER_TYPES:
            res = ComputeResidualImage(frame, filter_name, ShouldQuantizeTrunc)
            assert not IsMatNone(res), "failed to compute residuals for '{}'".format(frame_file) 
            cv.imwrite(os.path.join(outpath, frame_file.replace(FRAME_FILE_EXT, '_' + filter_name + FRAME_FILE_EXT)), res)

        if (i % progress_interval_len == 0):
            print("{}".format(int(100 * i / len(frame_files))))
        i += 1


def Run(shouldProcessRealData=False):
    if (len(sys.argv) < 3):
        print("Please provide image data path and subject id (e.g., 183).")
        sys.exit()

    imageRootPath = sys.argv[1]
    subject_id = sys.argv[2]
    print("Args: \nimage root path={}\nsubject id={}\n".format(imageRootPath, subject_id))

    for dataset in DATA_SET_NAMES:
        dataset_root_path = os.path.join(imageRootPath, dataset)
        target_data_path = os.path.join(imageRootPath, "OriginalFrames") if shouldProcessRealData else os.path.join(dataset_root_path, "ManipulatedFrames")
        masks_path = os.path.join(dataset_root_path, "ManipulatedMasks")

        if (not os.path.exists(target_data_path) and not os.path.exists(masks_path)):
            print("Failed to find expected folders under '{}'".format(dataset_root_path))
            return 1

        # enumerate original and manipulated images and corresponding masks
        items_in_target_path = os.listdir(target_data_path)
        items_in_mask_path = os.listdir(masks_path)
        subject_id_target = ""
        subject_target_frames_path = ""
        subject_mask_frames_path = ""
        if (subject_id):
            subject_id_target = [x for x in items_in_target_path if x.startswith(subject_id)][0]
            subject_id_mask = [x for x in items_in_mask_path if x.startswith(subject_id)][0]
            subject_target_frames_path = os.path.join(target_data_path, subject_id_target)
            subject_mask_frames_path = os.path.join(masks_path, subject_id_mask)

        if (os.path.exists(subject_target_frames_path) and os.path.exists(subject_mask_frames_path)):
            print("Processing '{}' frames for subject '{}'\nProgress...".format(dataset, subject_id))
            if (shouldProcessRealData):
                ProcessAllFrames(subject_target_frames_path, os.path.join(target_data_path, dataset + "Res"), subject_mask_frames_path, ShouldApplyFaceMask=True, ShouldQuantizeTrunc=True)
                ProcessAllFrames(subject_target_frames_path, os.path.join(target_data_path, dataset + "ResRaw"), subject_mask_frames_path, ShouldApplyFaceMask=True, ShouldQuantizeTrunc=False)
            else:
                ProcessAllFrames(subject_target_frames_path, target_data_path.replace("Frames", "Res"), subject_mask_frames_path, ShouldApplyFaceMask=True, ShouldQuantizeTrunc=True)
                ProcessAllFrames(subject_target_frames_path, target_data_path.replace("Frames", "ResRaw"), subject_mask_frames_path, ShouldApplyFaceMask=True, ShouldQuantizeTrunc=False)

        else:
            print("Failed to process frames for subject '{}':\nid: {}\npaths: {},{}".format(\
                subject_id, subject_id_target, subject_target_frames_path, subject_mask_frames_path))
            return 1

    return 0

if (__name__ == "__main__"):
    print("Exit with error code {}".format(Run()))
