#!/usr/bin/python3
import sys 
import os
from matplotlib import pyplot as plt 
import numpy as np
import pandas as pd
import image_proc
from libsvm.svmutil import * 
from random import sample

SER_FIRST_N_PELS = 100
TRAIN_BATCH_SIZE = 64 
TEST_BATCH_SIZE = 64

"""
The input format expected by svm_read_problem is as follows
<label_0> <index_0>:<value> ... <index_M>:<value>
	...
<label_N> <index_0>:<value> ... <index_M>:<value>

For N training data points with M-dimensional features. This 
file format is parsed into a list of labesl and corresponding 
input data. Each input data is represented as a dictionary, where
each dimension index and value are used as the key and value, resp.
Kind of a strange representation to use for numerical data...
"""
def SerializeAsDict(image, shouldPartSerialize):
	x = {}
	i = 0
	# NOTE: temp crop images for compute
	full_serialied_im = image.reshape(image.shape[0] * image.shape[1],-1)
	p = len(full_serialied_im) // 2
	cropped_serialized_im = full_serialied_im[p:p + image.shape[1]] if shouldPartSerialize else full_serialied_im
	for n in cropped_serialized_im:
		x[i] = n[0]
		i += 1

	# Randomly sample pels
	#for p in sample(list(full_serialied_im), SER_FIRST_N_PELS):
	#	x[i] = p[0]
	#	i += 1
	return x

def HelloWorld(data_path):
	if (not os.path.exists(data_path)):
		print("Failed to find data path '{0}'".format(data_path))
		return None

	y, x = svm_read_problem(data_path)
	m = svm_train(y[:200], x[:200], '-c 4')
	p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)
	return

def RunUsingImages(data_path, shouldReloadData=False):
	# TODO: need to downsize images for training
	print("[WARN] Hack: using cropped image to ease compute ({} pels).".format(SER_FIRST_N_PELS))
	exit_code, real_and_fake_processed_images = image_proc.Run(data_path, shouldReloadData)
	real_processed_images = [SerializeAsDict(im, shouldPartSerialize=True) for im in real_and_fake_processed_images[0][1]]
	fake_processed_images = [SerializeAsDict(im, shouldPartSerialize=True) for im in real_and_fake_processed_images[1][1]]

	data_set = real_processed_images + fake_processed_images
	labels = [+1] * len(real_processed_images) + [-1] * len(fake_processed_images)

	print("--- Train ---")
	assert TRAIN_BATCH_SIZE < len(data_set), "training batch size exceeds data set size"
	train_data = data_set[:TRAIN_BATCH_SIZE] + data_set[len(data_set) - TRAIN_BATCH_SIZE -1:]
	train_labels = labels[:TRAIN_BATCH_SIZE] + labels[len(labels) - TRAIN_BATCH_SIZE -1:]
	model = svm_train(train_labels, train_data, '-c 4')

	print("--- Eval ---")
	assert TEST_BATCH_SIZE < len(data_set), "test batch size exceeds data set size"
	test_data = data_set[TRAIN_BATCH_SIZE + 1:TRAIN_BATCH_SIZE + TEST_BATCH_SIZE]
	test_labels = labels[TRAIN_BATCH_SIZE + 1:TRAIN_BATCH_SIZE + TEST_BATCH_SIZE]
	p_label, p_acc, p_val = svm_predict(test_labels, test_data, model)
	return

def main():
	if (len(sys.argv) != 2):
		print("Please provide data path.")
		sys.exit()

	#HelloWorld(argv[1])
	RunUsingImages(sys.argv[1], shouldReloadData = True)

	return 0

if (__name__ == "__main__"):
	main()