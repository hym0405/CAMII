#!/usr/bin/env python
import cv2
import os
import sys
import time
import scipy
import pickle
import random
import datetime
import numpy as np
import pandas as pd
from functools import partial
from sklearn import preprocessing, manifold, decomposition
from sklearn.mixture import GaussianMixture
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker
from skimage import morphology
from scipy import ndimage
from Tkinter import *
from PIL import Image,ImageTk

class globalOutputObject(object):
	def __init__(self, total_image):
		self.image_label = [" ",] * total_image
		self.groupID = [" ",] * total_image
		self.image_trans_corrected = [" ",] * total_image
		self.image_epi_corrected = [" ",] * total_image
		self.final_pick = [" ",] * total_image
		self.bad_pick = [" ",] * total_image
		self.all_contours = [" ",] * total_image
		self.all_metadata = [" ",] * total_image
		self.all_metadata_PCA = [" ",] * total_image
		self.plateQC_flag = [" ", ] * total_image

class farthest_points_object(object):
	def __init__(self, iteration):
		self.min_dist_list = [0, ] * iteration
		self.choices_list = [0, ] * iteration

def readConfigureFile(configure_path):
	tmpConfigureOutput = {}
	f = open(configure_path,"rb")
	data = f.readlines()
	f.close()
	for each in data:
		each = each.rstrip()
		if len(each) == 0:
			continue
		if each[0] == "#":
			continue
		tmp = each.split("=")
		varID = tmp[0]
		varValue = tmp[1]
		if "(" == varValue[0]:
			tmp2 = varValue[1:-1].split(",")
			varOutput = tuple([int(e) for e in tmp2])
		elif not is_number(varValue):
			varOutput = varValue
		elif "." in varValue:
			varOutput = float(varValue)
		else:
			varOutput = int(varValue)
		tmpConfigureOutput[varID] = varOutput
	return tmpConfigureOutput

def multi_fun0_detectColonySingleImage(image_trans_path, image_epi_path, image_label, configure_pool, varPool, index):
    image_trans_corrected, image_epi_corrected, all_contours, all_metadata = fun0_detectColonySingleImage(image_trans_path, image_epi_path, image_label, configure_pool)
    varPool.image_label[index] = image_label
    varPool.image_trans_corrected[index] = image_trans_corrected
    varPool.image_epi_corrected[index] = image_epi_corrected
    varPool.all_contours[index] = all_contours
    varPool.all_metadata[index] = all_metadata

def fun0_detectColonySingleImage(image_trans_path, image_epi_path, image_label, configure_pool):
	start_time = time.time()
	cropXMin = configure_pool["cropXMin"]
	cropXMax = configure_pool["cropXMax"]
	cropYMin = configure_pool["cropYMin"]
	cropYMax = configure_pool["cropYMax"]
	size_subSample = configure_pool["size_subSample"]
##	background_STD = configure_pool["background_STD"]
	canny_upper_percentile = configure_pool["canny_upper_percentile"]
	farthest_points_iteration = configure_pool["farthest_points_iteration"]
	calib_parameter_PATH = configure_pool["calib_parameter_PATH"]
	calib_contrast_trans_alpha = configure_pool["calib_contrast_trans_alpha"]
	calib_contrast_trans_beta = configure_pool["calib_contrast_trans_beta"]
	bg_threshold_blockSize = configure_pool["bg_threshold_blockSize"]
	bg_threshold_offset = configure_pool["bg_threshold_offset"]	

	# load calibration parameters
	calib_parameter_list = np.load(calib_parameter_PATH)
	image_trans_calib = calib_parameter_list["image_trans_calib"]
	image_epi_calib_B = calib_parameter_list["image_epi_calib_B"]
	image_epi_calib_G = calib_parameter_list["image_epi_calib_G"]
	image_epi_calib_R = calib_parameter_list["image_epi_calib_R"]

	# load the image
	image_trans_raw = cv2.imread(image_trans_path, 0)
	image_epi_raw = cv2.imread(image_epi_path)
	
	time_dur = round(time.time() - start_time, 2)
	start_time = time.time()
	print "Finish " + image_label + " image loading... (Execution time: " + str(time_dur) + ")"

	# crop the images
	image_trans_crop = crop_image(image_trans_raw, cropXMin, cropXMax, cropYMin, cropYMax)
	image_epi_crop = crop_image(image_epi_raw, cropXMin, cropXMax, cropYMin, cropYMax)
	height_crop, width_crop = image_trans_crop.shape[:2]

	# correct image
	image_trans_corrected = (image_trans_crop.astype(np.float32) / image_trans_calib.astype(np.float32)) * calib_contrast_trans_alpha + calib_contrast_trans_beta
	image_epi_corrected = image_epi_crop.astype(np.float32)
	image_epi_corrected[:,:,0] = image_epi_corrected[:,:,0] / image_epi_calib_B.astype(np.float32)
	image_epi_corrected[:,:,1] = image_epi_corrected[:,:,1] / image_epi_calib_G.astype(np.float32)
	image_epi_corrected[:,:,2] = image_epi_corrected[:,:,2] / image_epi_calib_R.astype(np.float32)



	# copy of original image
	image_to_process = image_trans_corrected.copy()

	## remove background in grayscale
	image_gray_first = image_to_process.copy()
	kernel = 1.0 / (18 - 8) * np.array([[-1,-1,-1], [-1,18,-1], [-1,-1,-1]])
	image_gray_first_con = cv2.filter2D(image_gray_first, -1, kernel)
##	bg_gray_mean, bg_gray_sd = calculate_background_GMM(image_gray_first_con, size_subSample, [125,145])

	image_mask_bg = cv2.adaptiveThreshold(image_gray_first_con.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV,bg_threshold_blockSize,bg_threshold_offset)

##	image_mask_bg = cv2.inRange(image_gray_first, 0, bg_gray_mean - background_STD * bg_gray_sd)
	image_res_bg = cv2.bitwise_and(image_gray_first, image_gray_first, mask = image_mask_bg)
	time_dur = round(time.time() - start_time, 2)
	start_time = time.time()
	print "Finish " + image_label + " background removal... (Execution time: " + str(time_dur) + ")"

	## gaussian blur and perform edge detection
	image_res_GB = cv2.GaussianBlur(image_res_bg, (5, 5), 0)
	upper = calculate_canny_upper(image_res_GB, size_subSample, canny_upper_percentile)
	image_edged = cv2.Canny(image_res_GB.astype(np.uint8), 1, upper)
	image_edged = cv2.dilate(image_edged, None, iterations=3)
	image_edged = cv2.erode(image_edged, None, iterations=1)

	list_output = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	hierarchy = list_output[-1]
	contours = list_output[-2]

	time_dur = round(time.time() - start_time, 2)
	start_time = time.time()	
	print "Finish " + image_label + " first run of colony detection... (Execution time: " + str(time_dur) + ")"

	finalContours, finalDF = filterContours(contours, configure_pool, image_trans_corrected, image_epi_corrected, height_crop, width_crop)

	time_dur = round(time.time() - start_time, 2)
	start_time = time.time()
	print "Finish " + image_label + " colony filtering... (Execution time: " + str(time_dur) + ")"

	post_finalContours = postprocess_contours(finalDF, finalContours, image_trans_corrected, configure_pool)	
	time_dur = round(time.time() - start_time, 2)
	start_time = time.time()	
	print "Finish " + image_label + " multi-colony segmentation... (Execution time: " + str(time_dur) + ")"

	post_finalContours_final = filterContours_final(post_finalContours, configure_pool, image_trans_crop, image_epi_crop, height_crop, width_crop)

	post_finalDF = getFinalData(post_finalContours_final, configure_pool, image_trans_corrected, image_epi_corrected, height_crop, width_crop)
	time_dur = round(time.time() - start_time, 2)
	start_time = time.time()	
	print "Finish " + image_label + " final data gathering... (Execution time: " + str(time_dur) + ")"
	return image_trans_corrected, image_epi_corrected, post_finalContours_final, post_finalDF

def fun1_runPlateQualityControl(image_trans_crop, image_epi_crop, post_finalContours, post_finalDF, image_label, configure_pool):
	cropXMin = configure_pool["cropXMin"]
	cropXMax = configure_pool["cropXMax"]
	cropYMin = configure_pool["cropYMin"]
	cropYMax = configure_pool["cropYMax"]
	size_subSample = configure_pool["size_subSample"]
	canny_upper_percentile = configure_pool["canny_upper_percentile"]
	farthest_points_iteration = configure_pool["farthest_points_iteration"]
	plateQC_colonyContourPixel = configure_pool["plateQC_colonyContourPixel"]
	plateQC_colonyPinSize = configure_pool["plateQC_colonyContourPixel"]
	plateQC_imageScaleFactor = configure_pool["plateQC_imageScaleFactor"] 
	plateQC_imageWidthBias = configure_pool["plateQC_imageWidthBias"]
	plateQC_imageHeightBias = configure_pool["plateQC_imageHeightBias"]
	plateQC_ComfirmWindow = configure_pool["plateQC_ComfirmWindow"]
	plateQC_TextSizeLarge = configure_pool["plateQC_TextSizeLarge"]
	plateQC_TextSizeSmall = configure_pool["plateQC_TextSizeSmall"]
	plateQC_TextSizeButton = configure_pool["plateQC_TextSizeButton"]

	height_crop, width_crop = image_trans_crop.shape

	start_time = time.time()
	image_all_contours = drawContour(image_trans_crop, post_finalContours, plateQC_colonyContourPixel)
	image_all_contours_all_pin = drawPinSite(image_all_contours, post_finalContours, plateQC_colonyPinSize)
	flag = plateQualityControl(image_all_contours_all_pin, plateQC_imageScaleFactor, (width_crop * plateQC_imageScaleFactor + plateQC_imageWidthBias, \
								height_crop * plateQC_imageScaleFactor + plateQC_imageHeightBias), plateQC_ComfirmWindow, \
								image_label, len(post_finalContours), plateQC_TextSizeLarge, plateQC_TextSizeSmall, plateQC_TextSizeButton)
	time_dur = round(time.time() - start_time, 2)
	print "Finish " + image_label + " plate QC... (Execution time: " + str(time_dur) + ")"
	return flag

def fun2_pickColonyPilot(post_finalDF, num_of_colonies, image_label, configure_pool):
	farthest_points_iteration = configure_pool["farthest_points_iteration"]
	start_time = time.time()
	pick_choice, post_finalDF_PCA = pickColonyFirst(post_finalDF, num_of_colonies, farthest_points_iteration)
	time_dur = round(time.time() - start_time, 2)
	print "Finish " + image_label + " first run of colony selection... (Execution time: " + str(time_dur) + ")"
	return pick_choice, post_finalDF_PCA		

def fun3_runColonyQualityControl_group(eachGroupID, groupID_index, varPool, configure_pool, sample_config):
	spacing = configure_pool["colonyQC_image_spacing"]
	fontSize = configure_pool["colonyQC_image_labelSize"]
	fontThickness = configure_pool["colonyQC_image_thickness"]
	groupID_index = findGroupIDindex(varPool, eachGroupID)
	if len(groupID_index) > 0:
		groupID_label_list = [varPool.image_label[i] for i in groupID_index]
		groupID_image_trans_list = [varPool.image_trans_corrected[i] for i in groupID_index]
		groupID_image_epi_list = [varPool.image_epi_corrected[i] for i in groupID_index]
		groupID_contour_list = [varPool.all_contours[i] for i in groupID_index]
		groupID_metadata_list = [varPool.all_metadata[i] for i in groupID_index]
		groupID_totalColonies = sum([len(varPool.all_contours[i]) for i in groupID_index])
		groupID_colony_to_pick = getNumPickColonies(sample_config, eachGroupID)

		groupID_image_merge, groupID_image_heightStart = concatenateImages_gray(groupID_image_trans_list, groupID_label_list, spacing, fontSize, fontThickness)
		groupID_contour_merge = mergeModifyContour(groupID_contour_list, groupID_image_heightStart)
		groupID_metadata_merge = concat_metadata(groupID_metadata_list, groupID_image_heightStart)
		height_crop, width_crop = groupID_image_trans_list[0].shape
		groupID_init_pickChoice, groupID_metadata_merge_PCA = fun2_pickColonyPilot(groupID_metadata_merge, groupID_colony_to_pick, eachGroupID, configure_pool)
		final_pick, bad_pick = fun3_runColonyQualityControl(height_crop, width_crop, groupID_image_merge, groupID_contour_merge, groupID_metadata_merge, \
															groupID_init_pickChoice, groupID_metadata_merge_PCA, eachGroupID, configure_pool)
		tmp_pickStatus = ["not_pick", ] * groupID_metadata_merge.shape[0]
		for e in final_pick:
			tmp_pickStatus[e] = "pick"
		for e in bad_pick:
			tmp_pickStatus[e] = "bad_pick"
		groupID_metadata_merge["pickStatus"] = tmp_pickStatus
		tmp_pickIndex = ["NA", ] * groupID_metadata_merge.shape[0]
		for j in range(len(final_pick)):
			tmp_pickIndex[final_pick[j]] = str(j)
		groupID_metadata_merge["pickIndexGroup"] = tmp_pickIndex
		groupID_metadata_splitIndex = getMetadataLabelIndex(groupID_metadata_merge, groupID_label_list)
		for i in range(len(groupID_index)):
			image_index = groupID_index[i]
			varPool.all_metadata[image_index] = groupID_metadata_merge.iloc[groupID_metadata_splitIndex[i]]
			varPool.all_metadata_PCA[image_index] = groupID_metadata_merge_PCA[groupID_metadata_splitIndex[i]]
			tmpMetadata, tmp_finalPick, tmp_badPick = modifyMetadataSplit(varPool.all_metadata[image_index])
			varPool.all_metadata[image_index] = tmpMetadata
			varPool.final_pick[image_index] = tmp_finalPick
			varPool.bad_pick[image_index] = tmp_badPick		
	else:
		pass

def fun3_runColonyQualityControl(height_crop, width_crop, image_trans_crop, post_finalContours, post_finalDF, pick_choice, post_finalDF_PCA, image_label, configure_pool):
	size_subSample = configure_pool["size_subSample"]
	colonyQC_imageScaleFactor = configure_pool["colonyQC_imageScaleFactor"]
	colonyQC_imageWidthBias = configure_pool["colonyQC_imageWidthBias"]
	colonyQC_imageHeightBias = configure_pool["colonyQC_imageHeightBias"]
	colonyQC_ComfirmWindow = configure_pool["colonyQC_ComfirmWindow"]
	colonyQC_colonyShowBias = configure_pool["colonyQC_colonyShowBias"]
	colonyQC_colonyWindowBias = configure_pool["colonyQC_colonyWindowBias"]
	colonyQC_colonyContourPixel = configure_pool["colonyQC_colonyContourPixel"]
	colonyQC_colonyLabelSize = configure_pool["colonyQC_colonyLabelSize"]
	colonyQC_colonyLabelThickness = configure_pool["colonyQC_colonyLabelThickness"]
	colonyQC_TextSizeLarge = configure_pool["colonyQC_TextSizeLarge"]
	colonyQC_TextSizeMid = configure_pool["colonyQC_TextSizeMid"]
	colonyQC_TextSizeSmall = configure_pool["colonyQC_TextSizeSmall"]
	colonyQC_TextSizeButton = configure_pool["colonyQC_TextSizeButton"]
	colonyQC_colonyColumnNum = 6

	start_time = time.time()
	final_pick, bad_pick = colonyQualityControl(height_crop, width_crop, image_trans_crop, colonyQC_imageScaleFactor, (width_crop * colonyQC_imageScaleFactor + colonyQC_imageWidthBias, \
								height_crop * colonyQC_imageScaleFactor + colonyQC_imageHeightBias), colonyQC_ComfirmWindow, colonyQC_colonyShowBias, \
								(width_crop * colonyQC_imageScaleFactor - colonyQC_colonyWindowBias) / colonyQC_colonyColumnNum, image_label, post_finalContours, \
								pick_choice, post_finalDF, post_finalDF_PCA, colonyQC_colonyContourPixel, colonyQC_colonyLabelSize, colonyQC_colonyLabelThickness, \
								colonyQC_TextSizeLarge, colonyQC_TextSizeMid, colonyQC_TextSizeSmall, colonyQC_TextSizeButton)
	time_dur = round(time.time() - start_time, 2)
	print "Finish " + image_label + " colony QC... (Execution time: " + str(time_dur) + ")"	
	return final_pick, bad_pick



def concatenateImages_gray(image_list, label_list, spacing, fontSize, fontThickness):
	totalImage = len(image_list)
	height_crop, width_crop = image_list[0].shape
	image_label_list = []
	image_height_start = []
	for i in range(len(image_list)):
		eachImage = image_list[i]
		eachLabel = label_list[i]
		blankImage = np.full((spacing, width_crop), 255, dtype = np.float32)
		cv2.putText(blankImage, eachLabel, (fontSize * 10, spacing - fontSize * 10), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 0), fontThickness)
		tmpImage = np.concatenate((blankImage, eachImage), axis=0)
		image_label_list.append(tmpImage)
		image_height_start.append((i + 1) * spacing + i * height_crop)
	out_image = np.concatenate(image_label_list, axis = 0)
	return out_image, image_height_start

def mergeModifyContour(contour_list, image_height_start):
	output_contour_list = []
	for i in range(len(contour_list)):
		tmpContours = contour_list[i]
		tmpHeightStart = image_height_start[i]
		tmpContoursModified = [(e + [0, tmpHeightStart]) for e in tmpContours]
		output_contour_list += tmpContoursModified
	return output_contour_list

def is_number(s):
	try:
		p = float(s)
		return True
	except ValueError:
		return False

def calculate_calib_image(trans_file_path, epi_file_path, configure_pool):
	cropXMin = configure_pool["cropXMin"]
	cropXMax = configure_pool["cropXMax"]
	cropYMin = configure_pool["cropYMin"]
	cropYMax = configure_pool["cropYMax"]
	gaussian_kernal = configure_pool["calib_gaussian_kernal"]
	gaussian_iteration = configure_pool["calib_gaussian_iteration"]	
	calib_parameter_PATH = configure_pool["calib_parameter_PATH"]
	trans_image_list = []
	epi_image_list = []	
	for eachImage in trans_file_path:
		image_trans_tmp = cv2.imread(eachImage, 0)
		trans_image_list.append(crop_image(image_trans_tmp, cropXMin, cropXMax, cropYMin, cropYMax))
	for eachImage in epi_file_path:
		image_epi_tmp = cv2.imread(eachImage)
		epi_image_list.append(crop_image(image_epi_tmp, cropXMin, cropXMax, cropYMin, cropYMax))
	image_trans_calib = calculate_calib_background_gray(trans_image_list, gaussian_kernal, gaussian_iteration)
	image_epi_calib_B, image_epi_calib_G, image_epi_calib_R = calculate_calib_background_BGR(epi_image_list, gaussian_kernal, gaussian_iteration)
	np.savez(calib_parameter_PATH, image_trans_calib = image_trans_calib, image_epi_calib_B = image_epi_calib_B, image_epi_calib_G = image_epi_calib_G, image_epi_calib_R = image_epi_calib_R)

def calculate_calib_background_gray(image_list, gaussian_kernal, gaussian_iteration):
	image_trans_calib_sum = image_list[0].astype(np.float32)
	for e in image_list[1:]:
		image_trans_calib_sum += e.astype(np.float32)
	image_trans_calib_mean = image_trans_calib_sum / len(image_list)
	gaussian_input = image_trans_calib_mean
	for i in range(gaussian_iteration):
		gaussian_input = cv2.GaussianBlur(gaussian_input, gaussian_kernal,0)
	image_trans_calib_mean_blur = gaussian_input / np.mean(gaussian_input)
	return image_trans_calib_mean_blur

def calculate_calib_background_BGR(image_list, gaussian_kernal, gaussian_iteration):
	image_epi_calib_B_sum = image_list[0][:,:,0].astype(np.float32)
	image_epi_calib_G_sum = image_list[0][:,:,1].astype(np.float32)
	image_epi_calib_R_sum = image_list[0][:,:,2].astype(np.float32)
	for e in image_list[1:]:
		image_epi_calib_B_sum += e[:,:,0].astype(np.float32)
		image_epi_calib_G_sum += e[:,:,1].astype(np.float32)
		image_epi_calib_R_sum += e[:,:,2].astype(np.float32)
	image_epi_calib_B_mean = image_epi_calib_B_sum / len(image_list)
	image_epi_calib_G_mean = image_epi_calib_G_sum / len(image_list)
	image_epi_calib_R_mean = image_epi_calib_R_sum / len(image_list)
	gaussian_input_B = image_epi_calib_B_mean
	gaussian_input_G = image_epi_calib_G_mean
	gaussian_input_R = image_epi_calib_R_mean
	for i in range(gaussian_iteration):
		gaussian_input_B = cv2.GaussianBlur(gaussian_input_B, gaussian_kernal,0)
		gaussian_input_G = cv2.GaussianBlur(gaussian_input_G, gaussian_kernal,0)
		gaussian_input_R = cv2.GaussianBlur(gaussian_input_R, gaussian_kernal,0)
	image_epi_calib_B_mean_blur = gaussian_input_B / np.mean(gaussian_input_B)
	image_epi_calib_G_mean_blur = gaussian_input_G / np.mean(gaussian_input_G)
	image_epi_calib_R_mean_blur = gaussian_input_R / np.mean(gaussian_input_R)
	return image_epi_calib_B_mean_blur, image_epi_calib_G_mean_blur, image_epi_calib_R_mean_blur

def calculate_canny_upper(image_res_GB, size_subSample, percentile):
	image_res_GB_flatten = image_res_GB.flatten()
	image_res_GB_flatten_sub = np.random.choice(image_res_GB_flatten, size_subSample, replace=True)
	image_res_GB_flatten_sub_noZero = np.ma.masked_equal(image_res_GB_flatten_sub, 0).compressed()
	return np.percentile(image_res_GB_flatten_sub_noZero, percentile)

def calculate_background_GMM(image_gray_first_con, size_subSample, mean_empirical):
	image_gray_first_flatten = image_gray_first_con.flatten()
	GMM_input_data = np.random.choice(image_gray_first_flatten, size_subSample, replace = True)
	GMM_input = np.zeros((GMM_input_data.shape[0],1))	
	GMM_input[:,0] = GMM_input_data
	mean_init_empirical = np.zeros((2,1))
	mean_init_empirical[:,0] = mean_empirical
	clf_Gray = GaussianMixture(n_components = 2, max_iter = 500, means_init = mean_init_empirical)
	clf_Gray.fit(GMM_input)
	index_Gray = [e for e in range(len(clf_Gray.means_)) if clf_Gray.means_[e] == max(clf_Gray.means_)][0]
	bg_gray_mean = int(clf_Gray.means_[index_Gray][0])
	bg_gray_sd = clf_Gray.covariances_[index_Gray][0][0] ** 0.5
	return bg_gray_mean, bg_gray_sd

def getMetadataLabelIndex(metadata_merge, label_list):
	tmpIndex = {}
	for e in label_list:
		tmpIndex[e] = []
	tmpPlates = list(metadata_merge["plate_barcode"])
	for i in range(metadata_merge.shape[0]):
		tmpIndex[tmpPlates[i]].append(i)
	return [tmpIndex[e] for e in label_list]

def crop_image(image, cropX_min, cropX_max, cropY_min, cropY_max):
	return image[cropY_min:cropY_max, cropX_min:cropX_max]

def getFinalData(contours, configure_pool, image_trans_crop, image_epi_crop, height_crop, width_crop):
	# loop over the contours individually
	finalData = []
	for contour in contours:
		moms = cv2.moments(contour)
		area = moms['m00']
		#calculate center of colony
		x = int((moms['m10'])/(moms['m00']))
		y = int((moms['m01'])/(moms['m00']))
		#calculate radius
		dists = []
		for c in contour:
			a = np.array((x, y))
			b = np.array((c[0][0], c[0][1]))
			dists.append(np.linalg.norm(a-b))
		dists.sort()
		radius = (dists[int((len(dists) - 1)/2)] + dists[int(len(dists)/2)]) / 2
		perim = cv2.arcLength(contour,True)
		circularity = (4*np.pi*area) / (perim**2)
		hullArea = cv2.contourArea(cv2.convexHull(contour))
		convexityRatio = area/hullArea
		denom = np.sqrt((2*moms['mu11'])**2) + ((moms['mu20'] - moms['mu02'])**2)
		eps = .01
		inertiaRatio = 1
		if(denom > eps):
			cosmin = (moms['mu20'] - moms['mu02']) / denom
			sinmin = 2 * moms['mu11'] / denom
			cosmax = -cosmin
			sinmax = -sinmin
			imin = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] - moms['mu02']) * cosmin - moms['mu11'] * sinmin;
			imax = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] - moms['mu02']) * cosmax - moms['mu11'] * sinmax;
			inertiaRatio = imin / imax;
		blackImg = np.zeros((height_crop, width_crop), np.uint8)
		devNull = cv2.fillConvexPoly(blackImg, contour, 255)
		topX = cv2.boundingRect(contour)[0]
		topY = cv2.boundingRect(contour)[1]
		width = cv2.boundingRect(contour)[2]
		height = cv2.boundingRect(contour)[3]
		image_trans_mask = cv2.bitwise_and(image_trans_crop[topY: (topY + height + 1), topX: (topX + width + 1)], \
											image_trans_crop[topY: (topY + height + 1), topX: (topX + width + 1)], \
  											mask = blackImg[topY: (topY + height + 1), topX: (topX + width + 1)])
		image_epi_mask = cv2.bitwise_and(image_epi_crop[topY: (topY + height + 1), topX: (topX + width + 1)], \
											image_epi_crop[topY: (topY + height + 1), topX: (topX + width + 1)], \
											mask = blackImg[topY: (topY + height + 1), topX: (topX + width + 1)])
		gray_list = np.ma.masked_equal(image_trans_mask.flatten(), 0).compressed()
		Bepi_list = np.ma.masked_equal(image_epi_mask[:,:,0].flatten(), 0).compressed()
		Gepi_list = np.ma.masked_equal(image_epi_mask[:,:,1].flatten(), 0).compressed()
		Repi_list = np.ma.masked_equal(image_epi_mask[:,:,2].flatten(), 0).compressed()
		graymean = np.mean(gray_list)
		Repimean = np.mean(Repi_list)
		Gepimean = np.mean(Gepi_list)
		Bepimean = np.mean(Bepi_list)
		graystd = np.std(gray_list) / graymean
		Repistd = np.std(Repi_list) / Repimean
		Gepistd = np.std(Gepi_list) / Gepimean
		Bepistd = np.std(Bepi_list) / Bepimean
		tempDict = {}
		tempDict.update({
			'X': x,
			'Y': y,
			'Radius': radius,
			'Perimeter': perim,
			'Area': area,
			'Circularity': circularity,
			'Convexity': convexityRatio,
			'Inertia': inertiaRatio,
			'Graymean': graymean,
			'Graystd': graystd,
			'Repimean': Repimean,
			'Repistd': Repistd,
			'Gepimean': Gepimean,
			'Gepistd': Gepistd,
			'Bepimean': Bepimean,
			'Bepistd': Bepistd,
		})
		finalData.append(tempDict)
	finalDF = pd.DataFrame(finalData)
	finalDF = finalDF[['X', 'Y', 'Area', 'Perimeter', 'Radius', 'Circularity', 'Convexity', 'Inertia',
						'Graymean', 'Graystd', 'Repimean', 'Repistd', 'Gepimean', 'Gepistd', 'Bepimean', 'Bepistd']]
	for col in list(finalDF.columns.values):
		finalDF[col] = pd.to_numeric(finalDF[col], errors = 'raise')
	return finalDF

def filterContours(contours, configure_pool, image_trans_crop, image_epi_crop, height_crop, width_crop):
	minSize = configure_pool["minSize"]
	maxSize = configure_pool["maxSize"]
	minCircularity = configure_pool["minCircularity"]
	maxCircularity = configure_pool["maxCircularity"]
	smallSizeArea = configure_pool["smallSizeArea"]
	smallSizeCircularity = configure_pool["smallSizeCircularity"]
	minConvexity = configure_pool["minConvexity"]
	maxConvexity = configure_pool["maxConvexity"]
	minInertia = configure_pool["minInertia"]
	maxInertia = configure_pool["maxInertia"]
	minDist = configure_pool["minDist"]
	minDist_pin = configure_pool["minDist_pin"]

	# loop over the contours individually
	finalContours = []
	finalData = []
	for contour in contours:
		moms = cv2.moments(contour)
		area = moms['m00']
		if(area == 0):
			continue;
		#calculate center of colony
		x = int((moms['m10'])/(moms['m00']))
		y = int((moms['m01'])/(moms['m00']))
		#calculate radius
		dists = []
		for c in contour:
			a = np.array((x, y))
			b = np.array((c[0][0], c[0][1]))
			dists.append(np.linalg.norm(a-b))
		dists.sort()
		radius = (dists[int((len(dists) - 1)/2)] + dists[int(len(dists)/2)]) / 2
		perim = cv2.arcLength(contour,True)
		# test size
		if((area < minSize) or (area > maxSize)):
			continue
		# test circularity
		circularity = (4*np.pi*area) / (perim**2)
		if((circularity < minCircularity) or (circularity > maxCircularity)):
			continue
		# test small colonies with bad circularity
		if((circularity < smallSizeCircularity) and (area < smallSizeArea)):
			continue
		#test convexity
		hullArea = cv2.contourArea(cv2.convexHull(contour))
		convexityRatio = area/hullArea
		if ((convexityRatio < minConvexity) or (convexityRatio > maxConvexity)):
			continue
		#test inertia
		denom = np.sqrt((2*moms['mu11'])**2) + ((moms['mu20'] - moms['mu02'])**2)
		eps = .01
		inertiaRatio = 1
		if(denom > eps):
			cosmin = (moms['mu20'] - moms['mu02']) / denom
			sinmin = 2 * moms['mu11'] / denom
			cosmax = -cosmin
			sinmax = -sinmin
			imin = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] - moms['mu02']) * cosmin - moms['mu11'] * sinmin;
			imax = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] - moms['mu02']) * cosmax - moms['mu11'] * sinmax;
			inertiaRatio = imin / imax;
		if((inertiaRatio < minInertia) or (inertiaRatio > maxInertia)):
			continue
		a = np.array((x, y))
		dists = []
		dists_pin = []
		for testContour in contours:
			testMoms = cv2.moments(testContour)
			testArea = testMoms['m00']
			if(testArea == 0):
				continue
			testX = int((testMoms['m10'])/(testMoms['m00']))
			testY = int((testMoms['m01'])/(testMoms['m00']))
			testA = np.array((testX, testY))
			dist = np.linalg.norm(a-testA)
			if dist > 0.001:
				dists.append(dist)
				tmpCoordinates = a - testContour
				tmpDistMin = min(min(np.linalg.norm(tmpCoordinates, axis = 2)))
				dists_pin.append(tmpDistMin)

		if min(dists) < minDist:
			continue
		if min(dists_pin) < minDist_pin:
			continue

		tempDict = {}
		tempDict.update({
			'X': x,
			'Y': y,
			'Radius': radius,
			'Perimeter': perim,
			'Area': area,
			'Circularity': circularity,
			'Convexity': convexityRatio,
			'Inertia': inertiaRatio,
			'Graymean': 0,
			'Graystd': 0,
			'Repimean': 0,
			'Repistd': 0,
			'Gepimean': 0,
			'Gepistd': 0,
			'Bepimean': 0,
			'Bepistd': 0,
		})
		finalData.append(tempDict)
		finalContours.append(contour)

	finalDF = pd.DataFrame(finalData)
	finalDF = finalDF[['X', 'Y', 'Area', 'Perimeter', 'Radius', 'Circularity', 'Convexity', 'Inertia',
						'Graymean', 'Graystd', 'Repimean', 'Repistd', 'Gepimean', 'Gepistd', 'Bepimean', 'Bepistd']]
	for col in list(finalDF.columns.values):
		finalDF[col] = pd.to_numeric(finalDF[col], errors = 'raise')
	return finalContours, finalDF

def filterContours_final(contours, configure_pool, image_trans_crop, image_epi_crop, height_crop, width_crop):
	minSize = configure_pool["minSize"]
	maxSize = configure_pool["maxSize"]
	minCircularity = configure_pool["minCircularity"]
	maxCircularity = configure_pool["maxCircularity"]
	smallSizeArea = configure_pool["smallSizeArea"]
	smallSizeCircularity = configure_pool["smallSizeCircularity"]
	minConvexity = configure_pool["minConvexity"]
	maxConvexity = configure_pool["maxConvexity"]
	minInertia = configure_pool["minInertia"]
	maxInertia = configure_pool["maxInertia"]
	minDist = configure_pool["minDist"]
	minDist_pin = configure_pool["minDist_pin"]

	# loop over the contours individually
	finalContours = []
	for contour in contours:
		moms = cv2.moments(contour)
		area = moms['m00']
		#calculate center of colony
		x = int((moms['m10'])/(moms['m00']))
		y = int((moms['m01'])/(moms['m00']))

		a = np.array((x, y))
		dists = []
		dists_pin = []
		for testContour in contours:
			testMoms = cv2.moments(testContour)
			testArea = testMoms['m00']
			if(testArea == 0):
				continue
			testX = int((testMoms['m10'])/(testMoms['m00']))
			testY = int((testMoms['m01'])/(testMoms['m00']))
			testA = np.array((testX, testY))
			dist = np.linalg.norm(a-testA)
			if dist > 0.001:
				dists.append(dist)
				tmpCoordinates = a - testContour
				tmpDistMin = min(min(np.linalg.norm(tmpCoordinates, axis = 2)))
				dists_pin.append(tmpDistMin)

		if min(dists) < minDist:
			continue
		if min(dists_pin) < minDist_pin:
			continue

		finalContours.append(contour)
	return finalContours

def ZoomInContoursBox(contour,bias):
	xCor = []
	yCor = []
	for e in contour:
		xCor.append(e[0][0])
		yCor.append(e[0][1])
	xCorMin = min(xCor)
	xCorMax = max(xCor)
	yCorMin = min(yCor)
	yCorMax = max(yCor)
	xStart = xCorMin - bias
	xEnd = xCorMax + bias
	yStart = yCorMin - bias
	yEnd = yCorMax + bias
	modifiedContour = contour.copy()
	for i in range(len(modifiedContour)):
		modifiedContour[i][0][0] = modifiedContour[i][0][0] - xStart
		modifiedContour[i][0][1] = modifiedContour[i][0][1] - yStart
	return(modifiedContour,xStart,xEnd,yStart,yEnd)

def postprocess_segmentation(xStart,xEnd,yStart,yEnd, image_subSegmentation, image_subContour):
	totalContours = []
	segmentNum = np.amax(image_subSegmentation)
	for i in range(1,segmentNum + 1):
		image_tmp = image_subContour.copy()
		image_tmp[image_subSegmentation == i] = 255
		image_tmp[image_subSegmentation != i] = 0
		image_tmp_blur = cv2.GaussianBlur(image_tmp, (3, 3), 0)
		(t, binary) = cv2.threshold(image_tmp_blur, 100, 255, cv2.THRESH_BINARY)
		list_output = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		tmp_contours = list_output[-2]
		if len(tmp_contours) == 0:
			continue
		else:
			tmp_contour = tmp_contours[0]
			tmp_contour[:,:,0] += xStart
			tmp_contour[:,:,1] += yStart
			totalContours.append(tmp_contour)
	return(totalContours)

def process_segmentation(image_subContour,image_subContour_binary, configure_pool):
	randowWalker_maxiSize = configure_pool["randowWalker_maxiSize"]
	randomWalker_beta = configure_pool["randomWalker_beta"]
	randomWalker_method = configure_pool["randomWalker_method"]
	# Now we want to separate the two objects in image
	# Generate the markers as local maxima of the distance to the background
	image_subContour_gray = image_subContour
	distance = ndimage.distance_transform_edt(image_subContour_gray)
	local_maxi = peak_local_max(distance, indices = False, footprint = np.ones((randowWalker_maxiSize, randowWalker_maxiSize)), labels = image_subContour_binary)
	markers = morphology.label(local_maxi)
	markers[image_subContour_binary == 0] = -1
	labels_rw = random_walker(image_subContour, markers, multichannel = False, beta = randomWalker_beta, mode = randomWalker_method)
	return(labels_rw)

def post_filterContours(contours, configure_pool):
	post_minSize = configure_pool["post_minSize"]
	maxSize = configure_pool["maxSize"]
	post_minCircularity = configure_pool["post_minCircularity"]
	post_minConvexity = configure_pool["post_minConvexity"]
	minInertia = configure_pool["post_minInertia"]
	maxInertia = configure_pool["maxInertia"]
	# loop over the contours individually
	filteredContours = []
	for contour in contours:
		moms = cv2.moments(contour)
		area = moms['m00']
		if(area == 0):
			continue;      
		#calculate center of colony
		x = int((moms['m10'])/(moms['m00']))
		y = int((moms['m01'])/(moms['m00']))
		#calculate radius
		dists = []
		for c in contour:
			a = np.array((x, y))
			b = np.array((c[0][0], c[0][1]))
			dists.append(np.linalg.norm(a-b))
		dists.sort()
		radius = (dists[int((len(dists) - 1)/2)] + dists[int(len(dists)/2)])/2 
		perim = cv2.arcLength(contour,True)
		# test size
		if((area < post_minSize) or (area > maxSize)):
			continue
		# test circularity
		circularity = (4*np.pi*area) / (perim**2)
		if(circularity < post_minCircularity):
			continue
		#test convexity
		hullArea = cv2.contourArea(cv2.convexHull(contour))
		convexityRatio = area/hullArea
		if(convexityRatio < post_minConvexity):
			continue
		#test inertia
		denom = np.sqrt((2*moms['mu11'])**2) + ((moms['mu20'] - moms['mu02'])**2)
		eps = .01
		inertiaRatio = 1
		if(denom > eps):
			cosmin = (moms['mu20'] - moms['mu02']) / denom;
			sinmin = 2 * moms['mu11'] / denom;
			cosmax = -cosmin;
			sinmax = -sinmin;
			imin = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] - moms['mu02']) * cosmin - moms['mu11'] * sinmin;
			imax = 0.5 * (moms['mu20'] + moms['mu02']) - 0.5 * (moms['mu20'] - moms['mu02']) * cosmax - moms['mu11'] * sinmax;
			inertiaRatio = imin / imax;
		if((inertiaRatio < minInertia) or (inertiaRatio > maxInertia)):
			continue
		filteredContours.append(contour)
	return(filteredContours)

def postprocess_contours(finalDF, finalContours, image_trans_crop, configure_pool):
	post_minCircularity = configure_pool["post_minCircularity"]
	post_minConvexity = configure_pool["post_minConvexity"]
	circularity_threshold = configure_pool["circularity_threshold"]
	circularity_threshold_veryBad = configure_pool["circularity_threshold_veryBad"]
	area_segment_min = configure_pool["area_segment_min"]
	area_segment_max = configure_pool["area_segment_max"]
	segment_bias = configure_pool["segment_bias"]
	filter_bias = configure_pool["filter_bias"]
	post_finalContours = []
	image_trans_copy = image_trans_crop.copy()
	image_trans_gray = image_trans_copy.copy()
	for i in range(len(finalDF)):
		contour = finalContours[i]
		modifiedContour,xStart,xEnd,yStart,yEnd = ZoomInContoursBox(contour,segment_bias + filter_bias)
		if xStart < 0 or yStart < 0 or xEnd >= image_trans_crop.shape[1] or yEnd >= image_trans_crop.shape[0]:
			continue
		ifPostProcessFlag = 0
		if finalDF['Circularity'][i] < circularity_threshold and finalDF['Circularity'][i] >= circularity_threshold_veryBad:
			if finalDF['Area'][i] > area_segment_min and finalDF['Area'][i] < area_segment_max:
				ifPostProcessFlag = 1
		elif finalDF['Circularity'][i] < circularity_threshold_veryBad:
			ifPostProcessFlag = 1
		if ifPostProcessFlag == 0:
			if finalDF['Circularity'][i] > post_minCircularity and finalDF['Convexity'][i] > post_minConvexity:
				post_finalContours.append(contour)
		else:
			modifiedContour,xStart,xEnd,yStart,yEnd = ZoomInContoursBox(contour, segment_bias)
			image_subContour_binary = np.zeros((yEnd - yStart + 1,xEnd - xStart + 1), dtype = np.uint8)
			cv2.drawContours(image_subContour_binary,[modifiedContour],0,255,-1)
			cv2.drawContours(image_subContour_binary,[modifiedContour],0,255,1)
			image_subContour = cv2.bitwise_and(image_trans_copy[yStart:(yEnd + 1), xStart:(xEnd + 1)].astype(np.uint8), \
												image_trans_copy[yStart:(yEnd + 1), xStart:(xEnd + 1)].astype(np.uint8), \
												mask = image_subContour_binary)
			image_subSegmentation = process_segmentation(image_subContour.copy(), image_subContour_binary.copy(), configure_pool)
			post_contours = postprocess_segmentation(xStart,xEnd,yStart,yEnd, image_subSegmentation, image_subContour.copy())
			post_contours_filtered = post_filterContours(post_contours, configure_pool)
			for e in post_contours_filtered:
				post_finalContours.append(e)
	return(post_finalContours)

def generateMaskedSubImage(image_trans_copy, contour, colony_blank_offset, colony_contour_offset):
	modifiedContour,xStart,xEnd,yStart,yEnd = ZoomInContoursBox(contour, colony_blank_offset)
	image_subContour_binary = np.zeros((yEnd - yStart + 1,xEnd - xStart + 1), dtype = np.uint8)
	cv2.drawContours(image_subContour_binary,[modifiedContour],0,255,-1)
	cv2.drawContours(image_subContour_binary,[modifiedContour],0,255,colony_contour_offset)
	image_subContour = cv2.bitwise_and(image_trans_copy[yStart:(yEnd + 1), xStart:(xEnd + 1)], \
										image_trans_copy[yStart:(yEnd + 1), xStart:(xEnd + 1)], \
										mask = image_subContour_binary)
	return image_subContour

def plateQualityControl(image_trans_crop, resize_factor, main_window_size, confirm_window_size, image_label, num_of_colony, textSizeLarge, textSizeSmall, textSizeButton):
	max_index = 0
	class globalVarObject(object):
		def __init__(self):
			self.qc_plate_flag = self
	while True:
		globalVar = globalVarObject()
		globalVar.qc_plate_flag = -1
		root_size = [int(e) for e in main_window_size]
		confirm_size = [int(e) for e in confirm_window_size]
		root = Tk()
		root.title("All colonies after QC on plate: " + image_label)
		root.geometry("x".join([str(root_size[0]), str(root_size[1])]))
		root.resizable(width=False, height=False)
		root.config(cursor="arrow")
		height_crop, width_crop, trashValue = image_trans_crop.shape
		resize_image = cv2.resize(image_trans_crop, (int(width_crop * resize_factor), int(height_crop * resize_factor)))
		final_image = resize_image
		current_image = Image.fromarray(final_image)
		imgtk = ImageTk.PhotoImage(image = current_image)
		panel = Label(root, image = imgtk)
		panel.grid(row = 0, column = 0, columnspan = 17, sticky = W+E+N+S, padx = 5, pady=5)
		but_test = Label(root, text="Keep this plate?", font=('Times', textSizeLarge))
		but_test.grid(row = 1, column = 7, columnspan = 3, sticky = W + E + S, padx = 5, pady = 0)
		but_test2 = Label(root, text="(total number of colonies on this plate: " + str(num_of_colony) + ")", font=('Times', textSizeSmall))
		but_test2.grid(row = 2, column = 5, columnspan = 7, sticky = W + E + S, padx = 5, pady = 0)
		def click_Yes(varPool):
			time.sleep(0.1)
			top = Toplevel()
			top.title('Confirm')
			top.geometry("x".join([str(confirm_size[0]), str(confirm_size[1])]))
			confirm_test = Label(top, text="Keep this plate?", font=('Times', textSizeLarge))
			confirm_test.grid(row = 0, column = 0, columnspan = 5, sticky = W + E + S, padx = 10, pady = 10)
			def click_confirm_Yes(varPool_1):
				varPool_1.qc_plate_flag = 1
				time.sleep(0.1)
				top.destroy()
				root.destroy()
			click_confirm_Yes_with_arg = partial(click_confirm_Yes, varPool)
			confirm_button = Button(top, text='Yes', command = click_confirm_Yes_with_arg, foreground = "green", font=('Times', textSizeButton))
			confirm_button.grid(row = 1, column = 2, padx = 1,pady = 1, sticky = W+E+N)
		click_Yes_with_arg = partial(click_Yes, globalVar)
		def click_No(varPool):
			time.sleep(0.1)
			top = Toplevel()
			top.title('Confirm')
			top.geometry("x".join([str(confirm_size[0]), str(confirm_size[1])]))
			confirm_test = Label(top, text="Keep this plate?", font=('Times', textSizeLarge))
			confirm_test.grid(row = 0, column = 0, columnspan = 5, sticky = W + E + S, padx = 10, pady = 10)
			def click_confirm_No(varPool_1):
				varPool_1.qc_plate_flag = 0
				time.sleep(0.1)
				top.destroy()
				root.destroy()
			click_confirm_No_with_arg = partial(click_confirm_No, varPool)
			confirm_button = Button(top, text='No', command = click_confirm_No_with_arg, foreground = "red", font=('Times', textSizeButton))
			confirm_button.grid(row = 1, column = 2, padx = 1,pady = 1, sticky = W+E+N)
		click_No_with_arg = partial(click_No, globalVar)
		but_yes = Button(root, text="Yes", command = click_Yes_with_arg,  foreground = "green", font=('Times', textSizeButton))
		but_no = Button(root, text="No", command = click_No_with_arg, foreground = "red", font=('Times', textSizeButton))
		but_yes.grid(row = 3, column = 9, sticky = W+E+N, padx = 5, pady = 5)
		but_no.grid(row = 3, column = 7, sticky = W+E+N, padx = 5, pady = 5)
		root.mainloop()
		if max_index == 10:
			return True
		else:
			max_index += 1
		if globalVar.qc_plate_flag == 0:
			return False
		if globalVar.qc_plate_flag == 1:
			return True
		if globalVar.qc_plate_flag == -1:
			continue

def drawContour(image_trans_crop, contours, pixel):
	image_output = image_trans_crop.copy()
	for contour in contours:
		cv2.drawContours(image_output,[contour],0, [0, 0, 0], pixel)
	return image_output

def drawPinSite(image_all_contours, contours, pixel):
	image_output = cv2.cvtColor(image_all_contours.astype(np.uint8), cv2.COLOR_GRAY2BGR)
	for contour in contours:
		moms = cv2.moments(contour)
		x = int((moms['m10'])/(moms['m00']))
		y = int((moms['m01'])/(moms['m00']))
		cv2.circle(image_output, (x, y), pixel, (0, 255, 0), -1)
	return image_output

def drawContourLabel(image_trans_crop, finalDF, label_list, fontScale, thickness):
	image_output = image_trans_crop.copy().astype(np.uint8)
	for index, row in finalDF.iterrows():
		cv2.putText(image_output, str(label_list[index]), (int(row["X"]), int(row["Y_concat"])), \
					cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
	return image_output

def concat_metadata(metadata_list, heightStart):
	tmpList = []
	for i in range(len(metadata_list)):
		tmp_metadata = metadata_list[i]
		tmp_Height = heightStart[i]
		tmp_metadata["Y_concat"] = [e + tmp_Height for e in list(tmp_metadata["Y"])]
		tmpList.append(tmp_metadata)
	return pd.concat(tmpList)

def transform_data_PCA(num_mats, pca_dims):
	standardized = {}
	X_scaled = preprocessing.scale(np.asarray(num_mats))
	standardized = X_scaled
	pca = decomposition.PCA(pca_dims)
	pca_mats = pca.fit_transform(standardized)
	return pca_mats


def farthest_points_parallel(data, n, threadIndex, outputObject):
	dist_mat = scipy.spatial.distance.cdist(data, data, metric="euclidean")
	r = random.sample(range(data.shape[0]), n)
	r_old = None
	while r_old != r:
		r_old = r[:]
		for i in range(n):
			no_i = r[:]
			no_i.pop(i) 
			cols_in_play = np.asarray(range(dist_mat.shape[1]))[np.newaxis, :][:, filter(lambda n: n not in no_i, range(dist_mat.shape[1]))] 
			mm = dist_mat[no_i, :][:, filter(lambda n: n not in no_i, range(dist_mat.shape[1]))] 
			max_min_dist = np.argmax(np.min(mm, 0)) 
			r[i] = cols_in_play[0, :][max_min_dist]
	outputObject.choices_list[threadIndex] = r
	outputObject.min_dist_list[threadIndex] = np.max(np.min(mm, 0))

def farthest_points(data, n):
	dist_mat = scipy.spatial.distance.cdist(data, data, metric="euclidean")
	r = random.sample(range(data.shape[0]), n)
	r_old = None
	while r_old != r:
		r_old = r[:]
		for i in range(n):
			no_i = r[:]
			no_i.pop(i) 
			cols_in_play = np.asarray(range(dist_mat.shape[1]))[np.newaxis, :][:, filter(lambda n: n not in no_i, range(dist_mat.shape[1]))] 
			mm = dist_mat[no_i, :][:, filter(lambda n: n not in no_i, range(dist_mat.shape[1]))] 
			max_min_dist = np.argmax(np.min(mm, 0)) 
			r[i] = cols_in_play[0, :][max_min_dist]
	return r, np.max(np.min(mm, 0))

def pickColonyFirst(finalDF, num_of_pick, iteration):
	feats = finalDF[['Area', 'Perimeter', 'Radius', 'Circularity', 'Convexity', 'Inertia', \
         'Graymean', 'Graystd', 'Repimean', 'Repistd', \
         'Gepimean', 'Gepistd', 'Bepimean', 'Bepistd']]
	feats_list = feats.values.tolist()
	preprocessed_plates = transform_data_PCA(feats_list, 2)
	thresh = min(num_of_pick, int(len(feats)))

	if num_of_pick > preprocessed_plates.shape[0]:
		return range(preprocessed_plates.shape[0]), preprocessed_plates

	max_min_dist = 0.0
	best_choices = []
	startTime = time.time()
	print "Start farthest points optimization"
	for robust_iter in range(iteration):
		choices, min_dist = farthest_points(preprocessed_plates, thresh)
		if best_choices == [] or min_dist > max_min_dist:
			max_min_dist = min_dist
			best_choices = choices[:]
		time_dur = round(time.time() - startTime, 2)
		print "Finish farthest points iteration " + str(robust_iter) + "... (Execution time: " + str(time_dur) + ")"
		startTime = time.time()
	choices = best_choices
	choices.sort()
	return choices, preprocessed_plates

def reSelectColony(num_of_pick, previous_pick, ignore_pick, post_finalDF_PCA):
	num_of_total = len(post_finalDF_PCA)
	dist_mat = scipy.spatial.distance.cdist(post_finalDF_PCA, post_finalDF_PCA, metric="euclidean")
	candidates = []
	for i in range(num_of_total):
		if (i not in previous_pick) and (i not in ignore_pick):
			tmpDist = min([dist_mat[i, e] for e in previous_pick])
			candidates.append([tmpDist, i])
	candidates.sort()
	candidates.reverse()
	final_pick = [candidates[i][1] for i in range(num_of_pick)]
	return final_pick

def generateContourSubImage_QC(image_trans_crop, contour, midpoint, segment_bias, final_size, pixel, label, fontScale, thickness):
	image_output = cv2.cvtColor(image_trans_crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)
	cv2.drawContours(image_output,[contour], 0, [0,0,0], pixel)
	cv2.circle(image_output, midpoint, 2, (0, 255, 0), -1)
	height_crop, width_crop = image_trans_crop.shape[:2]
	modifiedContour,xStart,xEnd,yStart,yEnd = ZoomInContoursBox(contour,segment_bias)
	xLength = xEnd - xStart + 1
	yLength = yEnd - yStart + 1
	if xLength > yLength:
		if (xLength - yLength) % 2 == 0:
			xEnd_f = min(xEnd, width_crop - 1)
			xStart_f = max(xStart, 0)
			yEnd_f = min(yEnd + (xLength - yLength) / 2, height_crop - 1)
			yStart_f = max(yStart - (xLength - yLength) / 2, 0)
		else:
			xEnd_f = min(xEnd, width_crop - 1)
			xStart_f = max(xStart, 0)
			yEnd_f = min(yEnd + int((xLength - yLength) / 2) + 1, height_crop - 1)
			yStart_f = max(yStart - int((xLength - yLength) / 2), 0)
	else:
		if (yLength - xLength) % 2 == 0:
			yEnd_f = min(yEnd, height_crop - 1)
			yStart_f = max(yStart, 0)
			xEnd_f = min(xEnd + (yLength - xLength) / 2, width_crop - 1)
			xStart_f = max(xStart - (yLength - xLength) / 2, 0)
		else:
			yEnd_f = min(yEnd, height_crop - 1)
			yStart_f = max(yStart, 0)
			xEnd_f = min(xEnd + int((yLength - xLength) / 2) + 1, width_crop - 1)
			xStart_f = max(xStart - int((yLength - xLength) / 2), 0)
	image_sub_contour = image_output[yStart_f:(yEnd_f + 1), xStart_f:(xEnd_f + 1)]
	resize_image = cv2.resize(image_sub_contour, (final_size, final_size))
	cv2.putText(resize_image, str(label), (5, final_size - 6), cv2.FONT_HERSHEY_SIMPLEX, \
				fontScale, (0,0,0), thickness, cv2.LINE_AA, bottomLeftOrigin = False)
	return resize_image

def addRedXtoImage(image_to_add, pixel):
	if len(image_to_add.shape) == 2:
		image_output = cv2.cvtColor(image_to_add.astype(np.uint8), cv2.COLOR_GRAY2BGR)
	else:
		image_output = image_to_add.copy()
	pic_size = image_to_add.shape[0]
	cv2.line(image_output, (0, 0), (pic_size - 1, pic_size - 1), [0,0,255], pixel)
	cv2.line(image_output, (pic_size - 1, 0), (0, pic_size - 1), [0,0,255], pixel)
	return image_output

def image_to_tk(input_image):
	if len(input_image.shape) == 2:
		final_image = input_image
		current_image = Image.fromarray(final_image)
	else:
		final_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGBA)
		current_image = Image.fromarray(final_image.astype(np.uint8))
	imgtk = ImageTk.PhotoImage(image = current_image)
	return imgtk

def colonyQualityControl(height_crop, width_crop, image_trans_crop, resize_factor, main_window_size, confirm_window_size, colony_show_bias, \
						colony_show_size, image_label, post_finalContours, pick_choice, post_finalDF, post_finalDF_PCA, \
						colonyQC_colonyContourPixel, colonyQC_colonyLabelSize, colonyQC_colonyLabelThickness, \
						colonyQC_TextSizeLarge, colonyQC_TextSizeMid, colonyQC_TextSizeSmall, colonyQC_TextSizeButton):
	max_index = 0
	class globalVarObject(object):
		def __init__(self):
			self.breakFlag = self
			self.outputPick = self
			self.finalPick = self
			self.colony_button_flag = self
			self.remaining_colonies = self
			self.pick_finalContours = self
			self.bad_colonies = self
			self.num_of_pick = self
			self.pick_finalDF = self
			self.image_picked_contours = self
			self.image_picked_contours_label = self
			self.imgtk = self
			self.pick_colony_tk = self
			self.pick_colony_bad_tk = self

	globalVar = globalVarObject()
	globalVar.breakFlag = 0
	globalVar.bad_colonies = []
	globalVar.num_of_pick = len(pick_choice)
	globalVar.finalPick = [i for i in pick_choice]
	globalVar.pick_finalDF = post_finalDF.iloc[globalVar.finalPick]
	globalVar.pick_finalDF.index = range(len(globalVar.finalPick))
	globalVar.pick_finalContours = [post_finalContours[i] for i in globalVar.finalPick]
	globalVar.image_picked_contours = drawContour(image_trans_crop, globalVar.pick_finalContours, colonyQC_colonyContourPixel)
	globalVar.image_picked_contours_label = drawContourLabel(globalVar.image_picked_contours, globalVar.pick_finalDF, range(1, len(globalVar.finalPick) + 1), colonyQC_colonyLabelSize, colonyQC_colonyLabelThickness)
	globalVar.colony_button_flag = []

	height_crop_merge, width_crop_merge = image_trans_crop.shape

	for i in range(len(globalVar.finalPick)):
		globalVar.colony_button_flag.append(0)
	globalVar.remaining_colonies = len(post_finalContours) - len(globalVar.finalPick)	
	globalVar.outputPick = []
	root_size = [int(e) for e in main_window_size]
	confirm_size = [int(e) for e in confirm_window_size]
	colony_show_bias = int(colony_show_bias)
	colony_show_size = int(colony_show_size)
	while True:
		root = Tk()
		root.title("Colonies QC: " + image_label)
		root.geometry("x".join([str(root_size[0]), str(root_size[1])]))
		root.resizable(width=False, height=False)
		root.config(cursor="arrow")
		cv2image = globalVar.image_picked_contours_label
		resize_image = cv2.resize(cv2image, (int(width_crop_merge * resize_factor * 18 / 19), int(height_crop_merge * resize_factor * 18 / 19)))
		globalVar.imgtk = image_to_tk(resize_image)
		globalVar.pick_colony_tk = []
		globalVar.pick_colony_bad_tk = []
		for i in range(len(globalVar.finalPick)):
			tmpSub = generateContourSubImage_QC(image_trans_crop, globalVar.pick_finalContours[i], (globalVar.pick_finalDF["X"][i], globalVar.pick_finalDF["Y_concat"][i]), \
												colony_show_bias, colony_show_size, 1, str(i + 1), colony_show_size / 200.0, int(colony_show_size / 100.0))
			globalVar.pick_colony_bad_tk.append(image_to_tk(addRedXtoImage(tmpSub, 2)))
			globalVar.pick_colony_tk.append(image_to_tk(tmpSub))
		def show_whole_image(varPool):

			frm_whole = Frame(root, width = str(int(width_crop * resize_factor)), height = str(int(height_crop * resize_factor)))
			canvas_whole = Canvas(frm_whole, width = str(int(width_crop * resize_factor) - 30), height = str(int(height_crop * resize_factor)))
			frm_image_whole = Frame(canvas_whole)
			myscrollbar_whole = Scrollbar(frm_whole, orient = "vertical", command = canvas_whole.yview)
			canvas_whole.configure(yscrollcommand = myscrollbar_whole.set)
            
			def myfunction(event):
				canvas_whole.configure(scrollregion = canvas_whole.bbox("all"))
                
			canvas_whole.create_window((0,0), window = frm_image_whole, anchor = 'nw')
			myscrollbar_whole.grid(row = 0, column = 18, sticky = W + N + S)
			canvas_whole.grid(row = 0, column = 0, columnspan = 18)
			frm_whole.grid(row = 0, column = 0, columnspan = 19, padx = 20, pady = 5)
			frm_image_whole.bind("<Configure>", myfunction)

			panel = Label(frm_image_whole, image = varPool.imgtk)
			panel.grid(row = 0, column = 0, columnspan = 18, sticky = W+E+N+S, padx = 5, pady=5)

			but_test = Label(root, text="View all selected colonies on plates", font=('Times', colonyQC_TextSizeLarge))
			but_test.grid(row = 1, rowspan = 2, column = 5, columnspan = 9, sticky = W + E, padx = 5, pady = 10)
			def click_continue(varPool_1):
				panel.grid_forget()
				but_test.grid_forget()
				but_continue.grid_forget()
				show_ind_image(varPool_1)
			click_continue_with_arg = partial(click_continue, varPool)
			but_continue = Button(root, text="Continue", command = click_continue_with_arg, foreground = "black", font=('Times', colonyQC_TextSizeButton))
			but_continue.grid(row = 3, column = 9, sticky = W+E+N, padx = 5, pady = 5)
		def show_ind_image(varPool):
			frm = Frame(root, width = str(int(width_crop * resize_factor)), height = str(int(height_crop * resize_factor)))
			canvas = Canvas(frm, width = str(int(width_crop * resize_factor) - 30), height = str(int(height_crop * resize_factor)))
			frm_but = Frame(canvas)
			myscrollbar = Scrollbar(frm, orient = "vertical", command = canvas.yview)
			canvas.configure(yscrollcommand = myscrollbar.set)
			colony_button_list = []
 			def button_fun(args):
				k = args[0]
				varPool_1 = args[1]
				if varPool_1.colony_button_flag[k] == 0:
					varPool_1.colony_button_flag[k] = 1
					colony_button_list[k].config(image = varPool_1.pick_colony_bad_tk[k])
				else:
					varPool_1.colony_button_flag[k] = 0
					colony_button_list[k].config(image = varPool_1.pick_colony_tk[k])
			for i in range(len(varPool.finalPick)):
				button_fun_with_arg = partial(button_fun, (i, varPool))
				if varPool.colony_button_flag[i] == 0:
					tmpButton = Button(frm_but, command = button_fun_with_arg, image = varPool.pick_colony_tk[i])
				else:
					tmpButton = Button(frm_but, command = button_fun_with_arg, image = varPool.pick_colony_bad_tk[i])
				tmpButton.grid(row = i / 6, column =  3 * (i % 6), columnspan = 3, padx = 7, pady = 10)
				colony_button_list.append(tmpButton)
			def myfunction(event):
				canvas.configure(scrollregion = canvas.bbox("all"))
			canvas.create_window((0,0),window = frm_but,anchor = 'nw')
			myscrollbar.grid(row = 0, column = 18, sticky = W + N + S)
			canvas.grid(row = 0, column = 0, columnspan = 18)
			frm.grid(row = 0, column = 0, columnspan = 19, padx = 20, pady = 5)
			frm_but.bind("<Configure>", myfunction)
			but_text1 = Label(root, text="Please select bad colonies" , font=('Times', colonyQC_TextSizeMid))
			but_text1.grid(row = 1, column = 5, columnspan = 8, sticky = W + E + S, padx = 5, pady = 0)
			but_text2 = Label(root, text="(remaining colonies for re-select: " + str(varPool.remaining_colonies) + ")" , font=('Times', colonyQC_TextSizeSmall))
			but_text2.grid(row = 2, column = 5, columnspan = 8, sticky = W + E + N, padx = 5, pady = 0)
			def click_back(varPool_1):
				but_text1.grid_forget()
				but_text2.grid_forget()
				but_back.grid_forget()
				but_re.grid_forget()
				but_finish.grid_forget()
				frm.grid_forget()
				show_whole_image(varPool_1)
			click_back_with_arg = partial(click_back, varPool)
			def click_reselect(varPool_1):
				tmpPick = [varPool_1.finalPick[i] for i in range(len(varPool_1.finalPick)) if varPool_1.colony_button_flag[i] == 0]
				if len(tmpPick) == 0:
					time.sleep(0.1)
					top = Toplevel()
					top.title('Warning')
					top.geometry("x".join([str(confirm_size[0] + 60), str(confirm_size[1])]))
					warning_test = Label(top, text="At least one colony should be kept for re-pick", font=('Times', colonyQC_TextSizeMid))
					warning_test.grid(row = 0, column = 0, columnspan = 5, sticky = W + E + S, padx = 5, pady = 10)
					def click_warning_OK():
						time.sleep(0.1)
						top.destroy()
					warning_button = Button(top, text='OK', command = click_warning_OK, foreground = "black", font=('Times', colonyQC_TextSizeButton))
					warning_button.grid(row = 1, column = 2, padx = 1,pady = 1, sticky = W+E+N)
				else:
					toPick = min(varPool_1.remaining_colonies, varPool_1.num_of_pick - len(tmpPick))
					varPool_1.remaining_colonies -= toPick
					varPool_1.bad_colonies += [varPool_1.finalPick[i] for i in range(len(varPool_1.finalPick)) if varPool_1.colony_button_flag[i] == 1]
					if toPick != 0:
						pick_list = reSelectColony(toPick, tmpPick, varPool_1.bad_colonies, post_finalDF_PCA)
						varPool_1.finalPick = tmpPick + pick_list
						varPool_1.finalPick.sort()
						varPool_1.pick_finalContours = [post_finalContours[i] for i in varPool_1.finalPick]
						varPool_1.pick_finalDF = post_finalDF.iloc[varPool_1.finalPick]
						varPool_1.pick_finalDF.index = range(len(varPool_1.finalPick))
						varPool_1.image_picked_contours = drawContour(image_trans_crop, varPool_1.pick_finalContours, colonyQC_colonyContourPixel)
						varPool_1.image_picked_contours_label = drawContourLabel(varPool_1.image_picked_contours, varPool_1.pick_finalDF, range(1, len(varPool_1.finalPick) + 1), \
																				colonyQC_colonyLabelSize, colonyQC_colonyLabelThickness)
						cv2image = varPool_1.image_picked_contours_label
						resize_image = cv2.resize(cv2image, (int(width_crop_merge * resize_factor * 18 / 19), int(height_crop_merge * resize_factor * 18 / 19)))
						varPool_1.imgtk = image_to_tk(resize_image)
						varPool_1.pick_colony_tk = []
						varPool_1.pick_colony_bad_tk = []
						for i in range(len(varPool_1.finalPick)):
							tmpSub = generateContourSubImage_QC(image_trans_crop, varPool_1.pick_finalContours[i], (varPool_1.pick_finalDF["X"][i], varPool_1.pick_finalDF["Y_concat"][i]), \
										colony_show_bias, colony_show_size, 1, str(i + 1), colony_show_size / 200.0, int(colony_show_size / 100.0))
							varPool_1.pick_colony_bad_tk.append(image_to_tk(addRedXtoImage(tmpSub, 2)))
							varPool_1.pick_colony_tk.append(image_to_tk(tmpSub))
						varPool_1.colony_button_flag = []
						for i in range(len(varPool_1.finalPick)):
							varPool_1.colony_button_flag.append(0)
						but_text1.grid_forget()
						but_text2.grid_forget()
						but_back.grid_forget()
						but_re.grid_forget()
						but_finish.grid_forget()
						frm.grid_forget()
						show_ind_image(varPool_1)
					elif varPool_1.remaining_colonies == 0:
						time.sleep(0.1)
						top = Toplevel()
						top.title('Warning')
						top.geometry("x".join([str(confirm_size[0] + 15), str(confirm_size[1])]))
						warning_test = Label(top, text="There are no more colonies for re-pick", font=('Times', colonyQC_TextSizeMid))
						warning_test.grid(row = 0, column = 0, columnspan = 5, sticky = W + E + S, padx = 10, pady = 10)
						def click_warning_OK():
							time.sleep(0.1)
							top.destroy()
						warning_button = Button(top, text='OK', command = click_warning_OK, foreground = "black", font=('Times', colonyQC_TextSizeButton))
						warning_button.grid(row = 1, column = 2, padx = 1,pady = 1, sticky = W+E+N)
			click_reselect_with_arg = partial(click_reselect, varPool)
			def click_finish(varPool_1):
				time.sleep(0.1)
				top = Toplevel()
				top.title('Confirm')
				top.geometry("x".join([str(confirm_size[0]), str(confirm_size[1])]))
				confirm_test = Label(top, text="Are you sure to pick these colonies?", font=('Times', colonyQC_TextSizeMid))
				confirm_test.grid(row = 0, column = 0, columnspan = 5, sticky = W + E + S, padx = 10, pady = 10)
				def click_confirm_Yes(varPool_2):
					time.sleep(0.1)
					top.destroy()
					root.destroy()
					varPool_2.outputPick = [varPool_2.finalPick[i] for i in range(len(varPool_2.finalPick)) if varPool_2.colony_button_flag[i] == 0]
					varPool_2.breakFlag = 1
					varPool_2.bad_colonies += [varPool_2.finalPick[i] for i in range(len(varPool_2.finalPick)) if varPool_2.colony_button_flag[i] == 1]
				click_confirm_Yes_with_arg = partial(click_confirm_Yes, varPool_1)
				def click_confirm_No():
					time.sleep(0.1)
					top.destroy()
				confirm_button = Button(top, text='No', command = click_confirm_No, foreground = "red", font=('Times', colonyQC_TextSizeButton))
				confirm_button.grid(row = 1, column = 1, padx = 1,pady = 1, sticky = W+E+N)
				confirm_button = Button(top, text='Yes', command = click_confirm_Yes_with_arg, foreground = "green", font=('Times', colonyQC_TextSizeButton))
				confirm_button.grid(row = 1, column = 3, padx = 1,pady = 1, sticky = W+E+N)
			click_finish_with_arg = partial(click_finish, varPool)
			but_back = Button(root, text="View plate", command = click_back_with_arg, foreground = "black", font=('Times', colonyQC_TextSizeButton))
			but_re = Button(root, text="Re-pick", command = click_reselect_with_arg, foreground = "red", font=('Times', colonyQC_TextSizeButton))
			but_finish = Button(root, text="Finish", command = click_finish_with_arg, foreground = "green", font=('Times', colonyQC_TextSizeButton))
			but_back.grid(row = 3, column = 4, columnspan = 3, sticky = W+E+N, padx = 5, pady = 5)
			but_re.grid(row = 3, column = 8, columnspan = 3, sticky = W+E+N, padx = 5, pady = 5)
			but_finish.grid(row = 3, column = 12, columnspan = 3, sticky = W+E+N, padx = 5, pady = 5)
		show_whole_image(globalVar)
		root.mainloop()
		if max_index == 10:
			return globalVar.finalPick, globalVar.bad_colonies
		elif globalVar.breakFlag == 1:
			return globalVar.outputPick, globalVar.bad_colonies
		else:
			max_index += 1
def readFileList(input_dir):
	image_processed = []
	if os.path.isfile(input_dir + "/image_processed.txt"):
		f = open(input_dir + "/image_processed.txt","rb")
		data = f.readlines()
		f.close()
		for e in data:
			image_processed.append(e[:-1])
	image_all_list = os.listdir(input_dir)
	image_first_filter = []
	for e in image_all_list:
		if e[-4:] == ".bmp":
			image_first_filter.append(e[:-4].split("_"))
	image_second = {}
	for e in image_first_filter:
		if e[0] not in image_processed:
			image_second[e[0]] = []
	for e in image_first_filter:
		if e[0] not in image_processed:
			image_second[e[0]].append(int(e[1]))
	total_image = 0 
	image_label_list = []
	image_trans_list = []
	image_epi_list = []
	for e in image_second.keys():
		total_image += 1
		image_label_list.append(e)
		tmpList = image_second[e]
		tmpList.sort()
		image_trans_list.append(input_dir + "/" + e + "_" + str(tmpList[0]) + ".bmp")
		image_epi_list.append(input_dir + "/" + e + "_" + str(tmpList[1]) + ".bmp")
	return total_image, image_label_list, image_trans_list, image_epi_list

def initializePlateInfo(globalObj):
	totalImage = len(globalObj.image_label)
	for i in range(totalImage):
		globalObj.final_pick[i] = []
		globalObj.bad_pick[i] = []
		tmpMetadata = globalObj.all_metadata[i]
		tmpMetadata["Y_concat"] = [0, ] * tmpMetadata.shape[0]
		tmpMetadata["pickStatus"] = ["not_pick", ] * tmpMetadata.shape[0]
		tmpMetadata["pickIndexGroup"] = ["NA", ] * tmpMetadata.shape[0]
		tmpMetadata["pickIndexPlate"] = ["NA", ] * tmpMetadata.shape[0]
		globalObj.all_metadata[i] = tmpMetadata

def findGroupIDindex(globalObj, groupID):
	totalImage = len(globalObj.image_label)
	output_list = []
	for i in range(totalImage):
		if globalObj.groupID[i] == groupID and globalObj.plateQC_flag[i] == True:
			output_list.append(i)
	return output_list

def plateLabelIndex_Pool(globalObj):
	totalImage = len(globalObj.image_label)
	output_pool = {}
	for i in range(totalImage):
		output_pool[globalObj.image_label[i]] = i
	return output_pool

def getNumPickColonies(sample_config, groupID):
	tmpList1 = list(sample_config["groupID"])
	tmpList2 = list(sample_config["numGroupPick"])
	tmpPool = {}
	for i in range(len(tmpList1)):
		tmpPool[tmpList1[i]] = tmpList2[i]
	return tmpPool[groupID]

def modifyOSconfigure(configurePool):
	if sys.platform == "darwin":
		configurePool["plateQC_colonyContourPixel"] = configurePool["plateQC_macOS_colonyContourPixel"]
		configurePool["plateQC_imageScaleFactor"] = configurePool["plateQC_macOS_imageScaleFactor"]
		configurePool["plateQC_imageWidthBias"] = configurePool["plateQC_macOS_imageWidthBias"]
		configurePool["plateQC_imageHeightBias"] = configurePool["plateQC_macOS_imageHeightBias"]
		configurePool["plateQC_ComfirmWindow"] = configurePool["plateQC_macOS_ComfirmWindow"]
		configurePool["plateQC_TextSizeLarge"] = configurePool["plateQC_macOS_TextSizeLarge"]
		configurePool["plateQC_TextSizeSmall"] = configurePool["plateQC_macOS_TextSizeSmall"]
		configurePool["plateQC_TextSizeButton"] = configurePool["plateQC_macOS_TextSizeButton"]
		configurePool["colonyQC_imageScaleFactor"] = configurePool["colonyQC_macOS_imageScaleFactor"]
		configurePool["colonyQC_imageWidthBias"] = configurePool["colonyQC_macOS_imageWidthBias"]
		configurePool["colonyQC_imageHeightBias"] = configurePool["colonyQC_macOS_imageHeightBias"]
		configurePool["colonyQC_ComfirmWindow"] = configurePool["colonyQC_macOS_ComfirmWindow"]
		configurePool["colonyQC_colonyShowBias"] = configurePool["colonyQC_macOS_colonyShowBias"]
		configurePool["colonyQC_colonyWindowBias"] = configurePool["colonyQC_macOS_colonyWindowBias"]
		configurePool["colonyQC_colonyContourPixel"] = configurePool["colonyQC_macOS_colonyContourPixel"]
		configurePool["colonyQC_colonyLabelSize"] = configurePool["colonyQC_macOS_colonyLabelSize"]
		configurePool["colonyQC_colonyLabelThickness"] = configurePool["colonyQC_macOS_colonyLabelThickness"]
		configurePool["colonyQC_TextSizeLarge"] = configurePool["colonyQC_macOS_TextSizeLarge"]
		configurePool["colonyQC_TextSizeMid"] = configurePool["colonyQC_macOS_TextSizeMid"]
		configurePool["colonyQC_TextSizeSmall"] = configurePool["colonyQC_macOS_TextSizeSmall"]
		configurePool["colonyQC_TextSizeButton"] = configurePool["colonyQC_macOS_TextSizeButton"]
	else:
		configurePool["plateQC_colonyContourPixel"] = configurePool["plateQC_MSwin10_colonyContourPixel"]
		configurePool["plateQC_imageScaleFactor"] = configurePool["plateQC_MSwin10_imageScaleFactor"]
		configurePool["plateQC_imageWidthBias"] = configurePool["plateQC_MSwin10_imageWidthBias"]
		configurePool["plateQC_imageHeightBias"] = configurePool["plateQC_MSwin10_imageHeightBias"]
		configurePool["plateQC_ComfirmWindow"] = configurePool["plateQC_MSwin10_ComfirmWindow"]
		configurePool["plateQC_TextSizeLarge"] = configurePool["plateQC_MSwin10_TextSizeLarge"]
		configurePool["plateQC_TextSizeSmall"] = configurePool["plateQC_MSwin10_TextSizeSmall"]
		configurePool["plateQC_TextSizeButton"] = configurePool["plateQC_MSwin10_TextSizeButton"]
		configurePool["colonyQC_imageScaleFactor"] = configurePool["colonyQC_MSwin10_imageScaleFactor"]
		configurePool["colonyQC_imageWidthBias"] = configurePool["colonyQC_MSwin10_imageWidthBias"]
		configurePool["colonyQC_imageHeightBias"] = configurePool["colonyQC_MSwin10_imageHeightBias"]
		configurePool["colonyQC_ComfirmWindow"] = configurePool["colonyQC_MSwin10_ComfirmWindow"]
		configurePool["colonyQC_colonyShowBias"] = configurePool["colonyQC_MSwin10_colonyShowBias"]
		configurePool["colonyQC_colonyWindowBias"] = configurePool["colonyQC_MSwin10_colonyWindowBias"]
		configurePool["colonyQC_colonyContourPixel"] = configurePool["colonyQC_MSwin10_colonyContourPixel"]
		configurePool["colonyQC_colonyLabelSize"] = configurePool["colonyQC_MSwin10_colonyLabelSize"]
		configurePool["colonyQC_colonyLabelThickness"] = configurePool["colonyQC_MSwin10_colonyLabelThickness"]
		configurePool["colonyQC_TextSizeLarge"] = configurePool["colonyQC_MSwin10_TextSizeLarge"]
		configurePool["colonyQC_TextSizeMid"] = configurePool["colonyQC_MSwin10_TextSizeMid"]
		configurePool["colonyQC_TextSizeSmall"] = configurePool["colonyQC_MSwin10_TextSizeSmall"]
		configurePool["colonyQC_TextSizeButton"] = configurePool["colonyQC_MSwin10_TextSizeButton"]

def drawOutputContoursWhole(image_trans, image_epi, all_contours, pick_index, metadataDF, pixel, fontScale, thickness, pinSize):
	image_trans_BGR = cv2.cvtColor(image_trans.astype(np.uint8), cv2.COLOR_GRAY2BGR)
	image_epi_BGR = image_epi.copy()
	for contour in all_contours:
		cv2.drawContours(image_trans_BGR,[contour],0,[0,0,0],pixel)
		cv2.drawContours(image_epi_BGR,[contour],0,[0,0,0],pixel)  
	drawLabel = 1
	for i in pick_index:
		contour = all_contours[i]
		tmpX = metadataDF["X"][i]
		tmpY = metadataDF["Y"][i]
		tmpLabel = drawLabel
		cv2.drawContours(image_trans_BGR,[contour],0,[0,0,255],pixel)
		cv2.drawContours(image_epi_BGR,[contour],0,[0,0,255],pixel)
		cv2.putText(image_trans_BGR, str(tmpLabel), (int(tmpX), int(tmpY)), \
					cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
		cv2.putText(image_epi_BGR, str(tmpLabel), (int(tmpX), int(tmpY)), \
					cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
		cv2.circle(image_trans_BGR, (tmpX, tmpY), pinSize, (0, 255, 0), -1)
		cv2.circle(image_epi_BGR, (tmpX, tmpY), pinSize, (0, 255, 0), -1)
		drawLabel += 1
	return image_trans_BGR, image_epi_BGR

def modifyMetadataSplit(metadata):
	tmpOutMeta = metadata.copy()
	tmpPickStatus = metadata["pickStatus"]
	indexV = 0
	tmpOut = []
	tmpPickIndex = []
	tmpBadIndex = []
	for i in range(len(tmpPickStatus)):
		each = tmpPickStatus[i]
		if each == "pick":
			tmpOut.append(indexV)
			indexV += 1
			tmpPickIndex.append(i)
		else:
			tmpOut.append("NA")
		if each == "bad_pick":
			tmpBadIndex.append(i)
	tmpOutMeta["pickIndexPlate"] = tmpOut
	return tmpOutMeta, tmpPickIndex, tmpBadIndex

def saveOutputs_pickingOptimization(globalOutput, total_image, configure_pool, output_dir, groupID_index):
	for i in range(total_image):
		if i not in groupID_index:
			continue
		tmp_image_label = globalOutput.image_label[i]
		tmp_image_trans = globalOutput.image_trans_corrected[i]
		tmp_image_epi = globalOutput.image_epi_corrected[i]
		tmp_all_contours = globalOutput.all_contours[i]
		tmp_pick_index = globalOutput.final_pick[i]
		tmp_image_label = globalOutput.image_label[i]
		tmp_metadataDF = globalOutput.all_metadata[i]
		tmpPickDF = globalOutput.all_metadata[i].iloc[globalOutput.final_pick[i]]
		tmpPickX = list(tmpPickDF["X_rawImage"])
		tmpPickY = list(tmpPickDF["Y_rawImage"])
		f = open(output_dir + "/" + tmp_image_label + "_Coordinates.csv","w")
		for j in range(len(tmpPickX)):
			f.writelines([str(tmpPickX[j]) + "," + str(tmpPickY[j]) + os.linesep])
		f.close()
		output_image_trans, output_image_epi = drawOutputContoursWhole(tmp_image_trans, tmp_image_epi, tmp_all_contours, tmp_pick_index, tmp_metadataDF, 2, 0.7, 2, 2)
		cv2.imwrite(output_dir + "/" + tmp_image_label + "_Image_colony_trans_pick.jpg", output_image_trans)
		cv2.imwrite(output_dir + "/" + tmp_image_label + "_Image_colony_epi_pick.jpg", output_image_epi)
		output_image_trans_all = drawContour(tmp_image_trans, tmp_all_contours, 2)
		output_image_trans_all_pin = drawPinSite(output_image_trans_all, tmp_all_contours, 2)
		cv2.imwrite(output_dir + "/" + tmp_image_label + "_Image_colony_trans.jpg", output_image_trans_all_pin)
		tmp_metadataDF.to_csv(output_dir + "/" + tmp_image_label + "_Metadata_all.csv")
		np.save(output_dir + "/" + tmp_image_label + "_Contours_all.npy", tmp_all_contours)
	timeStamp = time.time()
	timeValue = datetime.datetime.fromtimestamp(timeStamp)
	timeFormat = timeValue.strftime('%Y%m%d_%H%M%S_%f')
	f = open(output_dir + "/pickingOptimization." + timeFormat + ".merge.obj", 'w') 
	pickle.dump(globalOutput, f)
	f.close()

def modifyOutputObject_colonyDetection(globalOutput, total_image, configure_pool):
	for i in range(total_image):
		tmp_metadataDF = globalOutput.all_metadata[i]
		tmp_all_contours = globalOutput.all_contours[i]
		tmp_metadataDF["X_rawImage"] = tmp_metadataDF["X"] + configure_pool["cropXMin"]
		tmp_metadataDF["Y_rawImage"] = tmp_metadataDF["Y"] + configure_pool["cropYMin"]
		tmp_metadataDF["plate_barcode"] = [globalOutput.image_label[i], ] * len(tmp_all_contours)
		tmp_metadataDF["colony_index"] = range(len(tmp_all_contours))
		globalOutput.all_metadata[i] = tmp_metadataDF

def saveOutputs_colonyDetection(globalOutput, total_image, configure_pool, output_dir):
	for i in range(total_image):
		tmp_image_label = globalOutput.image_label[i]
		tmp_image_trans = globalOutput.image_trans_corrected[i]
		tmp_image_epi = globalOutput.image_epi_corrected[i]
		tmp_all_contours = globalOutput.all_contours[i]
		tmp_image_label = globalOutput.image_label[i]
		tmp_metadataDF = globalOutput.all_metadata[i]
		output_image_trans_all = drawContour(tmp_image_trans, tmp_all_contours, 2)
		output_image_trans_all_pin = drawPinSite(output_image_trans_all, tmp_all_contours, 2)
		cv2.imwrite(output_dir + "/" + tmp_image_label + "_Image_colony_trans.jpg", output_image_trans_all_pin)
		tmp_metadataDF.to_csv(output_dir + "/" + tmp_image_label + "_Metadata_all.csv")
		np.save(output_dir + "/" + tmp_image_label + "_Contours_all.npy", tmp_all_contours)
	timeStamp = time.time()
	timeValue = datetime.datetime.fromtimestamp(timeStamp)
	timeFormat = timeValue.strftime('%Y%m%d_%H%M%S_%f')
	f = open(output_dir + "/colonyDetection." + timeFormat + ".merge.obj", 'w') 
	pickle.dump(globalOutput, f)
	f.close()

def savePCAdata(globalOutput, configure_pool, sample_config_path, pick_dir, output_path):
	sample_config = pd.read_csv(sample_config_path)
	image_labelIndex_pool = plateLabelIndex_Pool(globalOutput)
	for i in range(sample_config.shape[0]):
		globalOutput.groupID[image_labelIndex_pool[list(sample_config["barcode"])[i]]] = list(sample_config["groupID"])[i]
	unique_groupID = list(set(list(sample_config["groupID"])))
	unique_groupID.sort()
	for eachGroupID in unique_groupID:
		spacing = configure_pool["colonyQC_image_spacing"]
		fontSize = configure_pool["colonyQC_image_labelSize"]
		fontThickness = configure_pool["colonyQC_image_thickness"]
		groupID_index = findGroupIDindex(globalOutput, eachGroupID)
		groupID_label_list = [globalOutput.image_label[i] for i in groupID_index]
		groupID_image_trans_list = [globalOutput.image_trans_corrected[i] for i in groupID_index]
		groupID_image_merge, groupID_image_heightStart = concatenateImages_gray(groupID_image_trans_list, groupID_label_list, spacing, fontSize, fontThickness)
		groupID_metadata_list = [globalOutput.all_metadata[i] for i in groupID_index]
		groupID_metadata_merge = concat_metadata(groupID_metadata_list, groupID_image_heightStart)
		feats = groupID_metadata_merge[['Area', 'Perimeter', 'Radius', 'Circularity', 'Convexity', 'Inertia', \
										'Graymean', 'Graystd', 'Repimean', 'Repistd', \
										'Gepimean', 'Gepistd', 'Bepimean', 'Bepistd']]
		feats_list = feats.values.tolist()
		preprocessed_plates = transform_data_PCA(feats_list, 2)
		csvList = []
		for eachLabel in groupID_label_list:
			tmpCSV = pd.read_csv(pick_dir + "/" + eachLabel + "_Metadata_all.csv")
			csvList.append(tmpCSV)
		pick_csv = pd.concat(csvList)
		newDF_dist = {"pickStatus":list(pick_csv["pickStatus"]), "plateBarcode":list(pick_csv["plate_barcode"]), "PCA1":list(preprocessed_plates[:,0]), "PCA2":list(preprocessed_plates[:,1])}
		newDF = pd.DataFrame(newDF_dist)
		newDF.to_csv(output_path)





