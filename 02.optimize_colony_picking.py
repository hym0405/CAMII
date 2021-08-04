#!/usr/bin/env python
import imaging_picking_function as ipf
import time
import os
import sys
import pickle
import threading
import pandas as pd
import argparse
def main():

	parser = argparse.ArgumentParser(description = "Program of morphology-guided colony selection for isolation. See details in https://github.com/hym0405/CAMII")

	parser.add_argument("-c", "--config", type = str,
                    help="Configure file of parameters for general colony segmentation and filtering")

	parser.add_argument("-m", "--metadata", type = str,
                    help="Metadata of plates for optimized colony isolation")

	parser.add_argument("-i", "--input", type = str, 
                    help="colonyDetection.*.merge.obj file in the output folder of \"01.colony_detection.py\"")

	parser.add_argument("-o", "--output", type = str,
			help="Output folder. The folder will be created if not exists")

	args = parser.parse_args()
	configure_path = args.config
	sample_config_path = args.metadata
	python_obj_path = args.input
	output_dir = args.output

	configure_pool = ipf.readConfigureFile(configure_path)
	ipf.modifyOSconfigure(configure_pool)
	globalOutput = pickle.load(open(python_obj_path, "rb" ))
	sample_config = pd.read_csv(sample_config_path)
	
	ipf.initializePlateInfo(globalOutput)
	image_labelIndex_pool = ipf.plateLabelIndex_Pool(globalOutput)
	for i in range(sample_config.shape[0]):
		globalOutput.groupID[image_labelIndex_pool[list(sample_config["barcode"])[i]]] = list(sample_config["groupID"])[i]
	unique_groupID = list(set(list(sample_config["groupID"])))
	unique_groupID.sort()	

	for eachGroupID in unique_groupID:
		groupID_index = ipf.findGroupIDindex(globalOutput, eachGroupID)
		ipf.fun3_runColonyQualityControl_group(eachGroupID, groupID_index, globalOutput, configure_pool, sample_config)

	if not os.path.isdir(output_dir):
		os.system("mkdir -p " + output_dir)

	for eachGroupID in unique_groupID:
		groupID_index = ipf.findGroupIDindex(globalOutput, eachGroupID)
		ipf.saveOutputs_pickingOptimization(globalOutput, len(globalOutput.image_label), configure_pool, output_dir, groupID_index)

	ipf.savePCAdata(globalOutput, configure_pool, sample_config_path, output_dir, output_dir + "/PCAdata.csv")
	
	os.system("Rscript " + configure_pool["house_bin"] + "/visualizePCAandPickStatus.R " + output_dir + "/PCAdata.csv " + output_dir)

if __name__ == "__main__":
	main()
