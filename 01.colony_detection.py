#!/usr/bin/env python
import imaging_picking_function as ipf
import time
import os
import sys
import threading
import argparse
def main():

	parser = argparse.ArgumentParser(description = "Program of colony contour detection and segmentation. See details in https://github.com/hym0405/CAMII")

	parser.add_argument("-c", "--config", type = str,
                    help="Configure file of parameters for general colony segmentation and filtering")

	parser.add_argument("-i", "--input", type = str, 
					help="Input folder containing raw images of plates, both trans-illuminated and epi-illuminated. A file named \"image_processed.txt\" will be written to the folder after processing")

	parser.add_argument("-o", "--output", type = str,
					help="Output folder. The folder will be created if not exists")

	args = parser.parse_args()

	configure_path = args.config
	input_dir = args.input
	output_dir = args.output


	configure_pool = ipf.readConfigureFile(configure_path)
	ipf.modifyOSconfigure(configure_pool)
	total_image, image_label_list, image_trans_list, image_epi_list = ipf.readFileList(input_dir)

	globalOutput = ipf.globalOutputObject(total_image)
	threadPool_fun0 = []
	for i in range(total_image):
		tmpThread = threading.Thread(target = ipf.multi_fun0_detectColonySingleImage, args = (image_trans_list[i], image_epi_list[i], image_label_list[i], configure_pool, globalOutput, i))
		threadPool_fun0.append(tmpThread)
	for i in range(total_image):
		threadPool_fun0[i].start()
	for i in range(total_image):
		threadPool_fun0[i].join()

	for i in range(total_image):
		tmpFlag = ipf.fun1_runPlateQualityControl(globalOutput.image_trans_corrected[i], globalOutput.image_epi_corrected[i], globalOutput.all_contours[i], 
																	globalOutput.all_metadata[i], globalOutput.image_label[i], configure_pool)
		globalOutput.plateQC_flag[i] = tmpFlag

	f = open(input_dir + "/image_processed.txt", "a")
	for e in image_label_list:
		f.writelines([e + os.linesep])
	f.close()

	ipf.modifyOutputObject_colonyDetection(globalOutput, total_image, configure_pool)

	if not os.path.isdir(output_dir):
		os.system("mkdir -p " + output_dir)

	ipf.saveOutputs_colonyDetection(globalOutput, total_image, configure_pool, output_dir)

if __name__ == "__main__":
	main()
