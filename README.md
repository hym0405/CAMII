# Automated and Machine Learning Guided Culturomics of Personalized Gut Microbiomes

Tools for plate image analysis and morphology-guided colony selection (Examples can be found in ./example)

We have tested these scripts on Linux and MacOS.

* **01.colony_detection.py**: Script of raw image analysis, including colony contour detection, segmentation and features extraction.

* **02.optimize_colony_picking.py**: Script of morphology-guided colony selection and manual inspection for optimized strain isolation.

## Dependencies

* Python 2.7:
	- panda, numpy, scipy, sklearn, skimage, datetime, functools, pickle (_These libraries are bundled together in the [Anaconda distribution](https://www.anaconda.com/distribution)_)
	- Tkinter: Tk GUI toolkit
	- cv2: opencv library
	- PIL: python image library

* R 4.1.0:
 	- **Required for colony ordination visualization only**
 	- ggplot2
 	- reshape

## Colony contour detection, segmentation and features extraction from raw plate images

### Description
```
usage: 01.colony_detection.py [-h] [-c CONFIG] [-i INPUT] [-o OUTPUT]

Program of colony contour detection and segmentation.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configure file of parameters for general colony
                        segmentation and filtering
  -i INPUT, --input INPUT
                        Input folder containing raw images of plates, both
                        trans-illuminated and epi-illuminated. A file named
                        "image_processed.txt" will be written to the folder
                        after processing
  -o OUTPUT, --output OUTPUT
                        Output folder. The folder will be created if not
                        exists
```
### Input format
****[Important]**** Avoid underline in sample name or plate barcode


**Input folder:** trans-illuminated images and epi-illuminated images in BMP format. Files are named as \[plate\_barcode\]\_\[imaging_time\].bmp and trans-illuminated images were imaged first so \[imaging_time\] is earlier than epi-illuminated images. 

****[example: ./example/raw_plate_images]****


**Configure file:** files containing all parameters used in colony contour detection and segmentation. 

****[example: ./configure]****

```
# system setup
house_bin=./bin
parameters_dir=./parameters

# image calibration
calib_gaussian_kernal=(27,27)
calib_gaussian_iteration=20
calib_parameter_PATH=./parameters/calib_parameter.npz
calib_contrast_trans_alpha=5
calib_contrast_trans_beta=-100
calib_contrast_trans_beta=-70

# image crop
cropYMin=150
cropYMax=1150
cropXMin=150
cropXMax=1750
...
```

### Output format
****Contours, metadata and visualization of all detected colonies will be saved in output folder:****

* **\[plate\_barcode\]\_Contours\_all.npy:** Contours of colonies on plate saved in Python pickle object.
* **\[plate\_barcode\]\_Metadata\_all.csv:** Metadata of colonies on plate in CSV format, including coordinates and morphological features.
* **\[plate\_barcode\]\_Image\_colony_trans.jpg:** Image of plate with all colonies labeled in JPEG format.
* **colonyDetection.\[processing_time\].merge.obj:** Python object containing all related images and colonies data. This file will be used as input for downstream optimized colony selection.

****[Important]**** A file named image_processed.txt will be written to input folder to indicate images in this folder have been processed. Please delete the file if you want to rerun the colony detection for images in that folder

****[example: ./example/output_colony_detection]****

**\[plate\_barcode\]\_Metadata\_all.csv:**
```
,X,Y,Area,Perimeter,Radius,Circularity,Convexity,...
0,373,940,139.0,46.142,7.245,0.820,0.958,...
1,1490,936,119.5,42.727,6.082,0.822,0.959,...
2,1027,929,52.0,27.656,4.472,0.854,0.981,...
...
```

### Manual inspection
****After image processing, you could check detected colonies on each plates and determine whether you want to keep specific plates for colony picking in following GUI:****

<p align="center">
  <img src="https://github.com/hym0405/CAMII/blob/main/misc/check_colony_detection.png" width="500" title="hover text">
</p>

### Example of usage
```
python2 ./01.colony_detection.py -c ./configure \
		-i ./example/raw_plate_images \
		-o ./example/output_colony_detection
```







## Morphology-guided colony selection for optimized strain isolation

### Description
```
usage: 02.optimize_colony_picking.py [-h] [-c CONFIG] [-m METADATA] [-i INPUT]
                                     [-o OUTPUT]

Program of morphology-guided colony selection for isolation.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configure file of parameters for general colony
                        segmentation and filtering
  -m METADATA, --metadata METADATA
                        Metadata of plates for optimized colony isolation
  -i INPUT, --input INPUT
                        colonyDetection.*.merge.obj file in the output folder
                        of "01.colony_detection.py"
  -o OUTPUT, --output OUTPUT
                        Output folder. The folder will be created if not
                        exists
```

### Input format
****[Important]**** Avoid underline in sample name or plate barcode

**Configure file:** files containing all parameters used in colony contour detection and segmentation. 

****[example: ./configure]****

```
# system setup
house_bin=./bin
parameters_dir=./parameters

# image calibration
calib_gaussian_kernal=(27,27)
calib_gaussian_iteration=20
calib_parameter_PATH=./parameters/calib_parameter.npz
calib_contrast_trans_alpha=5
calib_contrast_trans_beta=-100
calib_contrast_trans_beta=-70

# image crop
cropYMin=150
cropYMax=1150
cropXMin=150
cropXMax=1750
...
```

**Metadata of plates for optimized colony selection:** information of plates, including plate barcodes, sample names and # of colonies to pick for each set of plates

****[example: ./example/sample_metadata.csv]****: This file indicates colonies on plate H2M1, H2M2, H2M3 will be considered together (as a group) for optimized selection and 32 colonies will be selected by the algorithm.

```
barcode,sampleID,plateTyte,plateID,groupID,numGroupPick
H2M1,H2,mGAM,H2,H2mGAM20200126,32
H2M2,H2,mGAM,H2,H2mGAM20200126,32
H2M3,H2,mGAM,H2,H2mGAM20200126,32
```

**Input colony detection file:** python object file generated by 01.colony_detection.py: ****colonyDetection.\[processing_time\].merge.obj****


### Output format

****Metadata, coordinates and visualization of colonies to pick will be saved in output folder:****

* **\[plate\_barcode\]\_Contours\_all.npy:** Contours of all colonies on plate saved in Python pickle object.
* **\[plate\_barcode\]\_Metadata\_all.csv:** Metadata of all colonies on plate in CSV format, including coordinates and morphological features. Colonies will be labeled as "pick", "not_pick" and "bad_pick".
* **\[plate\_barcode\]\_Image\_colony_trans.jpg:** Image of plate with all colonies labeled in JPEG format.
* **\[plate\_barcode\]\_Image\_colony_\[trans or epi\]\_pick.jpg:** Trans- or Ep-illuminated images of plate with colonies to pick highlighted in JPEG format.
* **PCAdata.csv:** PCA ordination of colonies based on their morphological features
* **PCAdata.pickStatus.pdf:** visualization of colonies picking in PCA ordination
* **PCAdata.plateBarcode.pdf:** visualization of colonies on different plates in PCA ordination
* **pickingOptimization.\[processing_time\].merge.obj:** Python object containing all related images and colonies data.

****[example: ./example/output_optimized_picking]****


### Example
```
python2 02.optimize_colony_picking.py -c configure \
		-m ./example/sample_metadata.csv \
		-i ./example/output_colony_detection/colonyDetection.20210809_112425_368541.merge.obj \
		-o output_optimized_picking
```

### Manual inspection
****After image processing, you could check selected colonies on each plates and determine whether these colonies are good enough for picking in following GUI:****

<p align="center">
  <img src="https://github.com/hym0405/CAMII/blob/main/misc/check_optimize_colony_picking1.png" width="427" title="hover text">
</p>

You could check selected colonies on plates or individually and also remove colonies that are (1) artifacts or (2) failed to be segemented by selecting them and click Re-pick

<p align="center">
 <img src="https://github.com/hym0405/CAMII/blob/main/misc/check_optimize_colony_picking2.png" width="427" title="hover text">
</p>



## Reference

Huang, Y., Sheth U. R., Zhao, S., Cohen, L., Dabaghi, K., Moody, T., Sun Y., Ricaurte D., Richardson M., Velez-Cortes F., Blazejewski T., Kaufman A., Ronda C., and Wang H. H., High-throughput microbial culturomics using automation and machine learning. GitHub. https://github.com/hym0405/CAMII (2023).
