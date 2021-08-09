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

****[Important]**** A file named image_processed.txt will be written to input folder to indicate images in this folder have been processed. Please delete the file if you want to rerun the colony detection for images in input folder

****[example: ./example/output_colony_detection]****

**\[plate\_barcode\]\_Metadata\_all.csv:**
```
,X,Y,Area,Perimeter,Radius,Circularity,Convexity,...
0,373,940,139.0,46.142,7.245,0.820,0.958,...
1,1490,936,119.5,42.727,6.082,0.822,0.959,...
2,1027,929,52.0,27.656,4.472,0.854,0.981,...
...
```

### Example of usage
```
python2 ./01.colony_detection.py -c ./configure \
		-i ./example/raw_plate_images \
		-o ./example/output_colony_detection
```


## Morphology-guided colony selection for optimized strain isolation

### Description
```
usage: 1.calculate_probe_identity.py [-h] [-t TARGET] [-p PROBE]
                                     [-o OUTPUT_PREFIX] [-m MUSCLE_PATH]

Calculate probe identity to new rRNA sequences to evaluate the ability of
pools to be applied to different sequences

optional arguments:
  -h, --help            show this help message and exit
  -t TARGET, --target TARGET
                        Path to target rRNA sequences. All rRNA sequences
                        should be labelled as [SampleID]_16S and
                        [SampleID]_23S in FASTA format
  -p PROBE, --probe PROBE
                        Path to probe sequences to be evaluated
  -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX
                        Prefix of output probe identity file. Results of probe
                        identity for different rRNA sequences will be saved as
                        individual files labelled as
                        [output_prefix].[rRNA_Label].tsv
  -m MUSCLE_PATH, --muscle_path MUSCLE_PATH
                        Path to executable file of muscle [default:
                        ./bin/muscle]
```

### Input format
****[Important] Avoid underline in sample IDs****

**target rRNA sequence:** 16S and 23S rRNA sequence in FASTA format and all rRNA sequences should be labelled as [SampleID]_16S and [SampleID]_23S

****[example: ./data/rRNA_sequence/rRNA_sequence.dorei.fa]****

```
>dorei_16S
AGAGTTTGATCCTGGCTC...
...
>dorei_23S
GAAAGTAAAGAAGGGCGC...
...
```

**probe sequences to be evaluated:** tab-delimited table and the format is exactly same as the output of 0.design_probe.py

****[example: ./output/rRNA_probe.dorei.tsv]****
```
rRNA_label      probe_ID        probe_sequence
dorei_16S       dorei_16S_0     AGGTGTTCCAGCCGC...
dorei_16S       dorei_16S_1     GTTTTACCCTAGGGC...
dorei_16S       dorei_16S_2     TCCCATGGCTTGACG...
...		...		...
dorei_23S       dorei_23S_0     TAAGGAAAGTGGACG...
dorei_23S       dorei_23S_1     CAACGTCGTAGTCTA...
dorei_23S       dorei_23S_2     TCGTACTTAGATGCT...
...
```

### Output format
****Results of probe identity for different rRNA sequences will be saved as individual files labelled as [output_prefix].[rRNA_Label].tsv****

****[example: ./output/probeIdentity.probe_dorei.uniformis_16S.tsv]****
```
## Target rRNA:uniformis_16S
## Probe set designed for: dorei_16S
## Total length of target rRNA uniformis_16S: 1515
## Total length of probe-target alignment: 1520
## Number of mismatches in probe-target alignment: 129
#target_ID	target_start	target_end	probe_ID	length_alignment	num_of_mismatches	ratio
uniformis_16S	1	60	dorei_16S_29	60	0	0.0 
uniformis_16S	61	110	dorei_16S_28	50	4	0.08
uniformis_16S	111	160	dorei_16S_27	50	10	0.2 
...
```

### Example
```
chmod +x ./1.calculate_probe_identity.py
python2 ./1.calculate_probe_identity.py -t ./data/rRNA_sequence/rRNA_sequence.uniformis.fa \
				-p ./output/rRNA_probe.dorei.tsv
				-o ./output/probeIdentity.probe_dorei \
				-m ./bin/muscle
```

## Predict potential off-targets for probe libraries
### Description
```
usage: 2.predict_probe_offtarget.py [-h] [-t TRANSCRIPT] [-r RRNA] [-p PROBE]
                                    [-pf {TSV,FASTA}] [-o OUTPUT_PREFIX]
                                    [-mb MAKEBLASTDB_PATH] [-bn BLASTN_PATH]
                                    [-br BURST_PATH]

Predict potential off-targets for probe libraries

optional arguments:
  -h, --help            show this help message and exit
  -t TRANSCRIPT, --transcript TRANSCRIPT
                        Path to transcript sequences. All transcript sequences
                        should be saved in FASTA format
  -r RRNA, --rRNA RRNA  Path to list of rRNA transcript IDs
  -p PROBE, --probe PROBE
                        Path to probe sequences to be evaluated. Probe
                        sequences can be saved in either TSV or FASTA format
                        (should be specified in probe format)
  -pf {TSV,FASTA}, --probe_format {TSV,FASTA}
                        Format of probe sequences, either TSV or FASTA
                        [default: TSV]
  -o OUTPUT_PREFIX, --output_prefix OUTPUT_PREFIX
                        Prefix of output predicted off-targets file. Results
                        of BLASTN and BURST will be saved in
                        [output_prefix].BLAST.tsv and
                        [output_prefix].BURST.tsv
  -mb MAKEBLASTDB_PATH, --makeblastdb_path MAKEBLASTDB_PATH
                        Path to executable file of makeblastdb (NCBI-BLAST)
                        [default: ./bin/makeblastdb]
  -bn BLASTN_PATH, --blastn_path BLASTN_PATH
                        Path to executable file of blastn (NCBI-BLAST)
                        [default: ./bin/blastn]
  -br BURST_PATH, --burst_path BURST_PATH
                        Path to executable file of burst [default:
                        ./bin/burst]
```
### Input format

**transcript sequences:** All transcript sequences should be saved in FASTA format

****[example: ./data/transcriptome_annotation/dorei.ffn]****

```
>GMBNIAIB_00001 Chromosomal replication initiator protein DnaA
ATGATTGAAAACGATCACGTCGTTTTATGGGGTCGTTGTCTGAACATTATCAGAGACAAC
GTACCTGAAACGACCTTTAAAACGTGGTTTGAGCCTATCGTACCGCTTAAATATGAGGAC
...
>GMBNIAIB_00002 FMN reductase [NAD(P)H]
ATGGAATCGATAAATAATAGACGGACGATCCGTAAATATAAGCAGGAAGATATTTCTGCT
TCTTTGTTAAATGATTTGCTTGAAAAGGCATTCCGTGCTTCTACAATGGGCAATATGCAA
...
>GMBNIAIB_00003 Vitamin B12-dependent ribonucleoside-diphosphate reductase
GTGGAAAAACAAACGTACACCTATGACGAAGCTTTTGAAGCATCTTTACAATACTTCAAA
GGTGATGAACTTGCTGCAAGGGTTTGGGTAAACAAATATGCAGTAAAAGATTCTTTCGGG
...
```

**list of rRNA transcript IDs:**

****[example: ./data/transcriptome_annotation/dorei.rRNA.list]****
```
GMBNIAIB_00241
GMBNIAIB_00242
GMBNIAIB_00245
...
```

**probe sequences to be evaluated:** Probe sequences can be provided in ****either TSV or FASTA format****

1. TSV format: tab-delimited table and the format is exactly same as the output of 0.design_probe.py
2. FASTA format: [example: ./output/rRNA_probe.dorei.fa]


### Output format

****Results of BLASTN and BURST will be saved in tab-delimited [output_prefix].BLAST.tsv and [output_prefix].BURST.tsv****

****[example: ./output/offtarget.longicatena.BURST.tsv]****

```
probeID			transcript(off-target)
longicatena_23S_11      JDJECPLG_03071
longicatena_23S_10      JDJECPLG_03071
longicatena_23S_12      JDJECPLG_03071
...
```

### Example
```
chmod +x ./2.predict_probe_offtarget.py
python2 ./2.predict_probe_offtarget.py -t ./data/transcriptome_annotation/longicatena.ffn \
				-r ./data/transcriptome_annotation/longicatena.rRNA.list \
				-p ./output/rRNA_probe.longicatena.tsv \
				-pf TSV \
				-o ./output/offtarget.longicatena \
				-mb ./bin/makeblastdb \
				-bn ./bin/blastn \
				-br ./bin/burst
```
