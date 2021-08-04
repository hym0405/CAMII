# Scalable and cost-effective ribonuclease-based rRNA depletion for bacterial transcriptomics

Tools for probe design and evaluation (Jupyter notebooks can be found in ./jupyter_notebook):

We have tested these scripts on Linux and MacOS

* **0.design_probe.py**: design probe libraries for target 16S and 23S rRNA sequence

* **1.calculate_probe_identity.py**: calculate probe identity to various different 16S and 23S sequences to evaluate the ability of pools to be applied to different sequences

* **2.predict_probe_offtarget.py**: predict potential off-targets for probe libraries

## Dependencies

* Python 2.7, Jupyter 4.3.0 
	- panda
	- numpy
	- argparse
	- _NB: Above libraries are bundled together in the [Anaconda distribution](https://www.anaconda.com/distribution)_


* [Muscle: MUltiple Sequence Comparison by Log-Expectation](https://www.drive5.com/muscle/)
	- **Required for probe identity calculation only**
	- Executable file of Muscle that compatible with your operating system should be put into ./bin or other place specified in 1.calculate_probe_identity.ipynb
	
* [NCBI BLAST+ Executables 2.9.0](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download)
	- **Required for probe off-targets prediction only**
	- Executable file of makeblastdb and blastn that compatible with your operating system should be put into ./bin or other place specified in 2.predict_probe_offtarget.sh
	
* [BURST v0.99.8 DB15](https://github.com/knights-lab/BURST)
	- **Required for probe off-targets prediction only**
	- Executable file of burst that compatible with your operating system should be put into ./bin or other place specified in 2.predict_probe_offtarget.sh


## Design probe libraries for 16S and 23S rRNA sequences

### Description
```
usage: 0.design_probe.py [-h] [-i INPUT] [-o OUTPUT] [-l LENGTH]

Design probe libraries for bacterial rRNA depletion

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to rRNA sequences. All rRNA sequences should be
                        labelled as [SampleID]_16S and [SampleID]_23S in FASTA
                        format
  -o OUTPUT, --output OUTPUT
                        Path to output probe file. Probe sequences will be
                        saved as a tab-delimited table
  -l LENGTH, --length LENGTH
                        Length of probes [default: 50]
```
### Input format
****[Important] Avoid underline in sample IDs****

**rRNA sequence:** 16S and 23S rRNA sequence in FASTA format and all rRNA sequences should be labelled as [SampleID]_16S and [SampleID]_23S

****In our paper, [Prokka](https://github.com/tseemann/prokka) is used to predict 16S and 23S rRNA sequences of bacterial species****

****[example: ./data/rRNA_sequence/rRNA_sequence.dorei.fa]****

```
>dorei_16S
AGAGTTTGATCCTGGCTC...
...
>dorei_23S
GAAAGTAAAGAAGGGCGC...
...
```
### Output format
****Probe sequences will be saved as a tab-delimited table****

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

### Example
```
chmod +x ./0.design_probe.py
python2 ./0.design_probe.py -i ./data/rRNA_sequence/rRNA_sequence.dorei.fa \
		-o ./output/rRNA_probe.dorei.tsv \
		-l 50
```


## Calculate probe identity to new rRNA sequences to evaluate the ability of pools to be applied

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
