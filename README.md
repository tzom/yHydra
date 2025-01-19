# yHydra: Deep Learning enables an Ultra Fast Open Search by Jointly Embedding MS/MS Spectra and Peptides of Mass Spectrometry-based Proteomics

This code is an implementation of the open search of yHydra as described in our preprint: https://doi.org/10.1101/2021.12.01.470818  

### Disclaimer:

This code-repository implements a GPU-accelerated open search that uses the joint embeddings of yHydra.
Note, this repository contains trained models of yHydra. The yHydra training pipeline will be available soon.
Since this is ongoing research it may not reflect the final performance of yHydra.

## Getting started

**System requirements:** 
- Linux (tested on Ubuntu 18.04),
- Conda installed (e.g. follow documentation at https://conda.io/projects/conda/en/latest/user-guide/install/index.html),
- a GPU machine with latest CUDA installed
- all dependencies are defined in **`yhydra_env.yml`**, 'installation guide' see below.
- (CPU-only is possible but will result in poor runtime performance)

## Installation Guide

**Note:** installation of required packages takes **up to several minutes**.

Start by cloning this github-repository:

``` BASH
git clone https://github.com/tzom/yHydra
cd yHydra
git switch workflow
```

To install the required packages create a new conda environment (required packages are automatically installed when using **`yhydra_env.yml`**):

``` BASH
conda env create -f yhydra_env.yml
conda activate yhydra_env
bash install_thermorawfileparser.sh
```

## Demo: run yHydra on example data

The following commands will download example data (from https://www.ebi.ac.uk/pride/archive/projects/PXD007963): 

``` BASH
wget -nc -i example_data_urls.txt -P example/
```

To run yHydra on the downloaded example data, run the yHydra pipeline on the **`example/config.yaml`**:

``` BASH
python run.py example/config.yaml
```

## Example Output

The output for the example data should look like this (GPU-runtime ~120sec):

```
                  raw_file   scan  index  precursorMZ  ...                                best_peptide  peptide_mass delta_mass         q
20085  qe2_03132014_13WT-3  18697  13500  1086.209800  ...            DQARPVEDSTAIAQVGAISAGNDKEVGDMIAK   3255.604134   0.003436  0.000000
20129  qe2_03132014_13WT-3  27761  19281  1093.257100  ...            IRQGAFAIEAAQTITAKADTIVNGAAQAVYSK   3276.746639   0.002831  0.000000
21008  qe2_03132014_13WT-3  30120  20404   836.438600  ...  SKTPGHPENFETPGVEVTTGPIGQGIANAVGIAIAEAHIAAR   4177.155619   0.000998  0.000000
19867  qe2_03132014_13WT-3  24182  17380  1040.231000  ...               INHHIAGIFGVSSIAWTGHIVHVAIPESR   3117.662456   0.008714  0.000000
20692  qe2_03132014_13WT-3  25013  17904   738.375400  ...            RDNIIAMVDYIKNPTSYDGEIDISQIHPNTVR   3686.836259   0.004358  0.000000
...                    ...    ...    ...          ...  ...                                         ...           ...        ...       ...
1255   qe2_03132014_13WT-3   9446   6113   464.777191  ...                                   IQAGIEVAK    927.538931   0.000898  0.009754
601    qe2_03132014_13WT-3  17990  12970   415.260986  ...                                     TIVDIIR    828.506903   0.000517  0.009754
492    qe2_03132014_13WT-3  11680   7859   402.229889  ...                                     SIEDVIK    802.443634   0.001591  0.009754
553    qe2_03132014_13WT-3  11475   7702   409.234131  ...                                     IGEMVIR    816.452759   0.000949  0.009754
2478   qe2_03132014_13WT-3   4230   1840   349.516785  ...                                   SYVVQTGHR   1045.530492  -0.001968  0.009935

[5848 rows x 16 columns]
```

## Instructions for use

**Search Parameters of yHydra:** 

To run yHydra specify the location of input files in **`./example/config.yaml`**:

``` YAML
# Input - File Locations
FASTA: example/*.fasta
RAWs: example/*.raw

# Output - Results directory
RESULTS_DIR: example/search

# General Parameters
BATCH_SIZE: 64
...
```

then you can run yHydra using specified parameters (**`./example/config.yaml`**):

``` BASH
python run.py ./example/config.yaml
```

## Author

Tom Altenburg (tzom)
