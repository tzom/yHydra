# yHydra: Deep Learning enables an Ultra Fast Open Search by Jointly Embedding MS/MS Spectra and Peptides of Mass Spectrometry-based Proteomics

This code is an implementation of the open search of yHydra as described in our preprint: https://doi.org/10.1101/2021.12.01.470818  

### Disclaimer:

Specifically, this code-repository implements a GPU-accelerated open search that uses the joint embeddings of yHydra.
Since this is ongoing research it may not reflect the final performance of yHydra.

## Getting started

**System requirements:** 
- Linux (tested on Ubuntu 18.04),
- Conda installed (e.g. follow documentation at https://conda.io/projects/conda/en/latest/user-guide/install/index.html),
- a GPU machine with latest CUDA installed
- (CPU-only is possible but will result in poor performance)

**Note:** installation of required packages takes **up to several minutes** and any individual step (on the provided examples) takes less than one minute.

Start by cloning this github-repository:

```
git clone https://github.com/tzom/yHydra
cd yHydra
```

To install the required packages create a new conda environment (required packages are automatically installed when using **`yhydra_env.yml`**):

```
conda env create -f yhydra_env.yml
conda activate yhydra_env
bash install_thermorawfileparser.sh
```

## Run example

The following commands will download example data (from https://www.ebi.ac.uk/pride/archive/projects/PXD007963) and run the pipeline of yHydra: 

```
mkdir example
wget -nc -i example_data_urls.txt -P example/
bash run.sh config.yaml
```

## Run yHydra

To run yHydra specify the location of input files in **`./config.yaml`** by replacing the example directory:

``` YAML
# Input - File Locations
FASTA: example/*.fasta.gz
RAWs: example/*.raw

# Output - Results directory
RESULTS_DIR: example/search

# General Parameters
BATCH_SIZE: 64
...
```

you can run yHydra on specified parameters (**`./config.yaml`**):

```
bash run.sh config.yaml
```

## Inspect search results



## Author

Tom Altenburg (tzom)
