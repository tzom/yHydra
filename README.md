# yHydra: Deep Learning enables an Ultra Fast Open Search by Jointly Embedding MS/MS Spectra and Peptides of Mass Spectrometry-based Proteomics

This code is an implementation of the open search of yHydra as described in our preprint: https://doi.org/10.1101/2021.12.01.470818  

### Disclaimer:

This code-repository implements a GPU-accelerated open search that uses the joint embeddings of yHydra.
Note, this repository contains trained models of yHydra and the yHydra training pipeline will be available soon.
Since this is ongoing research it may not reflect the final performance of yHydra.

## Getting started

**System requirements:** 
- Linux (tested on Ubuntu 18.04),
- Conda installed (e.g. follow documentation at https://conda.io/projects/conda/en/latest/user-guide/install/index.html),
- a GPU machine with latest CUDA installed
- (CPU-only is possible but will result in poor performance)

**Note:** installation of required packages takes **up to several minutes** and any individual step (on the provided examples) takes less than one minute.

Start by cloning this github-repository:

``` BASH
git clone https://github.com/tzom/yHydra
cd yHydra
```

To install the required packages create a new conda environment (required packages are automatically installed when using **`yhydra_env.yml`**):

``` BASH
conda env create -f yhydra_env.yml
conda activate yhydra_env
bash install_thermorawfileparser.sh
```

## Run the example

The following commands will download example data (from https://www.ebi.ac.uk/pride/archive/projects/PXD007963) and run the pipeline of yHydra: 

``` BASH
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

then you can run yHydra using specified parameters (**`./config.yaml`**):

``` BASH
bash run.sh config.yaml
```

## Inspect search results

The search results are dumped as dataframe in .hdf-files (e.g. locations is specified as **`RESULTS_DIR: example/search`** in the **`./config.yaml`**), in order to get a glimpse of identfications, you can run this:

```
python inspect_search.py
```

which gives you the following output:

```  BASH
(yhydra_env) tom.altenburg@node:~/$ python inspect_search.py 
                  raw_file     id  precursorMZ      pepmass  charge  ...           best_peptide peptide_mass delta_mass         q           accession
0       qe2_03132014_1WT-1  13677   699.032500  2094.074025       3  ...  ADTAGVHGAALGADEIELTRK  2094.070485   0.003540  0.000000  [SYNPCC7002_A1022]
1      qe2_03132014_13WT-3  17071   900.941528  1799.867407       2  ...      DIVTQFHGAEAAVDAEK  1799.868945  -0.001538  0.000000  [SYNPCC7002_A1609]
2       qe2_03132014_1WT-1  20532   925.985352  1849.955053       2  ...     TLIEGLDEISHGGLPSGR  1849.953345   0.001708  0.000000  [SYNPCC7002_A0287]
3       qe2_03132014_1WT-1  17209   900.942505  1799.869360       2  ...      DIVTQFHGAEAAVDAEK  1799.868945   0.000415  0.000000  [SYNPCC7002_A1609]
4      qe2_03132014_13WT-3  18497   779.408800  2335.202925       3  ...  SIEAEQLKDDLPTIHVGDTVR  2335.201905   0.001020  0.000000  [SYNPCC7002_A1033]
...                    ...    ...          ...          ...     ...  ...                    ...          ...        ...       ...                 ...
15738  qe2_03132014_13WT-3  18299   473.229095   944.442541       2  ...                MFDIFTR   928.447665  15.994876  0.009911  [SYNPCC7002_A2209]
15739  qe2_03132014_13WT-3  12375   608.333801  1214.651952       2  ...            NVADEVIKEAK  1214.650635   0.001318  0.009911  [SYNPCC7002_A0341]
15740   qe2_03132014_1WT-1   8078   474.262451   946.509252       2  ...              IAETLTGSR   946.508345   0.000908  0.009973  [SYNPCC7002_A1930]
15741  qe2_03132014_13WT-3   7441   337.198120  1008.570885       3  ...              LLGHTEIAR  1008.571605  -0.000719  0.009973  [SYNPCC7002_A0246]
15742  qe2_03132014_13WT-3  17280   473.229431   944.443212       2  ...                MFDIFTR   928.447665  15.995548  0.009973  [SYNPCC7002_A2209]
```

## Author

Tom Altenburg (tzom)
