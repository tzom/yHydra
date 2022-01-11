# yHydra: Deep Learning enables an Ultra Fast Open Search by Jointly Embedding MS/MS Spectra and Peptides of Mass Spectrometry-based Proteomics

This code is an implementation of the open search of yHydra as described in our preprint: https://doi.org/10.1101/2021.12.01.470818  

### Disclaimer:

This code-repository implements a GPU-accelerated open search that uses the joint embeddings of yHydra.
Note, this repository contains trained models of yHydra. The yHydra training pipeline will be available soon.
Since this is ongoing research it may not reflect the final performance of yHydra.

## Getting started

**System requirements:** 
- Linux (tested on Ubuntu 18.04), WSL(tested on 5.10.60.1-microsoft-standard-WSL2)
- Conda installed (e.g. follow documentation at https://conda.io/projects/conda/en/latest/user-guide/install/index.html),
- a GPU machine with latest CUDA installed
- (CPU-only is possible but will result in poor performance)

**Note:** installation of required packages takes **up to several minutes**.

Start by cloning this github-repository:

``` BASH
git clone https://github.com/tzom/yHydra
cd yHydra
```

If running in WSL

``` BASH
sudo apt install dos2unix
dos2unix *
``` 

To install the required packages create a new conda environment (required packages are automatically installed when using **`yhydra_env.yml`**):

``` BASH
conda env create -f yhydra_env.yml
conda activate yhydra_env
bash install_thermorawfileparser.sh
conda install -y pandas lxml
sudo apt install jq
```


## Run the example

The following commands will download example data (from https://www.ebi.ac.uk/pride/archive/projects/PXD007963) and run the pipeline of yHydra: 

``` BASH
mkdir example
wget -nc -i example_data_urls.txt -P example/
gzip example/SynPCC7002_Cbase.fasta
bash run.sh config.yaml
```

## Run yHydra

To run yHydra specify the location of input files in **`./config.yaml`**:

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
(yhydra_env) animeshs@DMED7596:/mnt/f/OneDrive - NTNU/yHydra$ python inspect_search.py
                  raw_file     id                                           is_decoy  precursorMZ      pepmass  ...           best_peptide peptide_mass  delta_mass         q           accession
0       qe2_03132014_1WT-1  13677  [False, False, True, True, True, True, False, ...   699.032500  2094.074025  ...  ADTAGVHGAALGADEIELTRK  2094.070485    0.003540  0.000000  [SYNPCC7002_A1022]
1      qe2_03132014_13WT-3  17071  [False, False, False, True, False, True, True,...   900.941528  1799.867407  ...      DIVTQFHGAEAAVDAEK  1799.868945   -0.001538  0.000000  [SYNPCC7002_A1609]
2       qe2_03132014_1WT-1  20532  [False, False, False, False, True, False, Fals...   925.985352  1849.955053  ...     TLIEGLDEISHGGLPSGR  1849.953345    0.001708  0.000000  [SYNPCC7002_A0287]
3       qe2_03132014_5WT-2  18652  [False, False, True, True, False, False, False...   779.409100  2335.203825  ...  SIEAEQLKDDLPTIHVGDTVR  2335.201905    0.001920  0.000000  [SYNPCC7002_A1033]
4       qe2_03132014_1WT-1  17209  [False, False, True, True, False, True, False,...   900.942505  1799.869360  ...      DIVTQFHGAEAAVDAEK  1799.868945    0.000415  0.000000  [SYNPCC7002_A1609]
...                    ...    ...                                                ...          ...          ...  ...                    ...          ...         ...       ...                 ...
23643  qe2_03132014_13WT-3  17280  [True, False, False, True, False, True, True, ...   473.229431   944.443212  ...                MFDIFTR   928.447665   15.995548  0.009897  [SYNPCC7002_A2209]
23644   qe2_03132014_5WT-2   3564  [True, False, True, False, True, True, False, ...   329.179352  1312.686107  ...           KEESELIDAHGK  1354.672825  -41.986718  0.009980  [SYNPCC7002_A2459]
23645  qe2_03132014_13WT-3  14744  [False, True, False, True, False, False, False...   757.902893  1513.790136  ...          AEKNIILSIEDIR  1512.851125    0.939011  0.009980  [SYNPCC7002_F0019]
23646   qe2_03132014_5WT-2   4053  [False, True, False, True, False, False, False...   310.158813   927.452965  ...                RGGGGDR   673.325565  254.127401  0.009980  [SYNPCC7002_A0148]
23647  qe2_03132014_13WT-3   4498  [True, False, True, False, True, False, False,...   452.226990  1353.657494  ...          TNYVPHVSFTGTK  1449.725215  -96.067721  0.009980  [SYNPCC7002_A2578]

[23648 rows x 19 columns]
```

## Author

Tom Altenburg (tzom)
