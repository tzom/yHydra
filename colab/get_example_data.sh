#!/bin/bash

#PXD026566
#wget "http://ftp.pride.ebi.ac.uk/pride/data/archive/2021/06/PXD026566/20210311_HW4_001.mgf"
#wget -O 10090.fasta "https://www.uniprot.org/uniprot/?query=taxonomy:10090&format=fasta"
#gzip 10090.fasta

#PXD007963
wget "http://ftp.pride.ebi.ac.uk/pride/data/archive/2019/01/PXD007963/SynPCC7002_Cbase.fasta"
gzip SynPCC7002_Cbase.fasta
wget "http://ftp.pride.ebi.ac.uk/pride/data/archive/2019/01/PXD007963/qe2_03132014_1WT-1.raw"