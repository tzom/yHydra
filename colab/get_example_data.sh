#!/bin/bash
#PXD026566
wget "http://ftp.pride.ebi.ac.uk/pride/data/archive/2021/06/PXD026566/20210311_HW4_001.mgf"
wget -O 10090.fasta "https://www.uniprot.org/uniprot/?query=taxonomy:10090&format=fasta"
gzip 10090.fasta