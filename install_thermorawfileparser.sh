#!/bin/bash

mkdir ext/
wget https://github.com/compomics/ThermoRawFileParser/releases/download/v1.3.4/ThermoRawFileParser.zip
mv ThermoRawFileParser.zip ext/
unzip ext/ThermoRawFileParser.zip -d ext/ThermoRawFileParser/
mono ext/ThermoRawFileParser/ThermoRawFileParser.exe --version > /dev/null
if [ $? -eq 0 ]; then
   echo "ThermoRawFileParser was successfully installed."
fi