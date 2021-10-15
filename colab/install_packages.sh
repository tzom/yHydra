pip install -r requirements.txt
apt install libomp-dev
pip install faiss-gpu
pip install --upgrade tables

#ThermoRawFileParser
apt-get install mono-complete
mkdir -p ThermoRawFileParser
cd ThermoRawFileParser
wget https://github.com/compomics/ThermoRawFileParser/releases/download/v1.3.4/ThermoRawFileParser.zip
unzip ThermoRawFileParser.zip