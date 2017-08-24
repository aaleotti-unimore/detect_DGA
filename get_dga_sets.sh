#!/usr/bin/env bash
wget -c --tries=10 https://github.com/andrewaeva/DGA/archive/master.zip
unzip master.zip 'DGA-master/all_dga.txt'
rm master.zip
cd DGA-master
wget -c --tries=10 http://downloads.majesticseo.com/majestic_million.csv
cd ..
mv DGA-master datasets