#!/usr/bin/env bash
wget -c --tries=10 https://github.com/andrewaeva/DGA/archive/master.zip
unzip master.zip 'DGA-master/all_dga.txt'
rm master.zip
cd DGA-master
shuf -n 10000 all_dga.txt > bad10k.txt
shuf -n 100000 all_dga.txt > bad100k.txt