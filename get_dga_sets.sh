#!/usr/bin/env bash
wget -c --tries=10 https://github.com/andrewaeva/DGA/archive/master.zip
unzip master.zip 'DGA-master/all_dga.txt'
rm master.zip
split -l 10000 'DGA-master/all_dga.txt' DGA-master/bad10k --additional-suffix=".txt"
cd DGA-master
find . -type f -not -name 'bad10kaa.txt,all_dga.txt' -print0 | xargs -0 rm --