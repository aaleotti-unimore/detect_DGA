#!/usr/bin/env bash
cd DGA-master
wget -c --tries=10 http://downloads.majesticseo.com/majestic_million.csv
split -l 10000 majestic_million.csv top10k --verbose --additional-suffix=".txt"
# split -l 10000 top 100000aa.txt top10k --verbose --additional-suffix=".txt"
# find . -type f -not -name 'top10kaa.txt,all_dga.txt' -print0 | xargs -0 rm --
