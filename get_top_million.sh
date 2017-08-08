#!/usr/bin/env bash
cd DGA-master
wget -c --tries=10 http://downloads.majesticseo.com/majestic_million.csv
echo "$(tail -n +2 majestic_million.csv)" > majestic_million.txt
shuf -n 10000 majestic_million.txt > top10k.txt
rm majestic_million.txt