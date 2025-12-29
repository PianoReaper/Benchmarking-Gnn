#!/bin/bash

# Command to download dataset:
# bash script_download_molecules.sh

DIR=molecules/
# Check if directory exists, if not create it
mkdir -p $DIR
cd $DIR

# Download ZINC (10k subset)
FILE=ZINC.pkl
if test -f "$FILE"; then
  echo -e "$FILE already downloaded."
else
  echo -e "\nDownloading $FILE..."
  curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl -o ZINC.pkl -J -L -k
fi

# Download ZINC-full (250k full dataset)
FILE=ZINC-full.pkl
if test -f "$FILE"; then
  echo -e "$FILE already downloaded."
else
  echo -e "\nDownloading $FILE..."
  curl https://data.dgl.ai/dataset/benchmarking-gnns/ZINC-full.pkl -o ZINC-full.pkl -J -L -k
fi