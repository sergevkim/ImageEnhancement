#!/bin/bash

mkdir data
wget https://www.dropbox.com/s/unasbd2z9d6204b/anime_kaggle.zip
unzip anime_kaggle.zip
mv images data
rm anime_kaggle.zip

