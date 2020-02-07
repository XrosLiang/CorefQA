#!/bin/bash
echo Downloading $1
wget -P $data_dir 39.96.28.249/home/work/$1.tar.gz
tar xvzf $data_dir/$1.tar.gz -C $data_dir
rm $data_dir/$1.tar.gz