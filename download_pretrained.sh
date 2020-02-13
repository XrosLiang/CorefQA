#!/bin/bash
echo Downloading $1
wget -P $data_dir https://www.dropbox.com/s/cf6feqgakafqnko/$1.tar.gz?dl=0
tar xvzf $data_dir/$1.tar.gz?dl=0 -C $data_dir
rm $data_dir/$1.tar.gz?dl=0