#!/bin/bash

test_data_archive_filename=vitis_ai_library_r3.0.0_images.tar.gz

VITIS_AI_LIBRARY_FOLDER=/workspace/Vitis-AI-Library
if [ -d $VITIS_AI_LIBRARY_FOLDER ]
then
    echo "$VITIS_AI_LIBRARY_FOLDER exists. Skipping downloading the data."
else
    echo "Downloading test data"
    wget https://www.xilinx.com/bin/public/openDownload?filename=$test_data_archive_filename -O $test_data_archive_filename
    echo "Making the directory $VITIS_AI_LIBRARY_FOLDER for test data"
    mkdir Vitis-AI-Library
    echo "Unzipping"
    tar -xzvf vitis_ai_library_r3.0.0_images.tar.gz -C $VITIS_AI_LIBRARY_FOLDER
    echo "Removing the downloaded archive"
    rm $test_data_archive_filename
fi
