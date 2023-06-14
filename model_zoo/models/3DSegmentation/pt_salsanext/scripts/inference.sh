#!/bin/bash

source config.env
source ../../../scripts/build.sh

display_help() {
    echo "Usage: $0 MODEL_PATH [OPTIONS] [IMAGE PATH]"
    echo
    echo "Options:"
    echo "  -h, --help            Display this help message"
    echo
    echo "Arguments:"
    echo "  MODEL_PATH            Path to the model"
    echo "  IMAGE PATH            Path to the test image"
    echo

}
if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    display_help
    exit 0
fi

MODEL_PATH="$1"


shift

filepaths_list="$(pwd)"/artifacts/inference/filepaths.list
if [ -f "$filepaths_list" ]; then
    rm "$filepaths_list"
fi

image_path=$1
image_name=$(basename -- "$image_path")
image_name="${image_name%.*}"

# Original folder path
original_folder=$(dirname -- "$image_path")

# Destination path
destination_path="$VAI_LIBRARY_SAMPLES_PATH/salsanext_input"

# Create the destination path if it doesn't exist
mkdir -p "$destination_path"

# Copy the image file to the destination path with a modified name
cp "$image_path" "$destination_path/-${image_name}.${image_path##*.}"

# Copy the text files to the destination path
cp "$original_folder/scan_x.txt" "$destination_path"
cp "$original_folder/scan_y.txt" "$destination_path"
cp "$original_folder/scan_z.txt" "$destination_path"
cp "$original_folder/scan_remission.txt" "$destination_path"

# Create the filepaths.list file and save the path to the renamed image
echo "$destination_path/-${image_name}.${image_path##*.}" > $filepaths_list

RESULTS_FOLDER="$(pwd)"/artifacts/inference/results
mkdir -p "$RESULTS_FOLDER"
cd "$VAI_LIBRARY_SAMPLES_PATH" || exit

EVAL_APP=test_accuracy_$VAI_SAMPLES_POSTFIX
build_if_not_exists $VAI_LIBRARY_SAMPLES_PATH $EVAL_APP

./$EVAL_APP $MODEL_PATH $filepaths_list $RESULTS_FOLDER
rm -rf $destination_path
