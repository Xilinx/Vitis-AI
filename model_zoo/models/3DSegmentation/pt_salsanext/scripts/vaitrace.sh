#!/bin/bash

source config.env
source ../../../scripts/build.sh

display_help() {
    echo "Usage: $0 MODEL_PATH TEST_IMAGE_PATH"
    echo "       $0 MODEL_PATH --dataset DIRECTORY"
    echo
    echo "Options:"
    echo "  --dataset DIRECTORY   Specify a directory containing images"
    echo "  -h, --help            Display this help message"
    echo
    echo "Arguments:"
    echo "  MODEL_PATH            Path to the model"
    echo "  TEST_IMAGE_PATH       Path to the image to be processed via vaitrace"
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
cp "$image_path" "$destination_path/${image_name}.${image_path##*.}"

# Copy the text files to the destination path
cp "$original_folder/scan_x.txt" "$destination_path"
cp "$original_folder/scan_y.txt" "$destination_path"
cp "$original_folder/scan_z.txt" "$destination_path"
cp "$original_folder/scan_remission.txt" "$destination_path"

# Create the filepaths.list file and save the path to the image
echo "$destination_path/${image_name}.${image_path##*.}" > $filepaths_list



VAITRACE_PATH="$(pwd)"/artifacts/inference/vaitrace
if [ -f "$VAITRACE_PATH" ]; then
    rm -rf "$VAITRACE_PATH"
fi
mkdir -p $VAITRACE_PATH



cd "$VAI_LIBRARY_SAMPLES_PATH" || exit

EVAL_APP=test_jpeg_$VAI_SAMPLES_POSTFIX
build_if_not_exists $VAI_LIBRARY_SAMPLES_PATH $EVAL_APP

echo "Run vaitrace on $EVAL_APP"
sudo -E vaitrace --txt_summary -o trace.txt ./$EVAL_APP $MODEL_PATH "$destination_path/scan_x.txt" "$destination_path/scan_y.txt" "$destination_path/scan_z.txt" "$destination_path/scan_remission.txt"
sudo mv -t $VAITRACE_PATH summary.csv trace.txt xrt.run_summary *3Dsegmentation_result.txt
rm -rf $destination_path
echo "Vaitrace completed."
