#!/bin/bash

source config.env
source ../../../scripts/build.sh

display_help() {
    echo "Usage: $0 MODEL_PATH [OPTIONS] [IMAGE PATHS]"
    echo "       $0 MODEL_PATH --dataset DIRECTORY"
    echo
    echo "Options:"
    echo "  --dataset DIRECTORY   Specify a directory containing images"
    echo "  -h, --help            Display this help message"
    echo
    echo "Arguments:"
    echo "  MODEL_PATH            Path to the model"
    echo "  IMAGE PATHS           List of image paths"
    echo

}

process_image() {
    echo "$1" >> $filepaths_list
}

process_directory() {
    local directory="$1"
    find "$directory" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | while read -r file; do
        process_image "$file"
    done
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

if [ "$1" == "--dataset" ]; then
    if [ -z "$2" ]; then
        echo "No directory specified for --dataset option"
        exit 1
    elif [ "$2" == "-h" ] || [ "$2" == "--help" ]; then
        display_help
        exit 0
    elif [ -d "$2" ]; then
        process_directory "$2"
    else
        echo "Invalid directory specified: $2"
        exit 1
    fi
else
    for arg in "$@"; do
        if [ "$arg" == "-h" ] || [ "$arg" == "--help" ]; then
            display_help
            exit 0
        elif [ -f "$arg" ]; then
            process_image "$arg"
        else
            echo "Invalid file specified: $arg"
        fi
    done
fi

RESULTS_FOLDER="$(pwd)"/artifacts/inference/results
mkdir -p "$RESULTS_FOLDER"
RESULT_FILE=$RESULTS_FOLDER/result.txt
touch "$RESULT_FILE"
cd "$VAI_LIBRARY_SAMPLES_PATH" || exit

EVAL_APP=test_accuracy_"$VAI_SAMPLES_POSTFIX"_mt
build_if_not_exists $VAI_LIBRARY_SAMPLES_PATH $EVAL_APP

POSTFIX="_acc"
RENAMED_MODEL_PATH="${MODEL_PATH}${POSTFIX}"
mv "$MODEL_PATH" "$RENAMED_MODEL_PATH"
./$EVAL_APP $MODEL_PATH $filepaths_list $RESULT_FILE
mv "$RENAMED_MODEL_PATH" "$MODEL_PATH"
