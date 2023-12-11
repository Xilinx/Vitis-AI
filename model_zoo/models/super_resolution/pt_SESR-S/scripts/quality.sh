#!/bin/bash

source config.env
source ../../../scripts/build.sh



display_help() {
    echo "Usage: $0 DATASET_FOLDER INFERENCE_FOLDER"
    echo
    echo "Options:"
    echo "  -h, --help            Display this help message"
    echo
    echo "Arguments:"
    echo "  DATASET_FOLDER            The folder where original dataset is stored."
    echo "  INFERENCE_FOLDER          The folder where results of model inference is stored."
    echo

}
if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    display_help
    exit 0
fi
bash scripts/setup_venv.sh

QUALITY_FOLDER="$(pwd)"/artifacts/inference/quality
mkdir -p "$QUALITY_FOLDER"
result_file=$QUALITY_FOLDER/psnr.txt

python src/quality.py $1 $2 | tee $result_file
