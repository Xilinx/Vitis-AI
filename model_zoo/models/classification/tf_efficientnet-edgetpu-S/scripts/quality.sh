#!/bin/bash

source config.env
source ../../../scripts/build.sh


display_help() {
    echo "Usage: $0 <inference_result> <ground_truth> [--batch] [--dataset]"
    echo ""
    echo "Evaluate the accuracy of predicted classes on the given image."
    echo ""
    echo "Positional arguments:"
    echo "  inference_result  Path to the inference result image or folder"
    echo "  ground_truth      Path to the ground truth image or folder"
    echo ""
    echo "Optional arguments:"
    echo "  --batch        Evaluate a folder (default: individual images)"
    echo "  --dataset      Evaluate CompCars dataset"
    echo ""

}
if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    display_help
    exit 0
fi
bash scripts/setup_venv.sh

QUALITY_FOLDER="$(pwd)"/artifacts/inference/quality
mkdir -p "$QUALITY_FOLDER"
result_file=$QUALITY_FOLDER/metrics.txt

inference_result="$1"
ground_truth="$2"
dataset=""
batch=""

if [[ "$3" == "--batch" ]]; then
  batch="--batch"
fi

if [[ "$3" == "--dataset" ]]; then
  dataset="--dataset"
fi

if [[ "$4" == "--dataset" ]]; then
  dataset="--datasets"
fi

python src/quality.py $inference_result $ground_truth $batch $dataset | tee $result_file
