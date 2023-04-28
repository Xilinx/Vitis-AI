#!/bin/bash

source scripts/config.env
source scripts/build.sh

if [[ "$1" == "-h" || "$1" == "--help" || $# -lt 2 ]]; then
    echo "Usage: $0 <MODEL_PATH> [<image paths list>]"
    exit 2
fi

filepaths_list=$CURRENT_MODEL_ZOO_PATH/artifacts/inference/filepaths.list
rm "$filepaths_list"

for arg in "$@"; do
   if [ "$arg" != "$1" ]
   then
      echo "$arg" >> "$filepaths_list"
   fi
done

MODEL_PATH="$1"

RESULTS_FOLDER=$CURRENT_MODEL_ZOO_PATH/artifacts/inference/results
mkdir -p "$RESULTS_FOLDER"

echo "Run the inference script"
cd $EVAL_SCRIPT_FOLDER

EVAL_APP=test_accuracy_$VAI_SAMPLES_POSTFIX
build_if_not_exists $EVAL_APP

./$EVAL_APP $MODEL_PATH $filepaths_list $RESULTS_FOLDER
