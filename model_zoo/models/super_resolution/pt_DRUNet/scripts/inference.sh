#!/bin/bash

source scripts/config.env
source ../../../scripts/build.sh

if [[ "$1" == "-h" || "$1" == "--help" || $# -lt 2 ]]; then
    echo "Usage: $0 <MODEL_PATH> [<image paths list>]"
    exit 2
fi

filepaths_list="$(pwd)"/artifacts/inference/filepaths.list
if [ -f "$filepaths_list" ]; then
    rm "$filepaths_list"
fi

for arg in "$@"; do
   if [ "$arg" != "$1" ]
   then
      echo "$arg" >> "$filepaths_list"
   fi
done

MODEL_PATH="$1"

RESULTS_FOLDER="$(pwd)"/artifacts/inference/results
mkdir -p "$RESULTS_FOLDER"

cd "$VAI_LIBRARY_SAMPLES_PATH" || exit

EVAL_APP=test_accuracy_$VAI_SAMPLES_POSTFIX
build_if_not_exists $VAI_LIBRARY_SAMPLES_PATH $EVAL_APP

./$EVAL_APP $MODEL_PATH $filepaths_list $RESULTS_FOLDER
