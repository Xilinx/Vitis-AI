#!/bin/bash


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
TEST_IMAGE="$2"
MODEL_FOLDER=${MODEL_PATH%%/artifacts*}

source $MODEL_FOLDER/config.env
source scripts/build.sh

VAITRACE_PATH=$MODEL_FOLDER/artifacts/inference/vaitrace
if [ -f "$VAITRACE_PATH" ]; then
    rm -rf "$VAITRACE_PATH"
fi
mkdir -p $VAITRACE_PATH



cd "$VAI_LIBRARY_SAMPLES_PATH" || exit

EVAL_APP=test_jpeg_$VAI_SAMPLES_POSTFIX
build_if_not_exists $VAI_LIBRARY_SAMPLES_PATH $EVAL_APP

echo "Run vaitrace on $EVAL_APP"
sudo -E vaitrace --txt_summary -o trace.txt ./$EVAL_APP $MODEL_PATH $TEST_IMAGE
sudo mv -t $VAITRACE_PATH summary.csv trace.txt xrt.run_summary
echo "Vaitrace completed."
