#!/bin/bash

for model_folder in "$(pwd)"/models/*/*/; do
    cd "$model_folder"
    mkdir -p artifacts/models
    mkdir -p artifacts/inference
done
