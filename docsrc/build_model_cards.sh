#!/bin/bash

source_path="../model_zoo"
destination_path="source/model_cards"

readme_files=$(find "$source_path" -type f -name "README.md")

mkdir -p "$destination_path"

for readme_file in $readme_files; do
    task_name=$(dirname "$readme_file" | awk -F/ '{print $(NF-1)}')
    model_name=$(dirname "$readme_file" | awk -F/ '{print $NF}')

    destination_file="$destination_path/$model_name.md"

    cp "$readme_file" "$destination_file"

    sed -i 's/..\/..\/..\/README.md#quick-start/\..\/..\/getting-started-model-zoo.html#quick-start/g' "$destination_file"
    sed -i 's/..\/..\/..\/README.md#vaitrace/\..\/..\/getting-started-model-zoo.html#vaitrace/g' "$destination_file"
done
