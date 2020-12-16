# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/bin/bash
# usage:
#  bash data/download_and_preprocess.sh
# Download the data.

# Extract the data.
cd data/
unzip BDD100K/bdd100k_images.zip
unzip BDD100K/bdd100k_labels.zip
#
caffe_path="/Path_to_caffe_Xilinx/"
export PYTHONPATH=$caffe_path/python/
root_dir="bdd100k"
convert_label='labels_txt'
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

get_image_size_path="/build/tools/get_image_size"
get_image_size_path_docker="/bin/get_image_size"
create_annoset_path="/scripts/create_annoset.py"
create_annoset_path_docker="/bin/create_annoset.py"


caffe_xilinx_dir_docker="/opt/vitis_ai/conda/envs/vitis-ai-caffe/"
in_caffe_path() {
  exec_name=$1
  exec_path=$caffe_path$(eval echo '$'"${exec_name}_path")
  if [ ! -f "$exec_path" ]; then
    echo >&2 "$exec_path does not exist, try use path in pre-build docker"
    exec_path=$caffe_xilinx_dir_docker$(eval echo '$'"${exec_name}_path_docker")
  fi
  echo "$exec_path"
}

caffe_exec() {
  exec_path=$(in_caffe_path "$1")
  shift
  $exec_path "$@"
}

for dataset in train val
do
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi
  for name in images labels
  do
    if [ $name = "images" ]
    then
      mkdir $dataset
      mkdir $dataset/$name
   
      echo "copy images ...."
      find $root_dir/$name/100k/$dataset/ -name "*.jpg" | xargs -i cp {} $dataset/$name/
      find $dataset/$name -name "*.jpg" | xargs -i ls {} >$dataset"_img.txt"
      if [ $dataset = "val" ]
      then
        caffe_exec get_image_size ./ $dataset'_img.txt' $dataset"_name_size.txt"
      fi
    fi
    if [ $name = "labels" ]
    then
      mkdir $dataset/$name
      echo "copy the json ...."
      find $root_dir/$name/100k/$dataset/ -name "*.json" | xargs -i cp {} $dataset/$name/
      mkdir $dataset/$convert_label
      echo "convert the json file to txt and clean  the dataset"
      python ./../code/convert_jsonTotxt.py --json_file_path $dataset/$name --txt_file_path $dataset/$convert_label/
      find $dataset/$convert_label/ -name "*.txt" | xargs -i ls {} >$dataset'_label.txt'
    fi
  done
    paste -d' ' $dataset'_img.txt' $dataset'_label.txt' >> $dataset'.txt'
    rm $dataset'_img.txt'
    rm $dataset'_label.txt'
done


redo=1
data_root_dir="."
dataset_name="bdd100k"
mapfile="../labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0
label_type='txt'

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
echo "convert jpg format to lmdb"
for subset in val train
do
  python "$(in_caffe_path create_annoset)" --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --label-type=$label_type --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $subset.txt $db/$dataset_name"_"$subset"_"$db $dataset_name
done

#echo "Now for training and testing"
#cd ../../
#sh code/train/trainval.sh
