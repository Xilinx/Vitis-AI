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
# ==============================================================================
set -e

export SQUAD_DIR=./data
export BERT_DIR=./float/bert-large-uncased-whole-word-masking-finetuned-squad
export OUTPUT=./output/bert_squadv1.1
export CACHE_FILE_NAME=./cache_dir/cache_file.arrow

export RUST_BACKTRACE=1
export HF_DATASETS_OFFLINE=1
export TF_CPP_MIN_LOG_LEVEL=1
export CUDA_VISIBLE_DEVICES=0

python code/run_qa.py \
    --model_name_or_path $BERT_DIR \
    --output_dir $OUTPUT  \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 1 \
    --per_device_train_batch_size 6 \
    --dataset_name $SQUAD_DIR/squad_offline \
    --cache_file_name $CACHE_FILE_NAME \
    --do_eval \
    --overwrite_output_dir \
    --overwrite_cache

python $SQUAD_DIR/SQuADv1.1/evaluate-v1.1.py $SQUAD_DIR/SQuADv1.1/dev-v1.1.json $OUTPUT/eval_predictions.json
