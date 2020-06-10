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


python export_graph.py

freeze_graph \
  --input_graph="resnet34_ssd.pbtxt" \
  --input_checkpoint="logs/model.ckpt-0" \
  --input_binary=false \
  --output_graph="resnet34_tf.pb" \
  --output_node_names="detection_bboxes,detection_scores,detection_classes,ssd1200/py_cls_pred,ssd1200/py_location_pred"
  
