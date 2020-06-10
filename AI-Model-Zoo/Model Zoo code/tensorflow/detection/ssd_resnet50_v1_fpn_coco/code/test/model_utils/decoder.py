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

# PART OF THIS FILE AT ALL TIMES.

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf

def combined_static_and_dynamic_shape(tensor):
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def get_center_coordinates_and_sizes(box_corners, scope=None):
  """Computes the center coordinates, height and width of the boxes.

  Args:
    scope: name scope of the function.

  Returns:
    a list of 4 1-D tensors [ycenter, xcenter, height, width].
  """
  with tf.name_scope(scope, 'get_center_coordinates_and_sizes'):
    ymin, xmin, ymax, xmax = tf.unstack(tf.transpose(box_corners))
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return [ycenter, xcenter, height, width]

def faster_rcnn_box_coder_decode(rel_codes, anchors, scale_factors):
  """Decode relative codes to boxes.

  Args:
    rel_codes: a tensor representing N anchor-encoded boxes.
    anchors: BoxList of anchors.

  Returns:
    boxes: BoxList holding N bounding boxes.
  """
  ycenter_a, xcenter_a, ha, wa = get_center_coordinates_and_sizes(anchors)

  ty, tx, th, tw = tf.unstack(tf.transpose(rel_codes))
  if scale_factors:
    ty /= scale_factors[0]
    tx /= scale_factors[1]
    th /= scale_factors[2]
    tw /= scale_factors[3]
  w = tf.exp(tw) * wa
  h = tf.exp(th) * ha
  ycenter = ty * ha + ycenter_a
  xcenter = tx * wa + xcenter_a
  ymin = ycenter - h / 2.
  xmin = xcenter - w / 2.
  ymax = ycenter + h / 2.
  xmax = xcenter + w / 2.
  return tf.transpose(tf.stack([ymin, xmin, ymax, xmax]))


def batch_decode(box_encodings, anchors, scale_factors):
  """Decodes a batch of box encodings with respect to the anchors.
  """
  combined_shape = combined_static_and_dynamic_shape(box_encodings)
  batch_size = combined_shape[0]
  tiled_anchor_boxes = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])
  tiled_anchors_boxlist = tf.reshape(tiled_anchor_boxes, [-1, 4])
  decoded_boxes = faster_rcnn_box_coder_decode(tf.reshape(box_encodings, [-1, 4]),
                        tiled_anchors_boxlist, scale_factors)
  decoded_boxes = tf.reshape(decoded_boxes, tf.stack([combined_shape[0], combined_shape[1], 4]))
  return decoded_boxes

