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
import numpy as np
from .postprocess import to_normalized_coordinates

def expanded_shape(orig_shape, start_dim, num_dims):
  """Inserts multiple ones into a shape vector.

  Inserts an all-1 vector of length num_dims at position start_dim into a shape.
  Can be combined with tf.reshape to generalize tf.expand_dims.

  Args:
    orig_shape: the shape into which the all-1 vector is added (int32 vector)
    start_dim: insertion position (int scalar)
    num_dims: length of the inserted all-1 vector (int scalar)
  Returns:
    An int32 vector of length tf.size(orig_shape) + num_dims.
  """
  with tf.name_scope('ExpandedShape'):
    start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
    before = tf.slice(orig_shape, [0], start_dim)
    add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
    after = tf.slice(orig_shape, start_dim, [-1])
    new_shape = tf.concat([before, add_shape, after], 0)
    return new_shape

def meshgrid(x, y):
  """Tiles the contents of x and y into a pair of grids.

  Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
  are vectors. Generally, this will give:

  xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
  ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)

  Keep in mind that the order of the arguments and outputs is reverse relative
  to the order of the indices they go into, done for compatibility with numpy.
  The output tensors have the same shapes.  Specifically:

  xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
  ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())

  Args:
    x: A tensor of arbitrary shape and rank. xgrid will contain these values
       varying in its last dimensions.
    y: A tensor of arbitrary shape and rank. ygrid will contain these values
       varying in its first dimensions.
  Returns:
    A tuple of tensors (xgrid, ygrid).
  """
  with tf.name_scope('Meshgrid'):
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
    y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

    xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
    ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
    new_shape = y.get_shape().concatenate(x.get_shape())
    xgrid.set_shape(new_shape)
    ygrid.set_shape(new_shape)

    return xgrid, ygrid

def center_size_bbox_to_corners_bbox(centers, sizes):
  """Converts bbox center-size representation to corners representation.

  Args:
    centers: a tensor with shape [N, 2] representing bounding box centers
    sizes: a tensor with shape [N, 2] representing bounding boxes

  Returns:
    corners: tensor with shape [N, 4] representing bounding boxes in corners
      representation
  """
  return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)

def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):
  """Create a tiled set of anchors strided along a grid in image space.

  This op creates a set of anchor boxes by placing a "basis" collection of
  boxes with user-specified scales and aspect ratios centered at evenly
  distributed points along a grid.  The basis collection is specified via the
  scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]
  and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
  .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
  and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
  placing it over its respective center.

  Grid points are specified via grid_height, grid_width parameters as well as
  the anchor_stride and anchor_offset parameters.

  Args:
    grid_height: size of the grid in the y direction (int or int scalar tensor)
    grid_width: size of the grid in the x direction (int or int scalar tensor)
    scales: a 1-d  (float) tensor representing the scale of each box in the
      basis set.
    aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
      box in the basis set.  The length of the scales and aspect_ratios tensors
      must be equal.
    base_anchor_size: base anchor size as [height, width]
      (float tensor of shape [2])
    anchor_stride: difference in centers between base anchors for adjacent grid
                   positions (float tensor of shape [2])
    anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                   upper left element of the grid, this should be zero for
                   feature networks with only VALID padding and even receptive
                   field size, but may need some additional calculation if other
                   padding is used (float tensor of shape [2])
  Returns:
    a Tensor holding a collection of N anchor boxes
  """
  ratio_sqrts = tf.sqrt(aspect_ratios)
  heights = scales / ratio_sqrts * base_anchor_size[0]
  widths = scales * ratio_sqrts * base_anchor_size[1]

  # Get a grid of box centers
  y_centers = tf.to_float(tf.range(grid_height))
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = tf.to_float(tf.range(grid_width))
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = meshgrid(x_centers, y_centers)

  widths_grid, x_centers_grid = meshgrid(widths, x_centers)
  heights_grid, y_centers_grid = meshgrid(heights, y_centers)
  bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
  bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
  bbox_centers = tf.reshape(bbox_centers, [-1, 2])
  bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
  bbox_corners = center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
  return bbox_corners

def build_multiple_grid_anchors(anchor_config, feature_map_shape_list, im_height=1, im_width=1):
  num_layers = anchor_config.num_layers
  min_scale = anchor_config.min_scale
  max_scale = anchor_config.max_scale
  scales=[float(scale) for scale in anchor_config.scales]
  aspect_ratios = anchor_config.aspect_ratios
  interpolated_scale_aspect_ratio = anchor_config.interpolated_scale_aspect_ratio
  base_anchor_size = anchor_config.base_anchor_size
  anchor_strides = anchor_config.anchor_strides
  anchor_offsets = anchor_config.anchor_offsets
  reduce_boxes_in_lowest_layer = anchor_config.reduce_boxes_in_lowest_layer

  if base_anchor_size is None:
    base_anchor_size = [1.0, 1.0]
  box_specs_list = []
  if scales is None or not scales:
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)] + [1.0]
  else:
    # Add 1.0 to the end, which will only be used in scale_next below and used
    # for computing an interpolated scale for the largest scale in the list.
    scales += [1.0]

  for layer, scale, scale_next in zip(
      range(num_layers), scales[:-1], scales[1:]):
    layer_box_specs = []
    if layer == 0 and reduce_boxes_in_lowest_layer:
      layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
    else:
      for aspect_ratio in aspect_ratios:
        layer_box_specs.append((scale, aspect_ratio))
      # Add one more anchor, with a scale between the current scale, and the
      # scale for the next layer, with a specified aspect ratio (1.0 by
      # default).
      if interpolated_scale_aspect_ratio > 0.0:
        layer_box_specs.append((np.sqrt(scale*scale_next),
                                interpolated_scale_aspect_ratio))
    box_specs_list.append(layer_box_specs)

  scales_list = []
  aspect_ratios_list = []
  for box_spec in box_specs_list:
    scales, aspect_ratios = zip(*box_spec)
    scales_list.append(scales)
    aspect_ratios_list.append(aspect_ratios)

  im_height = tf.to_float(im_height)
  im_width = tf.to_float(im_width)

  if not anchor_strides:
    anchor_strides = [(1.0 / tf.to_float(pair[0]), 1.0 / tf.to_float(pair[1]))
                      for pair in feature_map_shape_list]
  else:
    anchor_strides = [(tf.to_float(stride[0]) / im_height,
                       tf.to_float(stride[1]) / im_width)
                      for stride in anchor_strides]
  if not anchor_offsets:
    anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
                      for stride in anchor_strides]
  else:
    anchor_offsets = [(tf.to_float(offset[0]) / im_height,
                       tf.to_float(offset[1]) / im_width)
                      for offset in anchor_offsets]

  anchor_grid_list = []
  min_im_shape = tf.minimum(im_height, im_width)
  scale_height = min_im_shape / im_height
  scale_width = min_im_shape / im_width
  if not tf.contrib.framework.is_tensor(base_anchor_size):
    base_anchor_size = [
        scale_height * tf.constant(base_anchor_size[0],
                                   dtype=tf.float32),
        scale_width * tf.constant(base_anchor_size[1],
                                  dtype=tf.float32)
    ]
  else:
    base_anchor_size = [
        scale_height * base_anchor_size[0],
        scale_width * base_anchor_size[1]
    ]
  for feature_map_index, (grid_size, scales, aspect_ratios, stride,
                          offset) in enumerate(
                              zip(feature_map_shape_list, scales_list,
                                  aspect_ratios_list, anchor_strides,
                                  anchor_offsets)):
    tiled_anchors = tile_anchors(
        grid_height=grid_size[0],
        grid_width=grid_size[1],
        scales=scales,
        aspect_ratios=aspect_ratios,
        base_anchor_size=base_anchor_size,
        anchor_stride=stride,
        anchor_offset=offset)
    anchor_grid_list.append(tiled_anchors)

  all_anchors = tf.concat(anchor_grid_list, 0)

  return all_anchors

def build_multiscale_grid_anchors(anchor_config, feature_map_shape_list, im_height=1, im_width=1):
  min_level = anchor_config.min_level
  max_level = anchor_config.max_level
  anchor_scale = anchor_config.anchor_scale
  aspect_ratios = anchor_config.aspect_ratios
  scales_per_octave = anchor_config.scales_per_octave
  normalize_coordinates = anchor_config.normalize_coordinates

  scales = [2**(float(scale) / scales_per_octave)
            for scale in range(scales_per_octave)]
  aspects = list(aspect_ratios)

  anchor_grid_info = []
  for level in range(min_level, max_level + 1):
    anchor_stride = [2**level, 2**level]
    base_anchor_size = [2**level * anchor_scale, 2**level * anchor_scale]
    anchor_grid_info.append({
        'level': level,
        'info': [scales, aspects, base_anchor_size, anchor_stride]
    })

  anchor_grid_list = []
  for feat_shape, grid_info in zip(feature_map_shape_list,
                                   anchor_grid_info):
    level = grid_info['level']
    stride = 2**level
    scales, aspect_ratios, base_anchor_size, anchor_stride = grid_info['info']
    feat_h = feat_shape[0]
    feat_w = feat_shape[1]
    anchor_offset = [0, 0]

    if isinstance(im_height, int) and isinstance(im_width, int):
      if im_height % 2.0**level == 0 or im_height == 1:
        anchor_offset[0] = stride / 2.0
      if im_width % 2.0**level == 0 or im_width == 1:
        anchor_offset[1] = stride / 2.0

    single_map_shape_list = [(feat_h, feat_w)]
    if not (isinstance(single_map_shape_list, list)
            and len(single_map_shape_list) == 1):
      raise ValueError('feature_map_shape_list must be a list of length 1.')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in single_map_shape_list]):
      raise ValueError('feature_map_shape_list must be a list of pairs.')
    base_anchor_size = tf.to_float(tf.convert_to_tensor(base_anchor_size))
    anchor_stride = tf.to_float(tf.convert_to_tensor(anchor_stride))
    anchor_offset = tf.to_float(tf.convert_to_tensor(anchor_offset))

    grid_height, grid_width = single_map_shape_list[0]
    scales_grid, aspect_ratios_grid = meshgrid(scales, aspect_ratios)
    scales_grid = tf.reshape(scales_grid, [-1])
    aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
    anchors = tile_anchors(grid_height,
                           grid_width,
                           scales_grid,
                           aspect_ratios_grid,
                           base_anchor_size,
                           anchor_stride,
                           anchor_offset)
    anchor_grid = {}
    anchor_grid['boxes'] = anchors

    if normalize_coordinates:
      if im_height == 1 or im_width == 1:
        raise ValueError(
            'Normalized coordinates were requested upon construction of the '
            'MultiscaleGridAnchorGenerator, but a subsequent call to '
            'generate did not supply dimension information.')
      anchor_grid = to_normalized_coordinates(
          anchor_grid, im_height, im_width, check_range=False)
    anchor_grid_list.append(anchor_grid['boxes'])

  all_anchors = tf.concat(anchor_grid_list, 0)

  return all_anchors


# key: anchor_generator
BUILD_ANCHOR_FUNC = {'ssd_anchor_generator':  build_multiple_grid_anchors,
                     'multiscale_anchor_generator': build_multiscale_grid_anchors}
