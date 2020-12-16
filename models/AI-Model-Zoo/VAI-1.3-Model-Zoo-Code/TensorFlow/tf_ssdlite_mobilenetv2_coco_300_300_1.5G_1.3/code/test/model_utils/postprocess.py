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

"""Post-processing operations on detected boxes."""

import numpy as np
import tensorflow as tf


def compute_clip_window(resized_inputs_shape, true_image_shapes):
  if true_image_shapes is None:
    return tf.constant([0, 0, 1, 1], dtype=tf.float32)
  true_heights, true_widths, _ = tf.unstack(
      tf.to_float(true_image_shapes), axis=1)
  padded_height = tf.to_float(resized_inputs_shape[1])
  padded_width = tf.to_float(resized_inputs_shape[2])
  return tf.stack([tf.zeros_like(true_heights), tf.zeros_like(true_widths),
                   true_heights / padded_height, true_widths / padded_width], axis=1)

def score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale):
  """Create a function to scale logits then apply a Tensorflow function."""
  def score_converter_fn(logits):
    scaled_logits = tf.divide(logits, logit_scale, name='scale_logits')
    return tf_score_converter_fn(scaled_logits, name='convert_scores')
  score_converter_fn.__name__ = '%s_with_logit_scale' % (tf_score_converter_fn.__name__)
  return score_converter_fn

def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iterations=32, back_prop=True):
  """Runs map_fn as a (static) for loop when possible.

  This function rewrites the map_fn as an explicit unstack input -> for loop
  over function calls -> stack result combination.  This allows our graphs to
  be acyclic when the batch size is static.
  For comparison, see https://www.tensorflow.org/api_docs/python/tf/map_fn.

  Note that `static_or_dynamic_map_fn` currently is not *fully* interchangeable
  with the default tf.map_fn function as it does not accept nested inputs (only
  Tensors or lists of Tensors).  Likewise, the output of `fn` can only be a
  Tensor or list of Tensors.

  TODO(jonathanhuang): make this function fully interchangeable with tf.map_fn.

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same structure as elems. Its output must have the
      same structure as elems.
    elems: A tensor or list of tensors, each of which will
      be unpacked along their first dimension. The sequence of the
      resulting slices will be applied to fn.
    dtype:  (optional) The output type(s) of fn. If fn returns a structure of
      Tensors differing from the structure of elems, then dtype is not optional
      and must have the same structure as the output of fn.
    parallel_iterations: (optional) number of batch items to process in
      parallel.  This flag is only used if the native tf.map_fn is used
      and defaults to 32 instead of 10 (unlike the standard tf.map_fn default).
    back_prop: (optional) True enables support for back propagation.
      This flag is only used if the native tf.map_fn is used.

  Returns:
    A tensor or sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.
  Raises:
    ValueError: if `elems` a Tensor or a list of Tensors.
    ValueError: if `fn` does not return a Tensor or list of Tensors
  """
  if isinstance(elems, list):
    for elem in elems:
      if not isinstance(elem, tf.Tensor):
        raise ValueError('`elems` must be a Tensor or list of Tensors.')

    elem_shapes = [elem.shape.as_list() for elem in elems]
    # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
    # to all be the same size along the batch dimension.
    for elem_shape in elem_shapes:
      if (not elem_shape or not elem_shape[0]
          or elem_shape[0] != elem_shapes[0][0]):
        return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    arg_tuples = zip(*[tf.unstack(elem) for elem in elems])
    outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
  else:
    if not isinstance(elems, tf.Tensor):
      raise ValueError('`elems` must be a Tensor or list of Tensors.')
    elems_shape = elems.shape.as_list()
    if not elems_shape or not elems_shape[0]:
      return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
    outputs = [fn(arg) for arg in tf.unstack(elems)]
  # Stack `outputs`, which is a list of Tensors or list of lists of Tensors
  if all([isinstance(output, tf.Tensor) for output in outputs]):
    return tf.stack(outputs)
  else:
    if all([isinstance(output, list) for output in outputs]):
      if all([all(
          [isinstance(entry, tf.Tensor) for entry in output_list])
              for output_list in outputs]):
        return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
  raise ValueError('`fn` should return a Tensor or a list of Tensors.')


def pad_or_clip_box_list(boxlist, num_boxes, scope=None):
  """Pads or clips all fields of a BoxList.

  Args:
    boxlist: A BoxList with arbitrary of number of boxes.
    num_boxes: First num_boxes in boxlist are kept.
      The fields are zero-padded if num_boxes is bigger than the
      actual number of boxes.
    scope: name scope.

  Returns:
    BoxList with all fields padded or clipped.
  """
  with tf.name_scope(scope, 'PadOrClipBoxList'):
    subboxlist = {}
    for key, val in boxlist.items():
      subboxlist[key] = pad_or_clip_tensor(val, num_boxes)
    return subboxlist


def pad_or_clip_tensor(t, length):
  """Pad or clip the input tensor along the first dimension.

  Args:
    t: the input tensor, assuming the rank is at least 1.
    length: a tensor of shape [1]  or an integer, indicating the first dimension
      of the input tensor t after processing.

  Returns:
    processed_t: the processed tensor, whose first dimension is length. If the
      length is an integer, the first dimension of the processed tensor is set
      to length statically.
  """
  return pad_or_clip_nd(t, [length] + t.shape.as_list()[1:])


def pad_or_clip_nd(tensor, output_shape):
  """Pad or Clip given tensor to the output shape.

  Args:
    tensor: Input tensor to pad or clip.
    output_shape: A list of integers / scalar tensors (or None for dynamic dim)
      representing the size to pad or clip each dimension of the input tensor.

  Returns:
    Input tensor padded and clipped to the output shape.
  """
  tensor_shape = tf.shape(tensor)
  clip_size = [
      tf.where(tensor_shape[i] - shape > 0, shape, -1)
      if shape is not None else -1 for i, shape in enumerate(output_shape)
  ]
  clipped_tensor = tf.slice(
      tensor,
      begin=tf.zeros(len(clip_size), dtype=tf.int32),
      size=clip_size)

  # Pad tensor if the shape of clipped tensor is smaller than the expected
  # shape.
  clipped_tensor_shape = tf.shape(clipped_tensor)
  trailing_paddings = [
      shape - clipped_tensor_shape[i] if shape is not None else 0
      for i, shape in enumerate(output_shape)
  ]
  paddings = tf.stack(
      [
          tf.zeros(len(trailing_paddings), dtype=tf.int32),
          trailing_paddings
      ],
      axis=1)
  padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
  output_static_shape = [
      dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
  ]
  padded_tensor.set_shape(output_static_shape)
  return padded_tensor


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


def matmul_gather_on_zeroth_axis(params, indices, scope=None):
  """Matrix multiplication based implementation of tf.gather on zeroth axis.

  TODO(rathodv, jonathanhuang): enable sparse matmul option.

  Args:
    params: A float32 Tensor. The tensor from which to gather values.
      Must be at least rank 1.
    indices: A Tensor. Must be one of the following types: int32, int64.
      Must be in range [0, params.shape[0])
    scope: A name for the operation (optional).

  Returns:
    A Tensor. Has the same type as params. Values from params gathered
    from indices given by indices, with shape indices.shape + params.shape[1:].
  """
  with tf.name_scope(scope, 'MatMulGather'):
    params_shape = combined_static_and_dynamic_shape(params)
    indices_shape = combined_static_and_dynamic_shape(indices)
    params2d = tf.reshape(params, [params_shape[0], -1])
    indicator_matrix = tf.one_hot(indices, params_shape[0])
    gathered_result_flattened = tf.matmul(indicator_matrix, params2d)
    return tf.reshape(gathered_result_flattened,
                      tf.stack(indices_shape + params_shape[1:]))


def dict_gather(boxlist, indices, fields=None, scope=None, use_static_shapes=False):
  """Gather boxes from BoxList according to indices and return new BoxList.

  By default, `gather` returns boxes corresponding to the input index list, as
  well as all additional fields stored in the boxlist (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.

  Args:
    boxlist: BoxList holding N boxes
    indices: a rank-1 tensor of type int32 / int64
    fields: (optional) list of fields to also gather from.  If None (default),
      all fields are gathered from.  Pass an empty fields list to only gather
      the box coordinates.
    scope: name scope.
    use_static_shapes: Whether to use an implementation with static shape
      gurantees.

  Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
    specified by indices
  Raises:
    ValueError: if specified field is not contained in boxlist or if the
      indices are not of type int32
  """
  with tf.name_scope(scope, 'Gather'):
    if len(indices.shape.as_list()) != 1:
      raise ValueError('indices should have rank 1')
    if indices.dtype != tf.int32 and indices.dtype != tf.int64:
      raise ValueError('indices should be an int32 / int64 tensor')
    gather_op = tf.gather
    if use_static_shapes:
      gather_op = matmul_gather_on_zeroth_axis
    subboxlist = {}
    subboxlist['boxes'] = gather_op(boxlist['boxes'], indices)
    if fields is None:
      fields = [k for k in boxlist.keys() if k != 'boxes']
    fields += ['boxes']
    for field in fields:
      if field not in boxlist.keys():
        raise ValueError('boxlist must contain all specified fields')
      subfieldlist = gather_op(boxlist[field], indices)
      subboxlist[field] = subfieldlist
    return subboxlist


def dict_concatenate(boxlists, fields=None, scope=None):
  """Concatenate list of BoxLists.

  This op concatenates a list of input BoxLists into a larger BoxList.  It also
  handles concatenation of BoxList fields as long as the field tensor shapes
  are equal except for the first dimension.

  Args:
    boxlists: list of BoxList objects
    fields: optional list of fields to also concatenate.  By default, all
      fields from the first BoxList in the list are included in the
      concatenation.
    scope: name scope.

  Returns:
    a BoxList with number of boxes equal to
      sum([boxlist.num_boxes() for boxlist in BoxList])
  Raises:
    ValueError: if boxlists is invalid (i.e., is not a list, is empty, or
      contains non BoxList objects), or if requested fields are not contained in
      all boxlists
  """
  with tf.name_scope(scope, 'Concatenate'):
    if not isinstance(boxlists, list):
      raise ValueError('boxlists should be a list')
    if not boxlists:
      raise ValueError('boxlists should have nonzero length')
    concatenated = {}
    concatenated['boxes'] = tf.concat([boxlist['boxes'] for boxlist in boxlists], 0)
    if fields is None:
      fields = [k for k in boxlists[0].keys() if k != 'boxes']
    for field in fields:
      first_field_shape = boxlists[0][field].get_shape().as_list()
      first_field_shape[0] = -1
      if None in first_field_shape:
        raise ValueError('field %s must have fully defined shape except for the'
                         ' 0th dimension.' % field)
      for boxlist in boxlists:
        if field not in boxlist.keys():
          raise ValueError('boxlist must contain all requested fields')
        field_shape = boxlist[field].get_shape().as_list()
        field_shape[0] = -1
        if field_shape != first_field_shape:
          raise ValueError('field %s must have same shape for all boxlists '
                           'except for the 0th dimension.' % field)
      concatenated_field = tf.concat(
          [boxlist[field] for boxlist in boxlists], 0)
      concatenated[field] = concatenated_field
    return concatenated

class SortOrder(object):
  """Enum class for sort order.

  Attributes:
    ascend: ascend order.
    descend: descend order.
  """
  ascend = 1
  descend = 2

def sort_by_field(boxlist, field, order=SortOrder.descend, scope=None):
  """Sort boxes and associated fields according to a scalar field.

  A common use case is reordering the boxes according to descending scores.

  Args:
    boxlist: BoxList holding N boxes.
    field: A BoxList field for sorting and reordering the BoxList.
    order: (Optional) descend or ascend. Default is descend.
    scope: name scope.

  Returns:
    sorted_boxlist: A sorted BoxList with the field in the specified order.

  Raises:
    ValueError: if specified field does not exist
    ValueError: if the order is not either descend or ascend
  """
  with tf.name_scope(scope, 'SortByField'):
    if order != SortOrder.descend and order != SortOrder.ascend:
      raise ValueError('Invalid sort order')

    field_to_sort = boxlist[field]
    if len(field_to_sort.shape.as_list()) != 1:
      raise ValueError('Field should have rank 1')

    num_boxes = tf.shape(boxlist['boxes'])[0]
    num_entries = tf.size(field_to_sort)
    length_assert = tf.Assert(
        tf.equal(num_boxes, num_entries),
        ['Incorrect field size: actual vs expected.', num_entries, num_boxes])

    with tf.control_dependencies([length_assert]):
      _, sorted_indices = tf.nn.top_k(field_to_sort, num_boxes, sorted=True)

    if order == SortOrder.ascend:
      sorted_indices = tf.reverse_v2(sorted_indices, [0])

    return dict_gather(boxlist, sorted_indices)

def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
  """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

  Args:
    boxlist_to_copy_to: BoxList to which extra fields are copied.
    boxlist_to_copy_from: BoxList from which fields are copied.

  Returns:
    boxlist_to_copy_to with extra fields.
  """
  # for field in boxlist_to_copy_from.get_extra_fields():
  extra_fields = [k for k in boxlist_to_copy_from.keys() if k != 'boxes']
  for field in extra_fields:
    boxlist_to_copy_to[field] = boxlist_to_copy_from[field]
  return boxlist_to_copy_to

def area(boxlist, scope=None):
  """Computes area of boxes.

  Args:
    boxlist: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
  with tf.name_scope(scope, 'Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist['boxes'], num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

def clip_to_window(boxlist, window, filter_nonoverlapping=True, scope=None):
  """Clip bounding boxes to a window.

  This op clips any input bounding boxes (represented by bounding box
  corners) to a window, optionally filtering out boxes that do not
  overlap at all with the window.

  Args:
    boxlist: BoxList holding M_in boxes
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window to which the op should clip boxes.
    filter_nonoverlapping: whether to filter out boxes that do not overlap at
      all with the window.
    scope: name scope.

  Returns:
    a BoxList holding M_out boxes where M_out <= M_in
  """
  with tf.name_scope(scope, 'ClipToWindow'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist['boxes'], num_or_size_splits=4, axis=1)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
    y_min_clipped = tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
    y_max_clipped = tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
    x_min_clipped = tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
    x_max_clipped = tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
    clipped = {}
    clipped['boxes'] = tf.concat([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped], 1)
    clipped = _copy_extra_fields(clipped, boxlist)
    if filter_nonoverlapping:
      areas = area(clipped)
      nonzero_area_indices = tf.cast(
          tf.reshape(tf.where(tf.greater(areas, 0.0)), [-1]), tf.int32)
      clipped = dict_gather(clipped, nonzero_area_indices)
    return clipped

def scale(boxlist, y_scale, x_scale, scope=None):
  """scale box coordinates in x and y dimensions.

  Args:
    boxlist: BoxList holding N boxes
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    boxlist: BoxList holding N boxes
  """
  with tf.name_scope(scope, 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist['boxes'], num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = {}
    scaled_boxlist['boxes'] = tf.concat([y_min, x_min, y_max, x_max], 1)
    return _copy_extra_fields(scaled_boxlist, boxlist)

def to_normalized_coordinates(boxlist, height, width,
                              check_range=True, scope=None):
  """Converts absolute box coordinates to normalized coordinates in [0, 1].

  Usually one uses the dynamic shape of the image or conv-layer tensor:
    boxlist = box_list_ops.to_normalized_coordinates(boxlist,
                                                     tf.shape(images)[1],
                                                     tf.shape(images)[2]),

  This function raises an assertion failed error at graph execution time when
  the maximum coordinate is smaller than 1.01 (which means that coordinates are
  already normalized). The value 1.01 is to deal with small rounding errors.

  Args:
    boxlist: BoxList with coordinates in terms of pixel-locations.
    height: Maximum value for height of absolute box coordinates.
    width: Maximum value for width of absolute box coordinates.
    check_range: If True, checks if the coordinates are normalized or not.
    scope: name scope.

  Returns:
    boxlist with normalized coordinates in [0, 1].
  """
  with tf.name_scope(scope, 'ToNormalizedCoordinates'):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    if check_range:
      max_val = tf.reduce_max(boxlist['boxes'])
      max_assert = tf.Assert(tf.greater(max_val, 1.01),
                             ['max value is lower than 1.01: ', max_val])
      with tf.control_dependencies([max_assert]):
        width = tf.identity(width)

    return scale(boxlist, 1 / height, 1 / width)

def change_coordinate_frame(boxlist, window, scope=None):
  """Change coordinate frame of the boxlist to be relative to window's frame.

  Given a window of the form [ymin, xmin, ymax, xmax],
  changes bounding box coordinates from boxlist to be relative to this window
  (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

  An example use case is data augmentation: where we are given groundtruth
  boxes (boxlist) and would like to randomly crop the image to some
  window (window). In this case we need to change the coordinate frame of
  each groundtruth box to be relative to this new window.

  Args:
    boxlist: A BoxList object holding N boxes.
    window: A rank 1 tensor [4].
    scope: name scope.

  Returns:
    Returns a BoxList object with N boxes.
  """
  with tf.name_scope(scope, 'ChangeCoordinateFrame'):
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    boxlist_tmp = {}
    boxlist_tmp['boxes'] = boxlist['boxes'] - [window[0], window[1], window[0], window[1]]
    boxlist_new = scale(boxlist_tmp, 1.0 / win_height, 1.0 / win_width)
    boxlist_new = _copy_extra_fields(boxlist_new, boxlist)
    return boxlist_new

def multiclass_non_max_suppression(boxes,
                                   scores,
                                   score_thresh,
                                   iou_thresh,
                                   max_size_per_class,
                                   max_total_size=0,
                                   clip_window=None,
                                   change_coordinate_frame=False,
                                   masks=None,
                                   boundaries=None,
                                   pad_to_max_output_size=False,
                                   additional_fields=None,
                                   scope=None):
  """Multi-class version of non maximum suppression.

  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes.  It operates independently for each class for
  which scores are provided (via the scores field of the input box_list),
  pruning boxes with score less than a provided threshold prior to
  applying NMS.

  Please note that this operation is performed on *all* classes, therefore any
  background classes should be removed prior to calling this function.

  Selected boxes are guaranteed to be sorted in decreasing order by score (but
  the sort is not guaranteed to be stable).

  Args:
    boxes: A [k, q, 4] float32 tensor containing k detections. `q` can be either
      number of classes or 1 depending on whether a separate box is predicted
      per class.
    scores: A [k, num_classes] float32 tensor containing the scores for each of
      the k detections. The scores have to be non-negative when
      pad_to_max_output_size is True.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_size_per_class: maximum number of retained boxes per class.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.
    clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
      representing the window to clip and normalize boxes to before performing
      non-max suppression.
    change_coordinate_frame: Whether to normalize coordinates after clipping
      relative to clip_window (this can only be set to True if a clip_window
      is provided)
    masks: (optional) a [k, q, mask_height, mask_width] float32 tensor
      containing box masks. `q` can be either number of classes or 1 depending
      on whether a separate mask is predicted per class.
    boundaries: (optional) a [k, q, boundary_height, boundary_width] float32
      tensor containing box boundaries. `q` can be either number of classes or 1
      depending on whether a separate boundary is predicted per class.
    pad_to_max_output_size: If true, the output nmsed boxes are padded to be of
      length `max_size_per_class`. Defaults to false.
    additional_fields: (optional) If not None, a dictionary that maps keys to
      tensors whose first dimensions are all of size `k`. After non-maximum
      suppression, all tensors corresponding to the selected boxes will be
      added to resulting BoxList.
    scope: name scope.

  Returns:
    A tuple of sorted_boxes and num_valid_nms_boxes. The sorted_boxes is a
      BoxList holds M boxes with a rank-1 scores field representing
      corresponding scores for each box with scores sorted in decreasing order
      and a rank-1 classes field representing a class label for each box. The
      num_valid_nms_boxes is a 0-D integer tensor representing the number of
      valid elements in `BoxList`, with the valid elements appearing first.

  Raises:
    ValueError: if iou_thresh is not in [0, 1] or if input boxlist does not have
      a valid scores field.
  """
  if not 0 <= iou_thresh <= 1.0:
    raise ValueError('iou_thresh must be between 0 and 1')
  if scores.shape.ndims != 2:
    raise ValueError('scores field must be of rank 2')
  if scores.shape[1].value is None:
    raise ValueError('scores must have statically defined second '
                     'dimension')
  if boxes.shape.ndims != 3:
    raise ValueError('boxes must be of rank 3.')
  if not (boxes.shape[1].value == scores.shape[1].value or
          boxes.shape[1].value == 1):
    raise ValueError('second dimension of boxes must be either 1 or equal '
                     'to the second dimension of scores')
  if boxes.shape[2].value != 4:
    raise ValueError('last dimension of boxes must be of size 4.')
  if change_coordinate_frame and clip_window is None:
    raise ValueError('if change_coordinate_frame is True, then a clip_window'
                     'must be specified.')

  with tf.name_scope(scope, 'MultiClassNonMaxSuppression'):
    num_scores = tf.shape(scores)[0]
    num_classes = scores.get_shape()[1]

    selected_boxes_list = []
    num_valid_nms_boxes_cumulative = tf.constant(0)
    per_class_boxes_list = tf.unstack(boxes, axis=1)
    if masks is not None:
      per_class_masks_list = tf.unstack(masks, axis=1)
    if boundaries is not None:
      per_class_boundaries_list = tf.unstack(boundaries, axis=1)
    boxes_ids = (range(num_classes) if len(per_class_boxes_list) > 1
                 else [0] * num_classes.value)
    for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
      per_class_boxes = per_class_boxes_list[boxes_idx]
      boxlist_and_class_scores = {}
      boxlist_and_class_scores['boxes'] = per_class_boxes
      class_scores = tf.reshape(
          tf.slice(scores, [0, class_idx], tf.stack([num_scores, 1])), [-1])

      boxlist_and_class_scores['scores'] = class_scores
      if masks is not None:
        per_class_masks = per_class_masks_list[boxes_idx]
        boxlist_and_class_scores['masks'] = per_class_masks
      if boundaries is not None:
        per_class_boundaries = per_class_boundaries_list[boxes_idx]
        boxlist_and_class_scores['boundaries'] = per_class_boundaries
      if additional_fields is not None:
        for key, tensor in additional_fields.items():
          boxlist_and_class_scores['key'] = tensor

      if pad_to_max_output_size:
        max_selection_size = max_size_per_class
        selected_indices, num_valid_nms_boxes = (
            tf.image.non_max_suppression_padded(
                boxlist_and_class_scores['boxes'],
                boxlist_and_class_scores['scores'],
                max_selection_size,
                iou_threshold=iou_thresh,
                score_threshold=score_thresh,
                pad_to_max_output_size=True))
      else:
        max_selection_size = tf.minimum(max_size_per_class,
                                        tf.shape(boxlist_and_class_scores['boxes'])[0])
        selected_indices = tf.image.non_max_suppression(
            boxlist_and_class_scores['boxes'],
            boxlist_and_class_scores['scores'],
            max_selection_size,
            iou_threshold=iou_thresh,
            score_threshold=score_thresh)
        num_valid_nms_boxes = tf.shape(selected_indices)[0]
        selected_indices = tf.concat(
            [selected_indices,
             tf.zeros(max_selection_size-num_valid_nms_boxes, tf.int32)], 0)
      nms_result = dict_gather(boxlist_and_class_scores,
                                       selected_indices)
      # Make the scores -1 for invalid boxes.
      valid_nms_boxes_indx = tf.less(
          tf.range(max_selection_size), num_valid_nms_boxes)
      nms_scores = nms_result['scores']
      nms_result['scores'] = tf.where(valid_nms_boxes_indx,
                                    nms_scores, -1*tf.ones(max_selection_size))
      num_valid_nms_boxes_cumulative += num_valid_nms_boxes

      nms_result['classes'] = tf.zeros_like(nms_result['scores']) + class_idx
      selected_boxes_list.append(nms_result)
    selected_boxes = dict_concatenate(selected_boxes_list)
    sorted_boxes = sort_by_field(selected_boxes, 'scores')
    if clip_window is not None:
      # When pad_to_max_output_size is False, it prunes the boxes with zero area.
      sorted_boxes = clip_to_window(
          sorted_boxes,
          clip_window,
          filter_nonoverlapping=not pad_to_max_output_size)
      # Set the scores of boxes with zero area to -1 to keep the default
      # behaviour of pruning out zero area boxes.
      sorted_boxes_size = tf.shape(sorted_boxes['boxes'])[0]
      non_zero_box_area = tf.cast(area(sorted_boxes), tf.bool)
      sorted_boxes_scores = tf.where(
          non_zero_box_area,
          sorted_boxes['scores'],
          -1*tf.ones(sorted_boxes_size))
      sorted_boxes['scores'] = sorted_boxes_scores
      num_valid_nms_boxes_cumulative = tf.reduce_sum(
          tf.cast(tf.greater_equal(sorted_boxes_scores, 0), tf.int32))
      sorted_boxes = sort_by_field(sorted_boxes, 'scores')
      if change_coordinate_frame:
        sorted_boxes = change_coordinate_frame(
            sorted_boxes, clip_window)

    if max_total_size:
      max_total_size = tf.minimum(max_total_size,
                                  tf.shape(sorted_boxes['boxes'])[0])
      sorted_boxes = dict_gather(sorted_boxes,
                                         tf.range(max_total_size))
      num_valid_nms_boxes_cumulative = tf.where(
          max_total_size > num_valid_nms_boxes_cumulative,
          num_valid_nms_boxes_cumulative, max_total_size)
    # Select only the valid boxes if pad_to_max_output_size is False.
    if not pad_to_max_output_size:
      sorted_boxes = dict_gather(
          sorted_boxes, tf.range(num_valid_nms_boxes_cumulative))

    return sorted_boxes, num_valid_nms_boxes_cumulative

def batch_multiclass_non_max_suppression(boxes,
                                         scores,
                                         score_thresh,
                                         iou_thresh,
                                         max_size_per_class,
                                         max_total_size=0,
                                         clip_window=None,
                                         change_coordinate_frame=False,
                                         num_valid_boxes=None,
                                         masks=None,
                                         additional_fields=None,
                                         scope=None,
                                         use_static_shapes=False,
                                         parallel_iterations=32):
  """Multi-class version of non maximum suppression that operates on a batch.

  This op is similar to `multiclass_non_max_suppression` but operates on a batch
  of boxes and scores. See documentation for `multiclass_non_max_suppression`
  for details.

  Args:
    boxes: A [batch_size, num_anchors, q, 4] float32 tensor containing
      detections. If `q` is 1 then same boxes are used for all classes
        otherwise, if `q` is equal to number of classes, class-specific boxes
        are used.
    scores: A [batch_size, num_anchors, num_classes] float32 tensor containing
      the scores for each of the `num_anchors` detections. The scores have to be
      non-negative when use_static_shapes is set True.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_size_per_class: maximum number of retained boxes per class.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.
    clip_window: A float32 tensor of shape [batch_size, 4]  where each entry is
      of the form [y_min, x_min, y_max, x_max] representing the window to clip
      boxes to before performing non-max suppression. This argument can also be
      a tensor of shape [4] in which case, the same clip window is applied to
      all images in the batch. If clip_widow is None, all boxes are used to
      perform non-max suppression.
    change_coordinate_frame: Whether to normalize coordinates after clipping
      relative to clip_window (this can only be set to True if a clip_window
      is provided)
    num_valid_boxes: (optional) a Tensor of type `int32`. A 1-D tensor of shape
      [batch_size] representing the number of valid boxes to be considered
      for each image in the batch.  This parameter allows for ignoring zero
      paddings.
    masks: (optional) a [batch_size, num_anchors, q, mask_height, mask_width]
      float32 tensor containing box masks. `q` can be either number of classes
      or 1 depending on whether a separate mask is predicted per class.
    additional_fields: (optional) If not None, a dictionary that maps keys to
      tensors whose dimensions are [batch_size, num_anchors, ...].
    scope: tf scope name.
    use_static_shapes: If true, the output nmsed boxes are padded to be of
      length `max_size_per_class` and it doesn't clip boxes to max_total_size.
      Defaults to false.
    parallel_iterations: (optional) number of batch items to process in
      parallel.

  Returns:
    'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    'nmsed_scores': A [batch_size, max_detections] float32 tensor containing
      the scores for the boxes.
    'nmsed_classes': A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    'nmsed_masks': (optional) a
      [batch_size, max_detections, mask_height, mask_width] float32 tensor
      containing masks for each selected box. This is set to None if input
      `masks` is None.
    'nmsed_additional_fields': (optional) a dictionary of
      [batch_size, max_detections, ...] float32 tensors corresponding to the
      tensors specified in the input `additional_fields`. This is not returned
      if input `additional_fields` is None.
    'num_detections': A [batch_size] int32 tensor indicating the number of
      valid detections per batch item. Only the top num_detections[i] entries in
      nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
      entries are zero paddings.

  Raises:
    ValueError: if `q` in boxes.shape is not 1 or not equal to number of
      classes as inferred from scores.shape.
  """
  q = boxes.shape[2].value
  num_classes = scores.shape[2].value
  if q != 1 and q != num_classes:
    raise ValueError('third dimension of boxes must be either 1 or equal '
                     'to the third dimension of scores')
  if change_coordinate_frame and clip_window is None:
    raise ValueError('if change_coordinate_frame is True, then a clip_window'
                     'must be specified.')
  original_masks = masks
  original_additional_fields = additional_fields
  with tf.name_scope(scope, 'BatchMultiClassNonMaxSuppression'):
    boxes_shape = boxes.shape
    batch_size = boxes_shape[0].value
    num_anchors = boxes_shape[1].value

    if batch_size is None:
      batch_size = tf.shape(boxes)[0]
    if num_anchors is None:
      num_anchors = tf.shape(boxes)[1]

    # If num valid boxes aren't provided, create one and mark all boxes as
    # valid.
    if num_valid_boxes is None:
      num_valid_boxes = tf.ones([batch_size], dtype=tf.int32) * num_anchors

    # If masks aren't provided, create dummy masks so we can only have one copy
    # of _single_image_nms_fn and discard the dummy masks after map_fn.
    if masks is None:
      masks_shape = tf.stack([batch_size, num_anchors, q, 1, 1])
      masks = tf.zeros(masks_shape)

    if clip_window is None:
      clip_window = tf.stack([
          tf.reduce_min(boxes[:, :, :, 0]),
          tf.reduce_min(boxes[:, :, :, 1]),
          tf.reduce_max(boxes[:, :, :, 2]),
          tf.reduce_max(boxes[:, :, :, 3])
      ])
    if clip_window.shape.ndims == 1:
      clip_window = tf.tile(tf.expand_dims(clip_window, 0), [batch_size, 1])

    if additional_fields is None:
      additional_fields = {}

    def _single_image_nms_fn(args):
      """Runs NMS on a single image and returns padded output.

      Args:
        args: A list of tensors consisting of the following:
          per_image_boxes - A [num_anchors, q, 4] float32 tensor containing
            detections. If `q` is 1 then same boxes are used for all classes
            otherwise, if `q` is equal to number of classes, class-specific
            boxes are used.
          per_image_scores - A [num_anchors, num_classes] float32 tensor
            containing the scores for each of the `num_anchors` detections.
          per_image_masks - A [num_anchors, q, mask_height, mask_width] float32
            tensor containing box masks. `q` can be either number of classes
            or 1 depending on whether a separate mask is predicted per class.
          per_image_clip_window - A 1D float32 tensor of the form
            [ymin, xmin, ymax, xmax] representing the window to clip the boxes
            to.
          per_image_additional_fields - (optional) A variable number of float32
            tensors each with size [num_anchors, ...].
          per_image_num_valid_boxes - A tensor of type `int32`. A 1-D tensor of
            shape [batch_size] representing the number of valid boxes to be
            considered for each image in the batch.  This parameter allows for
            ignoring zero paddings.

      Returns:
        'nmsed_boxes': A [max_detections, 4] float32 tensor containing the
          non-max suppressed boxes.
        'nmsed_scores': A [max_detections] float32 tensor containing the scores
          for the boxes.
        'nmsed_classes': A [max_detections] float32 tensor containing the class
          for boxes.
        'nmsed_masks': (optional) a [max_detections, mask_height, mask_width]
          float32 tensor containing masks for each selected box. This is set to
          None if input `masks` is None.
        'nmsed_additional_fields':  (optional) A variable number of float32
          tensors each with size [max_detections, ...] corresponding to the
          input `per_image_additional_fields`.
        'num_detections': A [batch_size] int32 tensor indicating the number of
          valid detections per batch item. Only the top num_detections[i]
          entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The
          rest of the entries are zero paddings.
      """
      per_image_boxes = args[0]
      per_image_scores = args[1]
      per_image_masks = args[2]
      per_image_clip_window = args[3]
      per_image_additional_fields = {
          key: value
          for key, value in zip(additional_fields, args[4:-1])
      }
      per_image_num_valid_boxes = args[-1]
      if use_static_shapes:
        total_proposals = tf.shape(per_image_scores)
        per_image_scores = tf.where(
            tf.less(tf.range(total_proposals[0]), per_image_num_valid_boxes),
            per_image_scores,
            tf.fill(total_proposals, np.finfo('float32').min))
      else:
        per_image_boxes = tf.reshape(
            tf.slice(per_image_boxes, 3 * [0],
                     tf.stack([per_image_num_valid_boxes, -1, -1])), [-1, q, 4])
        per_image_scores = tf.reshape(
            tf.slice(per_image_scores, [0, 0],
                     tf.stack([per_image_num_valid_boxes, -1])),
            [-1, num_classes])
        per_image_masks = tf.reshape(
            tf.slice(per_image_masks, 4 * [0],
                     tf.stack([per_image_num_valid_boxes, -1, -1, -1])),
            [-1, q, per_image_masks.shape[2].value,
             per_image_masks.shape[3].value])
        if per_image_additional_fields is not None:
          for key, tensor in per_image_additional_fields.items():
            additional_field_shape = tensor.get_shape()
            additional_field_dim = len(additional_field_shape)
            per_image_additional_fields[key] = tf.reshape(
                tf.slice(per_image_additional_fields[key],
                         additional_field_dim * [0],
                         tf.stack([per_image_num_valid_boxes] +
                                  (additional_field_dim - 1) * [-1])),
                [-1] + [dim.value for dim in additional_field_shape[1:]])

      nmsed_boxlist, num_valid_nms_boxes = multiclass_non_max_suppression(
          per_image_boxes,
          per_image_scores,
          score_thresh,
          iou_thresh,
          max_size_per_class,
          max_total_size,
          clip_window=per_image_clip_window,
          change_coordinate_frame=change_coordinate_frame,
          masks=per_image_masks,
          pad_to_max_output_size=use_static_shapes,
          additional_fields=per_image_additional_fields)

      if not use_static_shapes:
        nmsed_boxlist = pad_or_clip_box_list(
            nmsed_boxlist, max_total_size)
      num_detections = num_valid_nms_boxes
      nmsed_boxes = nmsed_boxlist['boxes']
      nmsed_scores = nmsed_boxlist['scores']
      nmsed_classes = nmsed_boxlist['classes']
      nmsed_masks = nmsed_boxlist['masks']
      nmsed_additional_fields = [
          nmsed_boxlist[key] for key in per_image_additional_fields if key not in ['boxes', 'scores', 'classes', 'masks']
      ]
      return ([nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks] +
              nmsed_additional_fields + [num_detections])

    num_additional_fields = 0
    if additional_fields is not None:
      num_additional_fields = len(additional_fields)
    num_nmsed_outputs = 4 + num_additional_fields

    batch_outputs = static_or_dynamic_map_fn(
        _single_image_nms_fn,
        elems=([boxes, scores, masks, clip_window] +
               list(additional_fields.values()) + [num_valid_boxes]),
        dtype=(num_nmsed_outputs * [tf.float32] + [tf.int32]),
        parallel_iterations=parallel_iterations)

    batch_nmsed_boxes = batch_outputs[0]
    batch_nmsed_scores = batch_outputs[1]
    batch_nmsed_classes = batch_outputs[2]
    batch_nmsed_masks = batch_outputs[3]
    batch_nmsed_additional_fields = {
        key: value
        for key, value in zip(additional_fields, batch_outputs[4:-1])
    }
    batch_num_detections = batch_outputs[-1]

    if original_masks is None:
      batch_nmsed_masks = None

    if original_additional_fields is None:
      batch_nmsed_additional_fields = None

    return (batch_nmsed_boxes, batch_nmsed_scores, batch_nmsed_classes,
            batch_nmsed_masks, batch_nmsed_additional_fields,
            batch_num_detections)
