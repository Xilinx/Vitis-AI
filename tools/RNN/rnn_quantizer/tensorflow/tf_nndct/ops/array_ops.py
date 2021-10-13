from tensorflow.python.ops import array_ops

# Remove the deprecated argument 'validate_indices' from orginal signature
# of tf.gather so that the generated code by writer can match the correct
# argument. When we call
#   tf_nndct.ops.gather(t0, t1, t2),
# actually we will call:
#   tf.gather(t0, t1, axis=t2)
# See https://www.tensorflow.org/api_docs/python/tf/gather
def gather(params,
           indices,
           axis=None,
           batch_dims=0,
           name=None):
  return array_ops.gather_v2(
      params,
      indices,
      axis=axis,
      batch_dims=batch_dims,
      name=name)
