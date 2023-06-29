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

# SPDX-License-Identifier: Apache-2.0

# ==============================================================================

import tensorflow as tf

from tf2onnx.tf_loader import *

from packaging.version import Version

def _tf_optimize_grappler(input_names, output_names, graph_def, constfold=True):
    from tensorflow.core.protobuf import meta_graph_pb2 as meta_graph_pb2, config_pb2, rewriter_config_pb2
    from tensorflow.python.grappler import tf_optimizer as tf_opt

    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    config.graph_options.infer_shapes = True
    # TODO: if we turn on pruning, grappler removes some identities that the tf-1.x lstm rewriter
    # depends on so for now don't turn this on, constfold is always enabled now.
    rewrite_options.optimizers[:] = [
        # 'pruning', 'constfold', 'arithmetic', 'dependency', 'function',
        #'constfold', 'function'

        # we make dependency always enabled
        'dependency', 'function'
    ]
    # other options can be selected as required
    if constfold:
        rewrite_options.optimizers.append('constfold')

    #if LooseVersion(tf.__version__) >= "2.5":
    if Version(tf.__version__) >= Version("2.5"):
        # This flag disables folding QDQ nodes around constants in the network (eg: around conv/FC weights)
        rewrite_options.experimental_disable_folding_quantization_emulation = True

    meta_graph = tf.compat.v1.train.export_meta_graph(graph_def=graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for t in input_names + output_names:
        fetch_collection.node_list.value.append(t)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)
    graph_def = tf_opt.OptimizeGraph(config, meta_graph)
    return graph_def


def _tf_optimize(input_names, output_names, graph_def, constfold=True):
    """Extract inference subgraph and optimize graph."""
    assert isinstance(input_names, list)
    assert isinstance(output_names, list)

    # TODO: is this needed ?
    '''
    needed_names = [utils.node_name(i) for i in input_names] + \
                   [utils.node_name(i) for i in output_names]
    graph_def = extract_sub_graph(graph_def, needed_names)
    '''

    #want_grappler = is_tf2() or LooseVersion(tf.__version__) >= "1.15"
    want_grappler = is_tf2() or Version(tf.__version__) >= Version("1.15")
    if want_grappler:
        graph_def = _tf_optimize_grappler(input_names, output_names, graph_def, constfold=constfold)
    else:
        # the older transform path
        from tensorflow.tools.graph_transforms import TransformGraph  # pylint: disable=redefined-outer-name
        transforms = [
            "fold_constants(ignore_errors=true)",
            "remove_attribute(attribute_name=_class)",  # remove node colocation attributes
            "fold_batch_norms",
            "fold_old_batch_norms",
        ]
        graph_def = TransformGraph(graph_def, input_names, output_names, transforms)

    return graph_def


def _from_function(func, input_names, output_names, large_model=False, constfold=True):
    if large_model:
        return convert_variables_to_constants_large_model(func)

    try:
        #if get_tf_version() < LooseVersion("2.2"):
        if Version(tf.__version__) < Version("2.2"):
            frozen_func = convert_variables_to_constants_v2(func, lower_control_flow=False)
        else:
            frozen_func = convert_variables_to_constants_v2(func, lower_control_flow=False, aggressive_inlining=True)
    except ValueError as e:
        if "incompatible with expected resource" in str(e):
            bad_graph_def = convert_variables_to_constants_large_model(func)
            logger.warning("TF freezing failed. Attempting to fix freezing errors.")
            graph_def = fix_freezing_errors(bad_graph_def)
        else:
            raise e
    else:
        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
    graph_def = fix_freezing_errors_part2(graph_def)

    # output_names = [i.name for i in frozen_func.outputs]
    with tf.Graph().as_default() as tf_graph:
        with tf_session(graph=tf_graph) as sess:
            tf.import_graph_def(graph_def, name='')
            input_names = inputs_without_resource(sess, input_names)
            graph_def = _tf_optimize(input_names, output_names, graph_def, constfold=constfold)
    return graph_def


def _from_keras(keras_model, input_names, output_names, constfold=True):
    """Load keras model - experimental for now."""
    from tensorflow.python import keras as _keras
    from tensorflow.python.eager import context
    from tensorflow.python.keras.saving import saving_utils as _saving_utils

    # Handles Keras when Eager mode is enabled.
    custom_objects = None
    with tf.device("/cpu:0"):
        if keras_model._distribution_strategy is None:
        #if tf.distribute.has_strategy() == False:
            # This clear session may cause a pop empty error when distribution strategy enabled
            _keras.backend.clear_session()
            _keras.backend.set_learning_phase(False)

        if context.executing_eagerly():
            #_keras.backend.clear_session()
            #_keras.backend.set_learning_phase(False)
            #keras_model = _keras.models.load_model(model_path, custom_objects)

            function = _saving_utils.trace_model_call(keras_model)
            concrete_func = function.get_concrete_function()
            # allow to pass inputs and outputs from caller if we don't want all of them
            input_names = [input_tensor.name for input_tensor in concrete_func.inputs
                           if input_tensor.dtype != tf.dtypes.resource]
            output_names = [output_tensor.name for output_tensor in concrete_func.outputs
                            if output_tensor.dtype != tf.dtypes.resource]
            frozen_graph = _from_function(concrete_func, input_names, output_names, constfold=constfold)
        else:
            # Handles Keras when Eager mode is disabled.
            #_keras.backend.clear_session()
            #_keras.backend.set_learning_phase(False)
            #keras_model = _keras.models.load_model(model_path, custom_objects)

            # allow to pass inputs and outputs from caller if we don't want all of them
            input_names = keras_model.inputs
            output_names = keras_model.outputs
            sess = _keras.backend.get_session()
            input_names = inputs_without_resource(sess, input_names)
            frozen_graph = freeze_session(sess, input_names=input_names, output_names=output_names)
            tf_reset_default_graph()
            with tf_session() as sess:
                frozen_graph = _tf_optimize(input_names, output_names, frozen_graph, constfold=constfold)
            tf_reset_default_graph()

    return frozen_graph, input_names, output_names


def graph_from_keras(keras_model, input_names, output_names, constfold=True):
    """Convert keras model to frozen graph by tracing model call.
       Note that keras model with multiple inputs should be merged into a List
    """
    frozen_graph, _, _ = _from_keras(keras_model, input_names, output_names, constfold=constfold)
    return frozen_graph


def graph_from_keras_subclass(keras_subclass_model, dataspec, constfold=True):
    """Convert keras subclassing model with multiple independent inputs to frozen graph by tracing model call.
       You should pass a dataspec, which lists shape, dtype and name for each input, to create input layers.
    """
    # Add some input layers and call the model
    inputs = {}
    for spec in dataspec:
        inputs[spec['name']] = tf.keras.layers.Input(shape=spec['shape'], dtype=spec['dtype'], name=spec['name'])
    outputs = keras_subclass_model(inputs)

    # Create a new functional model and wrapper the subclass model as a sublayer
    keras_functional_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    frozen_graph, _, _ = _from_keras(keras_functional_model,
                                     keras_functional_model.input_names,
                                     keras_functional_model.output_names,
                                     constfold=constfold)

    return frozen_graph
