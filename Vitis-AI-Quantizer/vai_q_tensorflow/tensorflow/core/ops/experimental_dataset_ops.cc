/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("AssertNextDataset")
    .Input("input_dataset: variant")
    .Input("transformations: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // transformations should be a vector.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalAssertNextDataset")
    .Input("input_dataset: variant")
    .Input("transformations: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // transformations should be a vector.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("AutoShardDataset")
    .Input("input_dataset: variant")
    .Input("num_workers: int64")
    .Input("index: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalAutoShardDataset")
    .Input("input_dataset: variant")
    .Input("num_workers: int64")
    .Input("index: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("BytesProducedStatsDataset")
    .Input("input_dataset: variant")
    .Input("tag: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle tag_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &tag_shape));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalBytesProducedStatsDataset")
    .Input("input_dataset: variant")
    .Input("tag: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle tag_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &tag_shape));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ChooseFastestBranchDataset")
    .Input("input_dataset: variant")
    .Input("ratio_numerator: int64")
    .Input("ratio_denominator: int64")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("Targuments: list(type) >= 0")
    .Attr("num_elements_per_branch: int >= 1")
    .Attr("branches: list(func) >= 1")
    .Attr("other_arguments_lengths: list(int) >= 1")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ChooseFastestDataset")
    .Input("input_datasets: N * variant")
    .Output("handle: variant")
    .Attr("N: int >= 2")
    .Attr("num_experiments: int")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalChooseFastestDataset")
    .Input("input_datasets: N * variant")
    .Output("handle: variant")
    .Attr("N: int >= 2")
    .Attr("num_experiments: int")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("CSVDataset")
    .Input("filenames: string")
    .Input("compression_type: string")
    .Input("buffer_size: int64")
    .Input("header: bool")
    .Input("field_delim: string")
    .Input("use_quote_delim: bool")
    .Input("na_value: string")
    .Input("select_cols: int64")
    .Input("record_defaults: output_types")
    .Output("handle: variant")
    .Attr("output_types: list({float,double,int32,int64,string}) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // `filenames` must be a scalar or a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      // `compression_type`, `buffer_size`, `header`, `field_delim`,
      // `use_quote_delim`, `na_value` must be scalars
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      // `select_cols` must be a vector
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &unused));
      // `record_defaults` must be lists of scalars
      for (size_t i = 8; i < c->num_inputs(); ++i) {
        shape_inference::ShapeHandle v;
        TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(i), 1, &v));
        if (c->Rank(c->input(i)) == 1 && c->Value(c->Dim(v, 0)) > 1) {
          return errors::InvalidArgument(
              "Shape of a default must be a length-0 or length-1 vector, or a "
              "scalar.");
        }
      }
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalCSVDataset")
    .Input("filenames: string")
    .Input("compression_type: string")
    .Input("buffer_size: int64")
    .Input("header: bool")
    .Input("field_delim: string")
    .Input("use_quote_delim: bool")
    .Input("na_value: string")
    .Input("select_cols: int64")
    .Input("record_defaults: output_types")
    .Output("handle: variant")
    .Attr("output_types: list({float,double,int32,int64,string}) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // `filenames` must be a scalar or a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      // `compression_type`, `buffer_size`, `header`, `field_delim`,
      // `use_quote_delim`, `na_value` must be scalars
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      // `select_cols` must be a vector
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 1, &unused));
      // `record_defaults` must be lists of scalars
      for (size_t i = 8; i < c->num_inputs(); ++i) {
        shape_inference::ShapeHandle v;
        TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(i), 1, &v));
        if (c->Rank(c->input(i)) == 1 && c->Value(c->Dim(v, 0)) > 1) {
          return errors::InvalidArgument(
              "Shape of a default must be a length-0 or length-1 vector, or a "
              "scalar.");
        }
      }
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("DatasetCardinality")
    .Input("input_dataset: variant")
    .Output("cardinality: int64")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalDatasetCardinality")
    .Input("input_dataset: variant")
    .Output("cardinality: int64")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("DatasetFromGraph")
    .Input("graph_def: string")
    .Output("handle: variant")
    .SetShapeFn(shape_inference::ScalarShape);

// TODO(b/124308596): Instead of conservatively marking this op as stateful,
// implement a mechanism to determine whether `dataset` has a side-effect
// and use it to decide whether to use a stateless or stateful version of this
// op.
REGISTER_OP("DatasetToTFRecord")
    .Input("input_dataset: variant")
    .Input("filename: string")
    .Input("compression_type: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("ExperimentalDatasetToTFRecord")
    .Input("input_dataset: variant")
    .Input("filename: string")
    .Input("compression_type: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("DenseToSparseBatchDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("row_shape: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // row_shape should be a 1-D vector.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalDenseToSparseBatchDataset")
    .Input("input_dataset: variant")
    .Input("batch_size: int64")
    .Input("row_shape: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // batch_size should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      // row_shape should be a 1-D vector.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("DirectedInterleaveDataset")
    .Input("selector_input_dataset: variant")
    .Input("data_input_datasets: N * variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalDirectedInterleaveDataset")
    .Input("selector_input_dataset: variant")
    .Input("data_input_datasets: N * variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("N: int >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GroupByReducerDataset")
    .Input("input_dataset: variant")
    .Input("key_func_other_arguments: Tkey_func_other_arguments")
    .Input("init_func_other_arguments: Tinit_func_other_arguments")
    .Input("reduce_func_other_arguments: Treduce_func_other_arguments")
    .Input("finalize_func_other_arguments: Tfinalize_func_other_arguments")
    .Output("handle: variant")
    .Attr("key_func: func")
    .Attr("init_func: func")
    .Attr("reduce_func: func")
    .Attr("finalize_func: func")
    .Attr("Tkey_func_other_arguments: list(type) >= 0")
    .Attr("Tinit_func_other_arguments: list(type) >= 0")
    .Attr("Treduce_func_other_arguments: list(type) >= 0")
    .Attr("Tfinalize_func_other_arguments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalGroupByReducerDataset")
    .Input("input_dataset: variant")
    .Input("key_func_other_arguments: Tkey_func_other_arguments")
    .Input("init_func_other_arguments: Tinit_func_other_arguments")
    .Input("reduce_func_other_arguments: Treduce_func_other_arguments")
    .Input("finalize_func_other_arguments: Tfinalize_func_other_arguments")
    .Output("handle: variant")
    .Attr("key_func: func")
    .Attr("init_func: func")
    .Attr("reduce_func: func")
    .Attr("finalize_func: func")
    .Attr("Tkey_func_other_arguments: list(type) >= 0")
    .Attr("Tinit_func_other_arguments: list(type) >= 0")
    .Attr("Treduce_func_other_arguments: list(type) >= 0")
    .Attr("Tfinalize_func_other_arguments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("GroupByWindowDataset")
    .Input("input_dataset: variant")
    .Input("key_func_other_arguments: Tkey_func_other_arguments")
    .Input("reduce_func_other_arguments: Treduce_func_other_arguments")
    .Input(
        "window_size_func_other_arguments: Twindow_size_func_other_arguments")
    .Output("handle: variant")
    .Attr("key_func: func")
    .Attr("reduce_func: func")
    .Attr("window_size_func: func")
    .Attr("Tkey_func_other_arguments: list(type) >= 0")
    .Attr("Treduce_func_other_arguments: list(type) >= 0")
    .Attr("Twindow_size_func_other_arguments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalGroupByWindowDataset")
    .Input("input_dataset: variant")
    .Input("key_func_other_arguments: Tkey_func_other_arguments")
    .Input("reduce_func_other_arguments: Treduce_func_other_arguments")
    .Input(
        "window_size_func_other_arguments: Twindow_size_func_other_arguments")
    .Output("handle: variant")
    .Attr("key_func: func")
    .Attr("reduce_func: func")
    .Attr("window_size_func: func")
    .Attr("Tkey_func_other_arguments: list(type) >= 0")
    .Attr("Treduce_func_other_arguments: list(type) >= 0")
    .Attr("Twindow_size_func_other_arguments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IgnoreErrorsDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalIgnoreErrorsDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("IteratorGetDevice")
    .Input("resource: resource")
    .Output("device: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalIteratorGetDevice")
    .Input("resource: resource")
    .Output("device: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("LatencyStatsDataset")
    .Input("input_dataset: variant")
    .Input("tag: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle tag_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &tag_shape));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalLatencyStatsDataset")
    .Input("input_dataset: variant")
    .Input("tag: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle tag_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &tag_shape));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("LMDBDataset")
    .Input("filenames: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalLMDBDataset")
    .Input("filenames: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MapAndBatchDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("batch_size: int64")
    .Input("num_parallel_calls: int64")
    .Input("drop_remainder: bool")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("preserve_cardinality: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Use index from the end to retrieve the Input shapes,
      // so that to avoid guessing the length of "other_arguments".
      // batch_size, num_parallel_calls, and drop_remainder are 0-D scalars.
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 3), 0, &unused));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 2), 0, &unused));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 1), 0, &unused));

      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalMapAndBatchDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("batch_size: int64")
    .Input("num_parallel_calls: int64")
    .Input("drop_remainder: bool")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("preserve_cardinality: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Use index from the end to retrieve the Input shapes,
      // so that to avoid guessing the length of "other_arguments".
      // batch_size, num_parallel_calls, and drop_remainder are 0-D scalars.
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 3), 0, &unused));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 2), 0, &unused));
      TF_RETURN_IF_ERROR(
          c->WithRank(c->input(c->num_inputs() - 1), 0, &unused));

      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalMapDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("use_inter_op_parallelism: bool = true")
    .Attr("preserve_cardinality: bool = false")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("MatchingFilesDataset")
    .Input("patterns: string")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // `patterns` must be a scalar or a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalMatchingFilesDataset")
    .Input("patterns: string")
    .Output("handle: variant")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // `patterns` must be a scalar or a vector.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 1, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("MaxIntraOpParallelismDataset")
    .Input("input_dataset: variant")
    .Input("max_intra_op_parallelism: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalMaxIntraOpParallelismDataset")
    .Input("input_dataset: variant")
    .Input("max_intra_op_parallelism: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("NonSerializableDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalNonSerializableDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ParallelInterleaveDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("cycle_length: int64")
    .Input("block_length: int64")
    .Input("sloppy: bool")
    .Input("buffer_output_elements: int64")
    .Input("prefetch_input_elements: int64")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalParallelInterleaveDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Input("cycle_length: int64")
    .Input("block_length: int64")
    .Input("sloppy: bool")
    .Input("buffer_output_elements: int64")
    .Input("prefetch_input_elements: int64")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ParseExampleDataset")
    .Input("input_dataset: variant")
    .Input("num_parallel_calls: int64")
    .Input("dense_defaults: Tdense")
    .Output("handle: variant")
    .Attr("sparse_keys: list(string) >= 0")
    .Attr("dense_keys: list(string) >= 0")
    .Attr("sparse_types: list({float,int64,string}) >= 0")
    .Attr("Tdense: list({float,int64,string}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")  // Output components will be
                                              // sorted by key (dense_keys and
                                              // sparse_keys combined) here.
    .Attr("sloppy: bool = false")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalParseExampleDataset")
    .Input("input_dataset: variant")
    .Input("num_parallel_calls: int64")
    .Input("dense_defaults: Tdense")
    .Output("handle: variant")
    .Attr("sparse_keys: list(string) >= 0")
    .Attr("dense_keys: list(string) >= 0")
    .Attr("sparse_types: list({float,int64,string}) >= 0")
    .Attr("Tdense: list({float,int64,string}) >= 0")
    .Attr("dense_shapes: list(shape) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")  // Output components will be
                                              // sorted by key (dense_keys and
                                              // sparse_keys combined) here.
    .Attr("sloppy: bool = false")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("PrivateThreadPoolDataset")
    .Input("input_dataset: variant")
    .Input("num_threads: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalPrivateThreadPoolDataset")
    .Input("input_dataset: variant")
    .Input("num_threads: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalRandomDataset")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // buffer_size, seed, and seed2 should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("RandomDataset")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // buffer_size, seed, and seed2 should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalRebatchDataset")
    .Input("input_dataset: variant")
    .Input("num_replicas: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("use_fallback: bool = true")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("RebatchDataset")
    .Input("input_dataset: variant")
    .Input("num_replicas: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("use_fallback: bool = true")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("SamplingDataset")
    .Input("input_dataset: variant")
    .Input("rate: float32")
    .Input("seed: int64")
    .Input("seed2: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // rate, seed, and seed2 should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ScanDataset")
    .Input("input_dataset: variant")
    .Input("initial_state: Tstate")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Tstate: list(type) >= 1")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("preserve_cardinality: bool = false")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalScanDataset")
    .Input("input_dataset: variant")
    .Input("initial_state: Tstate")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("f: func")
    .Attr("Tstate: list(type) >= 1")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("preserve_cardinality: bool = false")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("SetStatsAggregatorDataset")
    .Input("input_dataset: variant")
    .Input("stats_aggregator: resource")
    .Input("tag: string")
    .Input("counter_prefix: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalSetStatsAggregatorDataset")
    .Input("input_dataset: variant")
    .Input("stats_aggregator: resource")
    .Input("tag: string")
    .Input("counter_prefix: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("SleepDataset")
    .Input("input_dataset: variant")
    .Input("sleep_microseconds: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // Both inputs are scalar.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalSleepDataset")
    .Input("input_dataset: variant")
    .Input("sleep_microseconds: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // Both inputs are scalar.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("SlidingWindowDataset")
    .Input("input_dataset: variant")
    .Input("window_size: int64")
    .Input("window_shift: int64")
    .Input("window_stride: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // window_size, window_shift, and window_stride should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalSlidingWindowDataset")
    .Input("input_dataset: variant")
    .Input("window_size: int64")
    .Input("window_shift: int64")
    .Input("window_stride: int64")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // window_size, window_shift, and window_stride should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("SnapshotDataset")
    .Input("input_dataset: variant")
    .Input("path: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .Attr("compression: string = ''")
    .Attr("reader_path_prefix: string = ''")
    .Attr("writer_path_prefix: string = ''")
    .Attr("shard_size_bytes: int = 10737418240")           // 10 GiB default
    .Attr("pending_snapshot_expiry_seconds: int = 86400")  // 1 day default
    .Attr("num_reader_threads: int = 1")
    .Attr("reader_buffer_size: int = 1")
    .Attr("num_writer_threads: int = 1")
    .Attr("writer_buffer_size: int = 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // snapshot_path should be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("SqlDataset")
    .Input("driver_name: string")
    .Input("data_source_name: string")
    .Input("query: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // driver_name, data_source_name, and query should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("ExperimentalSqlDataset")
    .Input("driver_name: string")
    .Input("data_source_name: string")
    .Input("query: string")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetIsStateful()  // TODO(b/123753214): Source dataset ops must be marked
                      // stateful to inhibit constant folding.
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle unused;
      // driver_name, data_source_name, and query should be scalars.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      return shape_inference::ScalarShape(c);
    });

REGISTER_OP("StatsAggregatorHandle")
    .Output("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''");

REGISTER_OP("ExperimentalStatsAggregatorHandle")
    .Output("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''");

REGISTER_OP("StatsAggregatorHandleV2")
    .Output("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''");

REGISTER_OP("StatsAggregatorSetSummaryWriter")
    .Input("stats_aggregator: resource")
    .Input("summary: resource")
    .SetShapeFn(shape_inference::NoOutputs);

REGISTER_OP("StatsAggregatorSummary")
    .Input("iterator: resource")
    .Output("summary: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalStatsAggregatorSummary")
    .Input("iterator: resource")
    .Output("summary: string")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("TakeWhileDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("predicate: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalTakeWhileDataset")
    .Input("input_dataset: variant")
    .Input("other_arguments: Targuments")
    .Output("handle: variant")
    .Attr("predicate: func")
    .Attr("Targuments: list(type) >= 0")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ThreadPoolDataset")
    .Input("input_dataset: variant")
    .Input("thread_pool: resource")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalThreadPoolDataset")
    .Input("input_dataset: variant")
    .Input("thread_pool: resource")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ThreadPoolHandle")
    .Output("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Attr("num_threads: int")
    .Attr("max_intra_op_parallelism: int = 1")
    .Attr("display_name: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''");

REGISTER_OP("ExperimentalThreadPoolHandle")
    .Output("handle: resource")
    .SetShapeFn(shape_inference::ScalarShape)
    .Attr("num_threads: int")
    .Attr("max_intra_op_parallelism: int = 1")
    .Attr("display_name: string")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''");

REGISTER_OP("UnbatchDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalUnbatchDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("UniqueDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("ExperimentalUniqueDataset")
    .Input("input_dataset: variant")
    .Output("handle: variant")
    .Attr("output_types: list(type) >= 1")
    .Attr("output_shapes: list(shape) >= 1")
    .SetShapeFn(shape_inference::ScalarShape);

}  // namespace tensorflow
