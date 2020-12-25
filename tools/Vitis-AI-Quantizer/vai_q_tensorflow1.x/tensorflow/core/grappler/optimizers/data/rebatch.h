/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_REBATCH_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_REBATCH_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/optimizer_base.h"

namespace tensorflow {
namespace grappler {

// This optimizer changes the batch size of the output dataset by dividing the
// current batch size by parameter `num_replicas`. Currently, this works only
// for very simple pipelines with a single BatchDatasetV2 transformation.
class RebatchOptimizer : public TFDataOptimizerBase {
 public:
  RebatchOptimizer() = default;
  ~RebatchOptimizer() override = default;

  string name() const override { return "tf_data_rebatcher"; }

  bool UsesFunctionLibrary() const override { return true; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

 private:
  int64 num_replicas_;
  bool use_fallback_;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_REBATCH_H_
