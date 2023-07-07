/*Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_DECENT_Q_UTILS_KNOWN_PATTERNS_H_
#define TENSORFLOW_CONTRIB_DECENT_Q_UTILS_KNOWN_PATTERNS_H_

#include "transform_utils.h"

namespace tensorflow {
namespace decent_q {

class OpTypePatternBase;

// Quantizable op types
extern const std::set<string> quantizable_op_types;

// Supported op types: not quantizable but can be handled by decent
extern const std::set<string> supported_op_types;

// Unsupported op types: not quantizable and cannot be handled by decent
extern const std::set<string> unsupported_op_types;

// Known patterns
extern const std::vector<const OpTypePatternBase*> known_patterns;

// Known ignore patterns
extern const std::vector<std::tuple<const string, const OpTypePattern>>
    known_ignore_patterns;

// Patterns with computation
extern const std::set<string> compute_patterns;

// Get pattern name from id
const string get_pattern_name_from_id(const int pattern_id);

// Get ignore pattern name from id
const string get_ignore_pattern_name_from_id(const int pattern_id);

// Get input nodes from pattern
std::vector<const NodeDef*> get_input_nodes(const NodeMatch& match,
                                            const string& pattern_name);

// Get ignore nodes from pattern
std::vector<const NodeDef*> get_ignore_nodes(const NodeMatch& match,
                                             const string& pattern_name);
// Get weights nodes from pattern
std::vector<const NodeDef*> get_weights_nodes(const NodeMatch& match,
                                              const string& pattern_name);

class OpTypePatternBase {
 public:
  OpTypePatternBase() {}

  OpTypePatternBase(const OpTypePattern& pattern, const string& name = "")
    : _pattern(pattern), _name(name) {}

  string GetName() const { return _name; };

  OpTypePattern GetPattern() const {
    return _pattern;
  }

  virtual std::vector<const NodeDef*> GetInputNodes(const NodeMatch& match) const { return std::vector<const NodeDef*>(); };

  virtual std::vector<const NodeDef*> GetWeightsNodes(const NodeMatch& match) const { return std::vector<const NodeDef*>(); };

  virtual ~OpTypePatternBase() {};

 private:
  string _name;
  OpTypePattern _pattern;
};


#define CREATE_PATTERN_CLASS(NAME)              \
class NAME##Pattern : public OpTypePatternBase {                                     \
 public:                                                                            \
  NAME##Pattern(const OpTypePattern& pattern, const string& name = "")              \
    : OpTypePatternBase(pattern, name) {}                                           \
                                                                                    \
  virtual std::vector<const NodeDef*> GetInputNodes (                               \
      const NodeMatch& match) const override;                                       \
                                                                                    \
  virtual std::vector<const NodeDef*> GetWeightsNodes (                             \
      const NodeMatch& match) const override;                                       \
};                                                                                  \

#define DEFINE_GET_INPUT_NODES(NAME) \
  std::vector<const NodeDef*> NAME##Pattern::GetInputNodes(const NodeMatch& match) const \

#define DEFINE_GET_WEIGHTS_NODES(NAME) \
  std::vector<const NodeDef*> NAME##Pattern::GetWeightsNodes(const NodeMatch& match) const \

CREATE_PATTERN_CLASS(Placeholder)
CREATE_PATTERN_CLASS(AtrousConvBiasRelu)
CREATE_PATTERN_CLASS(AtrousConvBias)
CREATE_PATTERN_CLASS(AtrousConvRelu)
CREATE_PATTERN_CLASS(AtrousConv)
CREATE_PATTERN_CLASS(ConvfcBiasLeakyrelu)
CREATE_PATTERN_CLASS(ConvfcBiasFusedLeakyrelu)
CREATE_PATTERN_CLASS(ConvfcBiasKerasLeakyrelu)
CREATE_PATTERN_CLASS(ConvfcLeakyrelu)
CREATE_PATTERN_CLASS(ConvfcFusedLeakyrelu)
CREATE_PATTERN_CLASS(ConvfcKerasLeakyrelu)
CREATE_PATTERN_CLASS(Swish)
CREATE_PATTERN_CLASS(HardSwish)
CREATE_PATTERN_CLASS(HardSigmoid)
CREATE_PATTERN_CLASS(Leakyrelu)
CREATE_PATTERN_CLASS(FusedLeakyrelu)
CREATE_PATTERN_CLASS(KerasLeakyrelu)
CREATE_PATTERN_CLASS(ConvfcBiasIdRelu)
CREATE_PATTERN_CLASS(ConvfcBiasRelu)
CREATE_PATTERN_CLASS(ConvfcBias)
CREATE_PATTERN_CLASS(ConvfcRelu)
CREATE_PATTERN_CLASS(Convfc)
CREATE_PATTERN_CLASS(Conv2dTransposeBiasRelu)
CREATE_PATTERN_CLASS(Conv2dTransposeBias)
CREATE_PATTERN_CLASS(Conv2dTransposeRelu)
CREATE_PATTERN_CLASS(Conv2dTranspose)
CREATE_PATTERN_CLASS(KerasConv2dTransposeBiasRelu)
CREATE_PATTERN_CLASS(KerasConv2dTransposeBias)
CREATE_PATTERN_CLASS(KerasConv2dTransposeRelu)
CREATE_PATTERN_CLASS(KerasConv2dTranspose)
CREATE_PATTERN_CLASS(Conv2dBackpropInputBiasRelu)
CREATE_PATTERN_CLASS(Conv2dBackpropInputBias)
CREATE_PATTERN_CLASS(Conv2dBackpropInputRelu)
CREATE_PATTERN_CLASS(Conv2dBackpropInput)
CREATE_PATTERN_CLASS(Upsampling)
CREATE_PATTERN_CLASS(Resize)
CREATE_PATTERN_CLASS(DepthToSpace)
CREATE_PATTERN_CLASS(TpuNearestNeighborUpsampling)
CREATE_PATTERN_CLASS(BatchnormRelu)
CREATE_PATTERN_CLASS(Batchnorm)
CREATE_PATTERN_CLASS(ArrayRelu)
CREATE_PATTERN_CLASS(Array)
CREATE_PATTERN_CLASS(AvgpoolMul)
CREATE_PATTERN_CLASS(ClipByValue)
CREATE_PATTERN_CLASS(OtherRelu)
CREATE_PATTERN_CLASS(Other)
CREATE_PATTERN_CLASS(Mul_v1)
CREATE_PATTERN_CLASS(Mul_v2)
}  // namespace decent_q
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_DECENT_Q_UTILS_KNOWN_PATTERNS_H_
