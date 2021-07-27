/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/vitisai/vitisai_delegate.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/minimal_logging.h"

#include "tensorflow/lite/delegates/vitisai/pyxir/include/pyxir/pyxir.hpp"

namespace tflite {
namespace vitisai {
namespace {

// Forward declaration.
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate);

class Delegate {
  friend class Subgraph;

 public:
  explicit Delegate(const TfLiteVitisAIDelegateOptions* options) {
    if (options) {
      options_ = *options;
    } else {
      strcpy(options_.target, "");
    }

    TFLITE_LOG_PROD_ONCE(tflite::TFLITE_LOG_INFO,
                         "Created TensorFlow Lite VITISAI delegate for FPGA.");
  }

  TfLiteIntArray* PrepareOpsToDelegate(TfLiteContext* context);
  TfLiteDelegate* tflite_delegate() { return &delegate_; }

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  TfLiteVitisAIDelegateOptions options_;
};

class Subgraph {
 public:
  static Subgraph* Create(TfLiteContext* context,
                          const TfLiteDelegateParams* params,
                          const Delegate* delegate) {
    // Create VITISAI input and output tensor lists
    std::vector<int> inputs;
    for (int i = 0; i < params->input_tensors->size; ++i) {
      std::cout << "Subgraph::Create -- inputs name: " << context->tensors[params->input_tensors->data[i]].name << std::endl;
      if (context->tensors[params->input_tensors->data[i]].data.raw == nullptr) {
        inputs.push_back(params->input_tensors->data[i]);
      }
    }

    std::vector<int> outputs;
    for (int i = 0; i < params->output_tensors->size; ++i) {
      std::cout << "Subgraph::Create -- outputs name: " << context->tensors[params->output_tensors->data[i]].name << std::endl;
      if (context->tensors[params->output_tensors->data[i]].data.raw == nullptr) {
        outputs.push_back(params->output_tensors->data[i]);
      }
    }

    // Create VITISAI nodes for TFLite delegate nodes
    std::shared_ptr<pyxir::graph::XGraph> x_graph(new pyxir::graph::XGraph(std::string("")));

    for (int i : inputs) {
      std::string name = context->tensors[i].name;
      std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Input"));
      std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
      shapes[0].push_back(-1);
      for (int j = 1; j < context->tensors[i].dims->size; ++j) {
        shapes[0].push_back(context->tensors[i].dims->data[j]);
      }
      std::string shapes_t = std::string("TensorShape");
      std::vector<int64_t> sizes = std::vector<int64_t>();
      std::vector<std::string> bottoms = std::vector<std::string>();
      std::vector<std::string> tops = std::vector<std::string>();
      std::vector<std::string> layer = std::vector<std::string>(1, name);

      pyxir::graph::XLayer x_layer(
        name,
        xtype,
        shapes,
        shapes_t,
        sizes,
        bottoms,
        tops,
        layer
      );

      x_layer.set_attr(std::string("data_layout"), pyxir::graph::XAttr(std::string("data_layout"), std::string("NHWC")));

      x_graph->add(x_layer);
    }

    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int node_index = params->nodes_to_replace->data[i];

      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        return nullptr;
      }

      if (VisitNode(x_graph, context, registration, node,
                    node_index) != kTfLiteOk) {
        return nullptr;
      }
    }

    pyxir::RtModHolder rt_mod_holder;
    try {
      std::string rt_mod_holder_filename("rt_mod_holder.bin");
      std::ifstream ifs;
      ifs.open(rt_mod_holder_filename);
      if (ifs.is_open()) {
        ifs.close();
        //rt_mod_holder->load(rt_mod_holder_filename);
        std::ifstream in_file(rt_mod_holder_filename);
        std::stringstream buffer;
        buffer << in_file.rdbuf();
        std::string serialized_rt_mod = buffer.str();
        in_file.close();

        std::istringstream sstream(serialized_rt_mod);
        rt_mod_holder.reset(new pyxir::runtime::RuntimeModule());
        rt_mod_holder->deserialize(sstream);
      } else {
        std::string backend_type = delegate->options_.target;
        pyxir::partition(x_graph, std::vector<std::string>(1, backend_type));

        std::vector<std::string> in_tensor_names;
        for (int i : inputs) {
          in_tensor_names.push_back(context->tensors[i].name);
        }

        std::vector<std::string> out_tensor_names;
        for (int i : outputs) {
          out_tensor_names.push_back(context->tensors[i].name);
        }

        pyxir::RunOptionsHolder run_options(new pyxir::runtime::RunOptions());
        run_options->on_the_fly_quantization = true;
        run_options->export_runtime_module_path = rt_mod_holder_filename;

        rt_mod_holder = pyxir::build_rt(x_graph, backend_type, in_tensor_names, out_tensor_names,
                                      "vai", run_options);
      }
    }
    catch (...) {
      //TF_LITE_KERNEL_LOG(context, "failed to create VITISAI runtime");
      return nullptr;
    }

    return new Subgraph(std::move(rt_mod_holder), inputs, outputs);
  }

  TfLiteStatus Prepare(TfLiteContext* context) { return kTfLiteOk; }

  TfLiteStatus Invoke(TfLiteContext* context) {
    if (first_run_) {

      first_run_ = false;
    }

    std::vector<pyxir::XBufferHolder> in_tensors;
    for (int i : inputs_) {
      std::cout << "Subgraph::Invoke -- in_tensor_name: " << context->tensors[i].name << std::endl;
      std::vector<ssize_t> shape;
      for (int j = 0; j < context->tensors[i].dims->size; ++j) {
        shape.push_back(context->tensors[i].dims->data[j]);
      }
      in_tensors.push_back(std::shared_ptr<pyxir::XBuffer>(new pyxir::XBuffer(
        (void *) context->tensors[i].data.raw,
        4,
        "f",
        context->tensors[i].dims->size,
        shape,
        false,
        false
      )));
    }

    std::vector<pyxir::XBufferHolder> out_tensors;
    for (int i : outputs_) {
      std::cout << "Subgraph::Invoke -- out_tensor_name: " << context->tensors[i].name << std::endl;
      std::vector<ssize_t> shape;
      for (int j = 0; j < context->tensors[i].dims->size; ++j) {
        shape.push_back(context->tensors[i].dims->data[j]);
      }
      out_tensors.push_back(std::shared_ptr<pyxir::XBuffer>(new pyxir::XBuffer(
        (void *) context->tensors[i].data.raw,
        4,
        "f",
        context->tensors[i].dims->size,
        shape,
        false,
        false
      )));
    }

    try {
      std::clock_t c_start = std::clock();
      auto t_start = std::chrono::high_resolution_clock::now();

      runtime_->execute(in_tensors, out_tensors);

      std::clock_t c_end = std::clock();
      auto t_end = std::chrono::high_resolution_clock::now();

      std::cout << "Subgraph::Invoke -- cpu time (ms): " << 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC << std::endl;
      std::cout << "Subgraph::Invoke -- wall time (ms): " << std::chrono::duration<double, std::milli>(t_end-t_start).count() << std::endl;

      //for (int i : inputs_) {
      //  size_t num_elements = 1;
      //  for (int j = 0; j < context->tensors[i].dims->size; ++j) {
      //    num_elements *= context->tensors[i].dims->data[j];
      //  }
      //  size_t size = 4 * num_elements;

      //  std::ofstream in_tensor_file;
      //  in_tensor_file.open("in_tensor.bin", std::ios::out);
      //  for (size_t j = 0; j < num_elements; ++j) {
      //    in_tensor_file << context->tensors[i].data.f[j] << std::endl;
      //  }
      //  in_tensor_file.close();
      //}

      //for (int i : outputs_) {
      //  size_t num_elements = 1;
      //  for (int j = 0; j < context->tensors[i].dims->size; ++j) {
      //    num_elements *= context->tensors[i].dims->data[j];
      //  }
      //  size_t size = 4 * num_elements;

      //  std::ofstream out_tensor_file;
      //  out_tensor_file.open("out_tensor.bin", std::ios::out);
      //  for (size_t j = 0; j < num_elements; ++j) {
      //    out_tensor_file << context->tensors[i].data.f[j] << std::endl;
      //  }
      //  out_tensor_file.close();
      //}
    }
    catch (...) {
      //TF_LITE_KERNEL_LOG(context, "failed to invoke VITISAI runtime");
      return kTfLiteError;
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* context,
      TfLiteRegistration* registration, TfLiteNode* node, int node_index) {
    // TFLite context used for logging purposes. When we create a new node
    // (subgraph is non-null), logging context is the same as context, and error
    // messages are passed to TFLite. When we detect supported operations
    // (subgraph is null), logging context is null, and error messages are
    // supressed.
    TfLiteContext* logging_context = x_graph == nullptr ? nullptr : context;
    switch (registration->builtin_code) {
      case kTfLiteBuiltinAdd: {
        const TfLiteAddParams* add_params =
            static_cast<const TfLiteAddParams*>(node->builtin_data);

        return VisitAddNode(x_graph, logging_context, node_index, node,
                            context->tensors, add_params);
      }
      case kTfLiteBuiltinAveragePool2d: {
        const TfLitePoolParams* pool_params =
            static_cast<const TfLitePoolParams*>(node->builtin_data);

        return VisitAveragePool2DNode(x_graph, logging_context, node_index,
                                      node, context->tensors, pool_params);
      }
      case kTfLiteBuiltinConcatenation: {
        const TfLiteConcatenationParams* concatenation_params =
            static_cast<const TfLiteConcatenationParams*>(node->builtin_data);

        return VisitConcatenationNode(x_graph, logging_context, node_index,
                                      node, context->tensors, concatenation_params);
      }
      case kTfLiteBuiltinConv2d: {
        const TfLiteConvParams* conv_params =
            static_cast<const TfLiteConvParams*>(node->builtin_data);

        return VisitConv2DNode(x_graph, logging_context, node_index, node,
                               context->tensors, conv_params);
      }
      case kTfLiteBuiltinDepthwiseConv2d: {
        const TfLiteDepthwiseConvParams* depthwise_conv_params =
            static_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);

        return VisitDepthwiseConv2DNode(x_graph, logging_context, node_index,
                                        node, context->tensors, depthwise_conv_params);
      }
      case kTfLiteBuiltinMaxPool2d: {
        const TfLitePoolParams* pool_params =
            static_cast<const TfLitePoolParams*>(node->builtin_data);

        return VisitMaxPool2DNode(x_graph, logging_context, node_index, node,
                                  context->tensors, pool_params);
      }
      case kTfLiteBuiltinMul: {
        const TfLiteMulParams* mul_params =
            static_cast<const TfLiteMulParams*>(node->builtin_data);

        return VisitMulNode(x_graph, logging_context, node_index, node,
                            context->tensors, mul_params);
      }
      case kTfLiteBuiltinRelu: {
        return VisitReluNode(x_graph, logging_context, node_index, node,
                             context->tensors);
      }
      case kTfLiteBuiltinPad: {
        const TfLitePadParams* pad_params =
            static_cast<const TfLitePadParams*>(node->builtin_data);

        return VisitPadNode(x_graph, logging_context, node_index, node,
                            context->tensors, pad_params);
      }
      case kTfLiteBuiltinMean: {
        const TfLiteReducerParams* reducer_params =
            static_cast<const TfLiteReducerParams*>(node->builtin_data);

        return VisitMeanNode(x_graph, logging_context, node_index, node,
                             context->tensors, reducer_params);
      }
      case kTfLiteBuiltinLeakyRelu: {
        const TfLiteLeakyReluParams* leaky_relu_params =
            static_cast<const TfLiteLeakyReluParams*>(node->builtin_data);

        return VisitLeakyReluNode(x_graph, logging_context, node_index, node,
                                  context->tensors, leaky_relu_params);
      }
      default: {
        return VisitDefaultNode(x_graph, logging_context, node_index, node,
                                context->tensors);
      }
    }
  }

  static TfLiteStatus VisitAddNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteAddParams* add_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tAdd\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype;
        if (tensors[node->inputs->data[1]].dims->size == 1) {
          xtype = std::vector<std::string>(1, std::string("Scale"));
        } else {
          xtype = std::vector<std::string>(1, std::string("Eltwise"));
        }
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
        if (tensors[node->inputs->data[1]].dims->size == 1) {
          bottoms.push_back(tensors[node->inputs->data[0]].name);
        } else {
          for (int j = 0; j < node->inputs->size; ++j) {
            bottoms.push_back(tensors[node->inputs->data[j]].name);
          }
        }
        std::vector<std::string> tops = std::vector<std::string>();
        std::vector<std::string> layer = std::vector<std::string>(1, name);
        std::vector<pyxir::XBuffer> data;
        if (tensors[node->inputs->data[1]].dims->size == 1) {
          std::vector<float> scale(tensors[node->inputs->data[1]].dims->data[0], 1);
          data.push_back(pyxir::XBuffer(
            (void *) scale.data(),
            4,
            "f",
            tensors[node->inputs->data[1]].dims->size,
            std::vector<ssize_t>{
              tensors[node->inputs->data[1]].dims->data[0]
            },
            false,
            false
          ));
          data.push_back(pyxir::XBuffer(
            (void *) tensors[node->inputs->data[1]].data.raw,
            4,
            "f",
            tensors[node->inputs->data[1]].dims->size,
            std::vector<ssize_t>{
              tensors[node->inputs->data[1]].dims->data[0]
            },
            false,
            false
          ));
        }

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms,
          tops,
          layer,
          data
        );

        if (tensors[node->inputs->data[1]].dims->size == 1) {
          x_layer.set_attr(std::string("axis"), pyxir::graph::XAttr(std::string("axis"), -1));
        } else {
          x_layer.set_attr(std::string("op"), pyxir::graph::XAttr(std::string("op"), std::string("Add")));
        }

        switch (add_params->activation) {
          case kTfLiteActNone: {
            printf("\t\tkTfLiteActNone\n");
            break;
          }
          case kTfLiteActRelu: {
            printf("\t\tkTfLiteActRelu\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU")));
            break;
          }
          case kTfLiteActRelu6: {
            printf("\t\tkTfLiteActRelu6\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU6")));
            break;
          }
          default: {
            printf("\t\tkTfLiteActNone (default)\n");
            break;
          }
        }

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate ADD node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitAveragePool2DNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLitePoolParams* pool_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tAveragePool2d\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Pooling"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
          bottoms.push_back(tensors[node->inputs->data[0]].name);

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms
        );

        switch (pool_params->padding) {
          case kTfLitePaddingUnknown: {
            printf("\t\tkTfLitePaddingUnknown\n");
            break;
          }
          case kTfLitePaddingSame: {
            printf("\t\tkTfLitePaddingSame\n");
            std::vector<std::vector<int64_t> > padding(4);
            {
              int64_t out_height = tensors[node->outputs->data[0]].dims->data[1];
              int64_t out_width = tensors[node->outputs->data[0]].dims->data[2];
              int64_t stride_height = pool_params->stride_height;
              int64_t stride_width = pool_params->stride_width;
              int64_t filter_height = pool_params->filter_height;
              int64_t filter_width = pool_params->filter_width;
              int64_t in_height = tensors[node->inputs->data[0]].dims->data[1];
              int64_t in_width = tensors[node->inputs->data[0]].dims->data[2];
              int64_t pad_along_height = std::max((out_height - 1) * stride_height + filter_height - in_height, (int64_t) 0);
              int64_t pad_along_width = std::max((out_width - 1) * stride_width + filter_width - in_width, (int64_t) 0);
              padding[0] = std::vector<int64_t>(2, 0);
              padding[1] = std::vector<int64_t>(2, 0);
              padding[2] = std::vector<int64_t>(2, 0);
              padding[3] = std::vector<int64_t>(2, 0);
              padding[1][0] = pad_along_height / 2;
              padding[1][1] = pad_along_height - padding[1][0];
              padding[2][0] = pad_along_width / 2;
              padding[2][1] = pad_along_width - padding[2][0];
            }
            x_layer.set_attr(std::string("padding"), pyxir::graph::XAttr(std::string("padding"), padding));
            x_layer.set_attr(std::string("padding_type"), pyxir::graph::XAttr(std::string("padding_type"), std::string("SAME")));
            break;
          }
          case kTfLitePaddingValid: {
            printf("\t\tkTfLitePaddingValid\n");
            std::vector<std::vector<int64_t> > padding(4);
            padding[0] = std::vector<int64_t>(2, 0);
            padding[1] = std::vector<int64_t>(2, 0);
            padding[2] = std::vector<int64_t>(2, 0);
            padding[3] = std::vector<int64_t>(2, 0);
            x_layer.set_attr(std::string("padding"), pyxir::graph::XAttr(std::string("padding"), padding));
            x_layer.set_attr(std::string("padding_type"), pyxir::graph::XAttr(std::string("padding_type"), std::string("VALID")));
            break;
          }
          default: {
            printf("\t\tkTfLitePaddingUnknown (default)\n");
            break;
          }
        }

        x_layer.set_attr(std::string("data_layout"), pyxir::graph::XAttr(std::string("data_layout"), std::string("NHWC")));

        std::vector<int64_t> strides(2);
        strides[0] = pool_params->stride_height;
        strides[1] = pool_params->stride_width;
        x_layer.set_attr(std::string("strides"), pyxir::graph::XAttr(std::string("strides"), strides));

        std::vector<int64_t> filters(2);
        filters[0] = pool_params->filter_height;
        filters[1] = pool_params->filter_width;
        x_layer.set_attr(std::string("kernel_size"), pyxir::graph::XAttr(std::string("kernel_size"), filters));

        std::vector<int64_t> insize(2);
        insize[0] = tensors[node->inputs->data[0]].dims->data[1];
        insize[1] = tensors[node->inputs->data[0]].dims->data[2];
        x_layer.set_attr(std::string("insize"), pyxir::graph::XAttr(std::string("insize"), insize));

        std::vector<int64_t> outsize(2);
        outsize[0] = tensors[node->outputs->data[0]].dims->data[1];
        outsize[1] = tensors[node->outputs->data[0]].dims->data[2];
        x_layer.set_attr(std::string("outsize"), pyxir::graph::XAttr(std::string("outsize"), outsize));

        x_layer.set_attr(std::string("pool_type"), pyxir::graph::XAttr(std::string("pool_type"), std::string("Avg")));

        switch (pool_params->activation) {
          case kTfLiteActNone: {
            printf("\t\tkTfLiteActNone\n");
            break;
          }
          case kTfLiteActRelu: {
            printf("\t\tkTfLiteActRelu\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU")));
            break;
          }
          case kTfLiteActRelu6: {
            printf("\t\tkTfLiteActRelu6\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU6")));
            break;
          }
          default: {
            printf("\t\tkTfLiteActNone (default)\n");
            break;
          }
        }

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate AVERAGE_POOL_2D node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitConcatenationNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteConcatenationParams* concatenation_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tConcatenation\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Concat"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
        for (int j = 0; j < node->inputs->size; ++j) {
          bottoms.push_back(tensors[node->inputs->data[j]].name);
        }

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms
        );

        int axis = concatenation_params->axis;
        x_layer.set_attr(std::string("axis"), pyxir::graph::XAttr(std::string("axis"), axis));

        switch (concatenation_params->activation) {
          case kTfLiteActNone: {
            printf("\t\tkTfLiteActNone\n");
            break;
          }
          case kTfLiteActRelu: {
            printf("\t\tkTfLiteActRelu\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU")));
            break;
          }
          case kTfLiteActRelu6: {
            printf("\t\tkTfLiteActRelu6\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU6")));
            break;
          }
          default: {
            printf("\t\tkTfLiteActNone (default)\n");
            break;
          }
        }

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate CONV_2D node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitConv2DNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteConvParams* conv_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tConv2d\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Convolution"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
          bottoms.push_back(tensors[node->inputs->data[0]].name);
        std::vector<std::string> tops = std::vector<std::string>();
        std::vector<std::string> layer = std::vector<std::string>(1, name);
        std::vector<pyxir::XBuffer> data;
        data.push_back(pyxir::XBuffer(
          (void *) tensors[node->inputs->data[1]].data.raw,
          4,
          "f",
          tensors[node->inputs->data[1]].dims->size,
          std::vector<ssize_t>{
            tensors[node->inputs->data[1]].dims->data[0],
            tensors[node->inputs->data[1]].dims->data[1],
            tensors[node->inputs->data[1]].dims->data[2],
            tensors[node->inputs->data[1]].dims->data[3]
          },
          false,
          false
        ));
        data.push_back(pyxir::XBuffer(
          (void *) tensors[node->inputs->data[2]].data.raw,
          4,
          "f",
          tensors[node->inputs->data[2]].dims->size,
          std::vector<ssize_t>{
            tensors[node->inputs->data[2]].dims->data[0]
          },
          false,
          false
        ));

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms,
          tops,
          layer,
          data
        );

        switch (conv_params->padding) {
          case kTfLitePaddingUnknown: {
            printf("\t\tkTfLitePaddingUnknown\n");
            break;
          }
          case kTfLitePaddingSame: {
            printf("\t\tkTfLitePaddingSame\n");
            std::vector<std::vector<int64_t> > padding(4);
            {
              int64_t out_height = tensors[node->outputs->data[0]].dims->data[1];
              int64_t out_width = tensors[node->outputs->data[0]].dims->data[2];
              int64_t stride_height = conv_params->stride_height;
              int64_t stride_width = conv_params->stride_width;
              int64_t filter_height = tensors[node->inputs->data[1]].dims->data[1];
              int64_t filter_width = tensors[node->inputs->data[1]].dims->data[2];
              int64_t in_height = tensors[node->inputs->data[0]].dims->data[1];
              int64_t in_width = tensors[node->inputs->data[0]].dims->data[2];
              int64_t pad_along_height = std::max((out_height - 1) * stride_height + filter_height - in_height, (int64_t) 0);
              int64_t pad_along_width = std::max((out_width - 1) * stride_width + filter_width - in_width, (int64_t) 0);
              padding[0] = std::vector<int64_t>(2, 0);
              padding[1] = std::vector<int64_t>(2, 0);
              padding[2] = std::vector<int64_t>(2, 0);
              padding[3] = std::vector<int64_t>(2, 0);
              padding[1][0] = pad_along_height / 2;
              padding[1][1] = pad_along_height - padding[1][0];
              padding[2][0] = pad_along_width / 2;
              padding[2][1] = pad_along_width - padding[2][0];
            }
            x_layer.set_attr(std::string("padding"), pyxir::graph::XAttr(std::string("padding"), padding));
            x_layer.set_attr(std::string("padding_type"), pyxir::graph::XAttr(std::string("padding_type"), std::string("SAME")));
            break;
          }
          case kTfLitePaddingValid: {
            printf("\t\tkTfLitePaddingValid\n");
            std::vector<std::vector<int64_t> > padding(4);
            padding[0] = std::vector<int64_t>(2, 0);
            padding[1] = std::vector<int64_t>(2, 0);
            padding[2] = std::vector<int64_t>(2, 0);
            padding[3] = std::vector<int64_t>(2, 0);
            x_layer.set_attr(std::string("padding"), pyxir::graph::XAttr(std::string("padding"), padding));
            x_layer.set_attr(std::string("padding_type"), pyxir::graph::XAttr(std::string("padding_type"), std::string("VALID")));
            break;
          }
          default: {
            printf("\t\tkTfLitePaddingUnknown (default)\n");
            break;
          }
        }

        x_layer.set_attr(std::string("data_layout"), pyxir::graph::XAttr(std::string("data_layout"), std::string("NHWC")));

        x_layer.set_attr(std::string("kernel_layout"), pyxir::graph::XAttr(std::string("kernel_layout"), std::string("OHWI")));

        std::vector<int64_t> kernels(2);
        kernels[0] = tensors[node->inputs->data[1]].dims->data[1];
        kernels[1] = tensors[node->inputs->data[1]].dims->data[2];
        x_layer.set_attr(std::string("kernel_size"), pyxir::graph::XAttr(std::string("kernel_size"), kernels));

        std::vector<int64_t> strides(2);
        strides[0] = conv_params->stride_height;
        strides[1] = conv_params->stride_width;
        x_layer.set_attr(std::string("strides"), pyxir::graph::XAttr(std::string("strides"), strides));

        int groups = 1;
        x_layer.set_attr(std::string("groups"), pyxir::graph::XAttr(std::string("groups"), groups));

        std::vector<int64_t> dilation_factors(2);
        dilation_factors[0] = conv_params->dilation_height_factor;
        dilation_factors[1] = conv_params->dilation_width_factor;
        x_layer.set_attr(std::string("dilation"), pyxir::graph::XAttr(std::string("dilation"), dilation_factors));

        std::vector<int64_t> channels(2);
        channels[0] = tensors[node->inputs->data[0]].dims->data[3];
        channels[1] = tensors[node->outputs->data[0]].dims->data[3];
        x_layer.set_attr(std::string("channels"), pyxir::graph::XAttr(std::string("channels"), channels));

        switch (conv_params->activation) {
          case kTfLiteActNone: {
            printf("\t\tkTfLiteActNone\n");
            break;
          }
          case kTfLiteActRelu: {
            printf("\t\tkTfLiteActRelu\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU")));
            break;
          }
          case kTfLiteActRelu6: {
            printf("\t\tkTfLiteActRelu6\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU6")));
            break;
          }
          default: {
            printf("\t\tkTfLiteActNone (default)\n");
            break;
          }
        }

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate CONV_2D node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitDepthwiseConv2DNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteDepthwiseConvParams* depthwise_conv_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tDepthwiseConv2d\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Convolution"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
          bottoms.push_back(tensors[node->inputs->data[0]].name);
        std::vector<std::string> tops = std::vector<std::string>();
        std::vector<std::string> layer = std::vector<std::string>(1, name);
        std::vector<pyxir::XBuffer> data;
        data.push_back(pyxir::XBuffer(
          (void *) tensors[node->inputs->data[1]].data.raw,
          4,
          "f",
          tensors[node->inputs->data[1]].dims->size,
          std::vector<ssize_t>{
            tensors[node->inputs->data[1]].dims->data[0],
            tensors[node->inputs->data[1]].dims->data[1],
            tensors[node->inputs->data[1]].dims->data[2],
            tensors[node->inputs->data[1]].dims->data[3]
          },
          false,
          false
        ));
        data.push_back(pyxir::XBuffer(
          (void *) tensors[node->inputs->data[2]].data.raw,
          4,
          "f",
          tensors[node->inputs->data[2]].dims->size,
          std::vector<ssize_t>{
            tensors[node->inputs->data[2]].dims->data[0]
          },
          false,
          false
        ));

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms,
          tops,
          layer,
          data
        );

        switch (depthwise_conv_params->padding) {
          case kTfLitePaddingUnknown: {
            printf("\t\tkTfLitePaddingUnknown\n");
            break;
          }
          case kTfLitePaddingSame: {
            printf("\t\tkTfLitePaddingSame\n");
            std::vector<std::vector<int64_t> > padding(4);
            {
              int64_t out_height = tensors[node->outputs->data[0]].dims->data[1];
              int64_t out_width = tensors[node->outputs->data[0]].dims->data[2];
              int64_t stride_height = depthwise_conv_params->stride_height;
              int64_t stride_width = depthwise_conv_params->stride_width;
              int64_t filter_height = tensors[node->inputs->data[1]].dims->data[1];
              int64_t filter_width = tensors[node->inputs->data[1]].dims->data[2];
              int64_t in_height = tensors[node->inputs->data[0]].dims->data[1];
              int64_t in_width = tensors[node->inputs->data[0]].dims->data[2];
              int64_t pad_along_height = std::max((out_height - 1) * stride_height + filter_height - in_height, (int64_t) 0);
              int64_t pad_along_width = std::max((out_width - 1) * stride_width + filter_width - in_width, (int64_t) 0);
              padding[0] = std::vector<int64_t>(2, 0);
              padding[1] = std::vector<int64_t>(2, 0);
              padding[2] = std::vector<int64_t>(2, 0);
              padding[3] = std::vector<int64_t>(2, 0);
              padding[1][0] = pad_along_height / 2;
              padding[1][1] = pad_along_height - padding[1][0];
              padding[2][0] = pad_along_width / 2;
              padding[2][1] = pad_along_width - padding[2][0];
            }
            x_layer.set_attr(std::string("padding"), pyxir::graph::XAttr(std::string("padding"), padding));
            x_layer.set_attr(std::string("padding_type"), pyxir::graph::XAttr(std::string("padding_type"), std::string("SAME")));
            break;
          }
          case kTfLitePaddingValid: {
            printf("\t\tkTfLitePaddingValid\n");
            std::vector<std::vector<int64_t> > padding(4);
            padding[0] = std::vector<int64_t>(2, 0);
            padding[1] = std::vector<int64_t>(2, 0);
            padding[2] = std::vector<int64_t>(2, 0);
            padding[3] = std::vector<int64_t>(2, 0);
            x_layer.set_attr(std::string("padding"), pyxir::graph::XAttr(std::string("padding"), padding));
            x_layer.set_attr(std::string("padding_type"), pyxir::graph::XAttr(std::string("padding_type"), std::string("VALID")));
            break;
          }
          default: {
            printf("\t\tkTfLitePaddingUnknown (default)\n");
            break;
          }
        }

        x_layer.set_attr(std::string("data_layout"), pyxir::graph::XAttr(std::string("data_layout"), std::string("NHWC")));

        x_layer.set_attr(std::string("kernel_layout"), pyxir::graph::XAttr(std::string("kernel_layout"), std::string("OHWI")));

        std::vector<int64_t> kernels(2);
        kernels[0] = tensors[node->inputs->data[1]].dims->data[1];
        kernels[1] = tensors[node->inputs->data[1]].dims->data[2];
        x_layer.set_attr(std::string("kernel_size"), pyxir::graph::XAttr(std::string("kernel_size"), kernels));

        std::vector<int64_t> strides(2);
        strides[0] = depthwise_conv_params->stride_height;
        strides[1] = depthwise_conv_params->stride_width;
        x_layer.set_attr(std::string("strides"), pyxir::graph::XAttr(std::string("strides"), strides));

        int groups = tensors[node->inputs->data[1]].dims->data[3];
        x_layer.set_attr(std::string("groups"), pyxir::graph::XAttr(std::string("groups"), groups));

        std::vector<int64_t> dilation_factors(2);
        dilation_factors[0] = depthwise_conv_params->dilation_height_factor;
        dilation_factors[1] = depthwise_conv_params->dilation_width_factor;
        x_layer.set_attr(std::string("dilation"), pyxir::graph::XAttr(std::string("dilation"), dilation_factors));

        std::vector<int64_t> channels(2);
        channels[0] = tensors[node->inputs->data[0]].dims->data[3];
        channels[1] = tensors[node->outputs->data[0]].dims->data[3];
        x_layer.set_attr(std::string("channels"), pyxir::graph::XAttr(std::string("channels"), channels));

        switch (depthwise_conv_params->activation) {
          case kTfLiteActNone: {
            printf("\t\tkTfLiteActNone\n");
            break;
          }
          case kTfLiteActRelu: {
            printf("\t\tkTfLiteActRelu\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU")));
            break;
          }
          case kTfLiteActRelu6: {
            printf("\t\tkTfLiteActRelu6\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU6")));
            break;
          }
          default: {
            printf("\t\tkTfLiteActNone (default)\n");
            break;
          }
        }

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate DEPTHWISE_CONV_2D node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMaxPool2DNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLitePoolParams* pool_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tMaxPool2d\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Pooling"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
          bottoms.push_back(tensors[node->inputs->data[0]].name);

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms
        );

        switch (pool_params->padding) {
          case kTfLitePaddingUnknown: {
            printf("\t\tkTfLitePaddingUnknown\n");
            break;
          }
          case kTfLitePaddingSame: {
            printf("\t\tkTfLitePaddingSame\n");
            std::vector<std::vector<int64_t> > padding(4);
            {
              int64_t out_height = tensors[node->outputs->data[0]].dims->data[1];
              int64_t out_width = tensors[node->outputs->data[0]].dims->data[2];
              int64_t stride_height = pool_params->stride_height;
              int64_t stride_width = pool_params->stride_width;
              int64_t filter_height = pool_params->filter_height;
              int64_t filter_width = pool_params->filter_width;
              int64_t in_height = tensors[node->inputs->data[0]].dims->data[1];
              int64_t in_width = tensors[node->inputs->data[0]].dims->data[2];
              int64_t pad_along_height = std::max((out_height - 1) * stride_height + filter_height - in_height, (int64_t) 0);
              int64_t pad_along_width = std::max((out_width - 1) * stride_width + filter_width - in_width, (int64_t) 0);
              padding[0] = std::vector<int64_t>(2, 0);
              padding[1] = std::vector<int64_t>(2, 0);
              padding[2] = std::vector<int64_t>(2, 0);
              padding[3] = std::vector<int64_t>(2, 0);
              padding[1][0] = pad_along_height / 2;
              padding[1][1] = pad_along_height - padding[1][0];
              padding[2][0] = pad_along_width / 2;
              padding[2][1] = pad_along_width - padding[2][0];
            }
            x_layer.set_attr(std::string("padding"), pyxir::graph::XAttr(std::string("padding"), padding));
            x_layer.set_attr(std::string("padding_type"), pyxir::graph::XAttr(std::string("padding_type"), std::string("SAME")));
            break;
          }
          case kTfLitePaddingValid: {
            printf("\t\tkTfLitePaddingValid\n");
            std::vector<std::vector<int64_t> > padding(4);
            padding[0] = std::vector<int64_t>(2, 0);
            padding[1] = std::vector<int64_t>(2, 0);
            padding[2] = std::vector<int64_t>(2, 0);
            padding[3] = std::vector<int64_t>(2, 0);
            x_layer.set_attr(std::string("padding"), pyxir::graph::XAttr(std::string("padding"), padding));
            x_layer.set_attr(std::string("padding_type"), pyxir::graph::XAttr(std::string("padding_type"), std::string("VALID")));
            break;
          }
          default: {
            printf("\t\tkTfLitePaddingUnknown (default)\n");
            break;
          }
        }

        x_layer.set_attr(std::string("data_layout"), pyxir::graph::XAttr(std::string("data_layout"), std::string("NHWC")));

        std::vector<int64_t> filters(2);
        filters[0] = pool_params->filter_height;
        filters[1] = pool_params->filter_width;
        x_layer.set_attr(std::string("kernel_size"), pyxir::graph::XAttr(std::string("kernel_size"), filters));

        std::vector<int64_t> strides(2);
        strides[0] = pool_params->stride_height;
        strides[1] = pool_params->stride_width;
        x_layer.set_attr(std::string("strides"), pyxir::graph::XAttr(std::string("strides"), strides));

        std::vector<int64_t> insize(2);
        insize[0] = tensors[node->inputs->data[0]].dims->data[1];
        insize[1] = tensors[node->inputs->data[0]].dims->data[2];
        x_layer.set_attr(std::string("insize"), pyxir::graph::XAttr(std::string("insize"), insize));

        std::vector<int64_t> outsize(2);
        outsize[0] = tensors[node->outputs->data[0]].dims->data[1];
        outsize[1] = tensors[node->outputs->data[0]].dims->data[2];
        x_layer.set_attr(std::string("outsize"), pyxir::graph::XAttr(std::string("outsize"), outsize));

        x_layer.set_attr(std::string("pool_type"), pyxir::graph::XAttr(std::string("pool_type"), std::string("Max")));

        switch (pool_params->activation) {
          case kTfLiteActNone: {
            printf("\t\tkTfLiteActNone\n");
            break;
          }
          case kTfLiteActRelu: {
            printf("\t\tkTfLiteActRelu\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU")));
            break;
          }
          case kTfLiteActRelu6: {
            printf("\t\tkTfLiteActRelu6\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU6")));
            break;
          }
          default: {
            printf("\t\tkTfLiteActNone (default)\n");
            break;
          }
        }

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate MAX_POOL_2D node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMulNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteMulParams* mul_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tMul\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Scale"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
          bottoms.push_back(tensors[node->inputs->data[0]].name);
        std::vector<std::string> tops = std::vector<std::string>();
        std::vector<std::string> layer = std::vector<std::string>(1, name);
        std::vector<pyxir::XBuffer> data;
        data.push_back(pyxir::XBuffer(
          (void *) tensors[node->inputs->data[1]].data.raw,
          4,
          "f",
          tensors[node->inputs->data[1]].dims->size,
          std::vector<ssize_t>{
            tensors[node->inputs->data[1]].dims->data[0]
          },
          false,
          false
        ));
        std::vector<float> bias(tensors[node->inputs->data[1]].dims->data[0], 0);
        data.push_back(pyxir::XBuffer(
          (void *) bias.data(),
          4,
          "f",
          tensors[node->inputs->data[1]].dims->size,
          std::vector<ssize_t>{
            tensors[node->inputs->data[1]].dims->data[0]
          },
          false,
          false
        ));

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms,
          tops,
          layer,
          data
        );

        x_layer.set_attr(std::string("axis"), pyxir::graph::XAttr(std::string("axis"), -1));

        switch (mul_params->activation) {
          case kTfLiteActNone: {
            printf("\t\tkTfLiteActNone\n");
            break;
          }
          case kTfLiteActRelu: {
            printf("\t\tkTfLiteActRelu\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU")));
            break;
          }
          case kTfLiteActRelu6: {
            printf("\t\tkTfLiteActRelu6\n");
            x_layer.set_attr(std::string("activation"), pyxir::graph::XAttr(std::string("activation"), std::string("ReLU6")));
            break;
          }
          default: {
            printf("\t\tkTfLiteActNone (default)\n");
            break;
          }
        }

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate MUL node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitReluNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors) {
    if (x_graph != nullptr) {
      try {
        printf("\tRelu\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("ReLU"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
          bottoms.push_back(tensors[node->inputs->data[0]].name);

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms
        );

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate RELU node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitPadNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLitePadParams* pad_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tPad\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Pad"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
          bottoms.push_back(tensors[node->inputs->data[0]].name);

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms
        );

        std::vector<std::vector<int64_t> > padding(4);
        padding[0] = std::vector<int64_t>(2, 0);
        padding[1] = std::vector<int64_t>(2, 0);
        padding[2] = std::vector<int64_t>(2, 0);
        padding[3] = std::vector<int64_t>(2, 0);
        padding[0][0] = tensors[node->inputs->data[1]].data.i32[0 * 2 + 0];
        padding[0][1] = tensors[node->inputs->data[1]].data.i32[0 * 2 + 1];
        padding[1][0] = tensors[node->inputs->data[1]].data.i32[1 * 2 + 0];
        padding[1][1] = tensors[node->inputs->data[1]].data.i32[1 * 2 + 1];
        padding[2][0] = tensors[node->inputs->data[1]].data.i32[2 * 2 + 0];
        padding[2][1] = tensors[node->inputs->data[1]].data.i32[2 * 2 + 1];
        padding[3][0] = tensors[node->inputs->data[1]].data.i32[3 * 2 + 0];
        x_layer.set_attr(std::string("padding"), pyxir::graph::XAttr(std::string("padding"), padding));

        float pad_value = 0;
        x_layer.set_attr(std::string("pad_value"), pyxir::graph::XAttr(std::string("pad_value"), pad_value));

        x_layer.set_attr(std::string("data_layout"), pyxir::graph::XAttr(std::string("data_layout"), std::string("NHWC")));

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate PAD node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitMeanNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteReducerParams* reducer_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tMean\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Mean"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
          bottoms.push_back(tensors[node->inputs->data[0]].name);

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms
        );

        std::vector<int64_t> axis;
        for (int j = 0; j < tensors[node->inputs->data[1]].dims->data[0]; ++j) {
          axis.push_back(tensors[node->inputs->data[1]].data.i32[j]);
        }
        x_layer.set_attr(std::string("axes"), pyxir::graph::XAttr(std::string("axes"), axis));

        bool keep_dims = reducer_params->keep_dims;
        x_layer.set_attr(std::string("keepdims"), pyxir::graph::XAttr(std::string("keepdims"), keep_dims));

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate MEAN node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitLeakyReluNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors,
      const TfLiteLeakyReluParams* leaky_relu_params) {
    if (x_graph != nullptr) {
      try {
        printf("\tLeakyRelu\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("LeakyReLU"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
          bottoms.push_back(tensors[node->inputs->data[0]].name);

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms
        );

        float alpha = leaky_relu_params->alpha;
        x_layer.set_attr(std::string("alpha"), pyxir::graph::XAttr(std::string("alpha"), alpha));

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate LEAKY_RELU node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

  static TfLiteStatus VisitDefaultNode(
      std::shared_ptr<pyxir::graph::XGraph> x_graph, TfLiteContext* logging_context, int node_index,
      TfLiteNode* node, const TfLiteTensor* tensors) {
    if (x_graph != nullptr) {
      try {
        printf("\tDefault\n");

        std::string name = std::string(tensors[node->outputs->data[0]].name);
        std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Default"));
        std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
        shapes[0].push_back(-1);
        for (int j = 1; j < tensors[node->outputs->data[0]].dims->size; ++j) {
          shapes[0].push_back(tensors[node->outputs->data[0]].dims->data[j]);
        }
        std::string shapes_t = std::string("TensorShape");
        std::vector<int64_t> sizes = std::vector<int64_t>();
        std::vector<std::string> bottoms = std::vector<std::string>();
        for (int j = 0; j < node->inputs->size; ++j) {
          if (tensors[node->inputs->data[j]].data.raw == nullptr) {
            bottoms.push_back(tensors[node->inputs->data[j]].name);
          }
        }

        pyxir::graph::XLayer x_layer(
          name,
          xtype,
          shapes,
          shapes_t,
          sizes,
          bottoms
        );

        x_graph->add(x_layer);
      }
      catch (...) {
        //TF_LITE_KERNEL_LOG(logging_context, "failed to delegate DEFAULT node #%d",
        //                   node_index);
        return kTfLiteError;
      }
    }

    return kTfLiteOk;
  }

 private:
  Subgraph(pyxir::RtModHolder runtime, std::vector<int> inputs,
           std::vector<int> outputs) : runtime_(std::move(runtime)),
           inputs_(inputs), outputs_(outputs) {}

  // VITISAI Runtime (subgraph + workspace) with smart-pointer for lifetime
  // management.
  pyxir::RtModHolder runtime_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
  bool first_run_{true};
};

TfLiteIntArray* Delegate::PrepareOpsToDelegate(TfLiteContext* context) {
  TfLiteIntArray* execution_plan = nullptr;
  if (context->GetExecutionPlan(context, &execution_plan) != kTfLiteOk) {
    //TF_LITE_KERNEL_LOG(context, "Unable to get graph execution plan.");
    return nullptr;
  }

  TfLiteIntArray* nodes_to_delegate =
      TfLiteIntArrayCreate(execution_plan->size);
  nodes_to_delegate->size = 0;

  std::shared_ptr<pyxir::graph::XGraph> x_graph(new pyxir::graph::XGraph(std::string("")));

  TfLiteContext* logging_context = x_graph == nullptr ? nullptr : context;

  std::vector<int> inputs;
  for (int i = 0; i < context->tensors_size; ++i) {
    if (context->tensors[i].name == nullptr) {
      continue;
    }

    if (context->tensors[i].data.raw != nullptr) {
      continue;
    }

    bool is_output = false;

    for (int j = 0; j < execution_plan->size; ++j) {
      const int node_index = execution_plan->data[j];

      // Check if TFLite nodes can be delegated to VITISAI
      TfLiteNode* node = nullptr;
      TfLiteRegistration* registration = nullptr;
      if (context->GetNodeAndRegistration(context, node_index, &node,
                                          &registration) != kTfLiteOk) {
        //TF_LITE_KERNEL_LOG(context,
        //                   "Unable to get node and registration for node %d.",
        //                   node_index);
        continue;  // Soft error (skip this node).
      }

      for (int m = 0; m < node->outputs->size; ++m) {
        if (!strcmp(context->tensors[node->outputs->data[m]].name, context->tensors[i].name)) {
          is_output = true;
          break;
        }
      }

      if (is_output) {
        break;
      }
    }

    if (!is_output) {
      inputs.push_back(i);
    }
  }

  for (int i : inputs) {
    std::string name = context->tensors[i].name;
    std::vector<std::string> xtype = std::vector<std::string>(1, std::string("Input"));
    std::vector<std::vector<int64_t>> shapes = std::vector<std::vector<int64_t>>(1);
    shapes[0].push_back(-1);
    for (int j = 1; j < context->tensors[i].dims->size; ++j) {
      shapes[0].push_back(context->tensors[i].dims->data[j]);
    }
    std::string shapes_t = std::string("TensorShape");
    std::vector<int64_t> sizes = std::vector<int64_t>();
    std::vector<std::string> bottoms = std::vector<std::string>();
    std::vector<std::string> tops = std::vector<std::string>();
    std::vector<std::string> layer = std::vector<std::string>(1, name);

    pyxir::graph::XLayer x_layer(
      name,
      xtype,
      shapes,
      shapes_t,
      sizes,
      bottoms,
      tops,
      layer
    );

    x_layer.set_attr(std::string("data_layout"), pyxir::graph::XAttr(std::string("data_layout"), std::string("NHWC")));

    x_graph->add(x_layer);
  }

  std::cout << std::endl;

  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];

    // Check if TFLite nodes can be delegated to VITISAI
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      //TF_LITE_KERNEL_LOG(context,
      //                   "Unable to get node and registration for node %d.",
      //                   node_index);
      continue;  // Soft error (skip this node).
    }

    printf("%d of %d, buildin_code: %d, inputs: %s, outputs: %s\n", node_index + 1, execution_plan->size, registration->builtin_code, context->tensors[node->inputs->data[0]].name, context->tensors[node->outputs->data[0]].name);

    if (Subgraph::VisitNode(x_graph, context, registration, node,
                  node_index) != kTfLiteOk) {
      return nullptr;
    }
  }

  std::cout << "x_graph's num layers: " << x_graph->len() << std::endl;
  std::cout << std::endl;

  std::vector<std::string> heads = x_graph->get_input_names();
  std::cout << "x_graph's num head(s): " << heads.size() << std::endl;
  std::cout << "x_graph's head(s):" << std::endl;
  for (int j = 0; j < heads.size(); ++j) {
    std::cout << heads[j] << std::endl;
  }
  std::cout << std::endl;

  std::vector<std::string> tails = x_graph->get_output_names();
  std::cout << "x_graph's num tail(s): " << tails.size() << std::endl;
  std::cout << "x_graph's tail(s):" << std::endl;
  for (int j = 0; j < tails.size(); ++j) {
    std::cout << tails[j] << std::endl;
  }
  std::cout << std::endl;



  // Call XGraph's partitioner
  std::string backend_type = options_.target;

  // resnet101
  //std::string last_layer("resnet_v2_101/Pad");
  //std::string last_layer("resnet_v2_101/conv1/BiasAdd");
  //std::string last_layer("resnet_v2_101/pool1/MaxPool");
  //...
  //std::string last_layer("resnet_v2_101/logits/BiasAdd");
  //std::string last_layer("");

  // inception_v3
  //std::string last_layer("InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu");
  //std::string last_layer("");

  std::string last_layer("");

  pyxir::partition(x_graph, std::vector<std::string>(1, backend_type), last_layer);



  for (int i = 0; i < execution_plan->size; ++i) {
    const int node_index = execution_plan->data[i];

    // Check if TFLite nodes can be delegated to VITISAI
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    if (context->GetNodeAndRegistration(context, node_index, &node,
                                        &registration) != kTfLiteOk) {
      //TF_LITE_KERNEL_LOG(context,
      //                   "Unable to get node and registration for node %d.",
      //                   node_index);
      continue;  // Soft error (skip this node).
    }

    std::shared_ptr<pyxir::graph::XLayer> x_layer = x_graph->get(context->tensors[node->outputs->data[0]].name);
    std::cout << "x_layer name: " << x_layer->name << std::endl;
    for (int i = 0; i < x_layer->xtype.size(); ++i) {
      std::cout << "x_layer xtype: " << x_layer->xtype[i] << std::endl;
    }
    for (int i = 0; i < x_layer->layer.size(); ++i) {
      std::cout << "x_layer layer: " << x_layer->layer[i] << std::endl;
    }
    std::cout << "x_layer target: " << x_layer->target << std::endl;
    std::cout << "x_layer subgraph: " << x_layer->subgraph << std::endl;

    if (x_layer->subgraph.compare(0, 2, "xp") != 0) {
      // Non-delegatable node is not an error.
      continue;
    }

    nodes_to_delegate->data[nodes_to_delegate->size++] = node_index;
  }

  return nodes_to_delegate;
}

void* SubgraphInit(TfLiteContext* context, const char* buffer, size_t length) {
  const TfLiteDelegateParams* params =
      reinterpret_cast<const TfLiteDelegateParams*>(buffer);

  return static_cast<void*>(Subgraph::Create(
      context, params,
      static_cast<::tflite::vitisai::Delegate*>(params->delegate->data_)));
}

TfLiteStatus SubgraphPrepare(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Prepare(context);
}

TfLiteStatus SubgraphInvoke(TfLiteContext* context, TfLiteNode* node) {
  if (node->user_data == nullptr) {
    return kTfLiteError;
  }

  return static_cast<Subgraph*>(node->user_data)->Invoke(context);
}

void SubgraphFree(TfLiteContext* context, void* buffer) {
  if (buffer != nullptr) {
    delete static_cast<Subgraph*>(buffer);
  }
}

const TfLiteRegistration kSubgraphRegistration = {
    /*.init=*/SubgraphInit,
    /*.free=*/SubgraphFree,
    /*.prepare=*/SubgraphPrepare,
    /*.invoke=*/SubgraphInvoke,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/0,
    /*.custom_name=*/"TfLiteVitisAIDelegate",
    /*.version=*/2,
};

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* ops_to_replace =
      static_cast<::tflite::vitisai::Delegate*>(delegate->data_)
          ->PrepareOpsToDelegate(context);
  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kSubgraphRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}

}  // namespace
}  // namespace vitisai
}  // namespace tflite

TfLiteVitisAIDelegateOptions TfLiteVitisAIDelegateOptionsDefault() {
  TfLiteVitisAIDelegateOptions options;
  strcpy(options.target, "");
  return options;
}

TfLiteDelegate* TfLiteVitisAIDelegateCreate(
    const TfLiteVitisAIDelegateOptions* options) {
  auto* vitisai_delegate = new ::tflite::vitisai::Delegate(options);
  return vitisai_delegate ? vitisai_delegate->tflite_delegate() : nullptr;
}

void TfLiteVitisAIDelegateDelete(TfLiteDelegate* delegate) {
  if (delegate != nullptr) {
    delete static_cast<::tflite::vitisai::Delegate*>(delegate->data_);
  }
}

TfLiteDelegate* tflite_plugin_create_delegate(
    char** options_keys,
    char** options_values,
    size_t num_options,
    ErrorHandler error_handler) {
  TfLiteVitisAIDelegateOptions options;
  if (num_options == 0) {
    options = TfLiteVitisAIDelegateOptionsDefault();
  } else {
    for (size_t i = 0; i < num_options; ++i) {
      if (strcmp(options_keys[i], "target") == 0) {
        if (strlen(options_values[i]) < 32) {
          strcpy(options.target, options_values[i]);
        }
      }
    }
  }
  TfLiteDelegate* delegate = TfLiteVitisAIDelegateCreate(&options);
  return delegate;
}

void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  TfLiteVitisAIDelegateDelete(delegate);
}
