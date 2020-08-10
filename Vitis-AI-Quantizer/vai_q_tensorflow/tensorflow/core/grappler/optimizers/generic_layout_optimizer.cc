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

#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/clusters/cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer.h"
#include "tensorflow/core/grappler/optimizers/generic_layout_optimizer_transposer_factory.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kNHWC[] = "NHWC";
constexpr char kNCHW[] = "NCHW";
constexpr float kVoltaGPURatioThreshold = 0.5;
constexpr float kConv2DGPUFP16Threshold = 0.5;

struct MutableNodeViewFormatter {
  void operator()(std::string* out, utils::MutableNodeView* node_view) const {
    absl::StrAppend(out, node_view->node()->name());
  }
};

inline std::pair<int, int> GetNumGPUs(const Cluster& cluster) {
  auto devices = cluster.GetDevices();
  int num_gpus = 0;
  int num_volta = 0;
  for (const auto& device : devices) {
    if (device.second.type() != kGPU) {
      continue;
    }
    num_gpus++;
    auto compute_capability_it =
        device.second.environment().find("architecture");
    if (compute_capability_it == device.second.environment().end()) {
      continue;
    }
    double compute_capability = 0.0;
    if (absl::SimpleAtod(compute_capability_it->second, &compute_capability) &&
        compute_capability >= 7.0) {
      num_volta++;
    }
  }
  return {num_gpus, num_volta};
}

inline bool NumConv2DOnDeviceWithDataTypeOverThreshold(
    const TransposeContext& context, absl::string_view device,
    const DataType& data_type) {
  int num_conv2d_gpu = 0;
  int num_conv2d_gpu_fp16 = 0;

  for (const auto& node : context.graph_view->GetNodes()) {
    const auto* node_def = node.node();
    if (!IsConv2D(*node_def)) {
      continue;
    }
    const string& device_name =
        GetDeviceName(context.virtual_placer.get(), *node_def);
    string device_type;
    string task;
    if (!DeviceNameUtils::SplitDeviceName(device_name, &task, &device_type) ||
        !absl::StrContains(absl::AsciiStrToLower(device_type),
                           absl::AsciiStrToLower(device))) {
      continue;
    }
    num_conv2d_gpu++;
    const auto* t_attr = node.GetAttr("T");
    if (t_attr == nullptr) {
      continue;
    }
    if (t_attr->type() == data_type) {
      num_conv2d_gpu_fp16++;
    }
  }

  if (num_conv2d_gpu == 0) return false;

  return (static_cast<float>(num_conv2d_gpu_fp16) /
          static_cast<float>(num_conv2d_gpu)) >= kConv2DGPUFP16Threshold;
}

inline std::pair<string, string> GetSrcAndDstDataFormats(
    const TransposeContext& context, int num_gpus, int num_voltas) {
  string src_format = kNHWC;
  string dst_format = kNCHW;
  if (((static_cast<float>(num_voltas) / static_cast<float>(num_gpus)) >=
       kVoltaGPURatioThreshold) &&
      NumConv2DOnDeviceWithDataTypeOverThreshold(context, kGPU, DT_HALF)) {
    std::swap(src_format, dst_format);
  }
  return {src_format, dst_format};
}

Status ExpandLayoutSensitiveOp(TransposeContext* context,
                               TransposerFactory* transposer_factory) {
  const int num_nodes = context->num_nodes;
  for (int i = 0; i < num_nodes; ++i) {
    auto* node_view = context->graph_view->GetNode(i);
    auto* node_def = node_view->node();
    if (IsLayoutSensitiveOp(*node_def)) {
      std::shared_ptr<Transposer> transposer =
          transposer_factory->GetTransposer(*node_def);
      if (transposer == nullptr) {
        return Status(
            error::NOT_FOUND,
            absl::StrCat(
                "Layout sensitive operation should have a transposer. Node: ",
                node_def->DebugString()));
      }
      TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
    }
  }
  return Status::OK();
}

Status ExpandLayoutAgnosticOp(TransposeContext* context,
                              TransposerFactory* transposer_factory) {
  const int num_nodes = context->num_nodes;
  for (int i = 0; i < num_nodes; ++i) {
    auto* node_view = context->graph_view->GetNode(i);
    auto* node_def = node_view->node();
    if (IsLayoutAgnosticOp(*node_def)) {
      const auto& transposer = transposer_factory->GetTransposer(*node_def);
      if (transposer == nullptr) {
        return Status(
            error::NOT_FOUND,
            absl::StrCat(
                "Layout agnostic operation should have a transposer. Node: ",
                node_def->DebugString()));
      }
      TF_RETURN_IF_ERROR(transposer->TransposeNode(context, node_view));
    }
  }
  return Status::OK();
}

inline bool IsCancellableConstPermTransposeNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
  Tensor fanout_tensor;
  if (!GetValueAttrFromConstInputNode(fanout_transpose, IsTranspose, 1,
                                      &fanout_tensor)) {
    return false;
  }
  Tensor fanin_tensor;
  if (!GetValueAttrFromConstInputNode(fanin_transpose, IsTranspose, 1,
                                      &fanin_tensor)) {
    return false;
  }
  if (fanout_tensor.NumElements() != fanin_tensor.NumElements()) {
    return false;
  }

  // Using dst->src to permute on src->dst will result in
  // seq(0, ..., num_elements - 1) if they are cancellable.
  const auto& fanout_tensor_data = fanout_tensor.unaligned_flat<int32>();
  const auto& fanin_tensor_data = fanin_tensor.unaligned_flat<int32>();
  const int num_elements = fanout_tensor.NumElements();
  for (int i = 0; i < num_elements; ++i) {
    if (fanout_tensor_data(fanin_tensor_data(i)) != i) {
      return false;
    }
  }
  return true;
}

inline bool IsCancellableDataFormatNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
  if (!IsDataFormatOp(fanout_transpose) || !IsDataFormatOp(fanin_transpose)) {
    return false;
  }

  auto src_dst_match = [](const utils::MutableNodeView& src,
                          const utils::MutableNodeView& dst) {
    const auto* src_format = src.GetAttr(kAttrSrcFormat);
    if (src_format == nullptr) {
      return false;
    }
    const auto* dst_format = dst.GetAttr(kAttrDstFormat);
    if (dst_format == nullptr) {
      return false;
    }
    return src_format->s() == dst_format->s();
  };

  // If src_format node A is equal to dst_format of node B and dst_format of
  // node A is equal to src_format of node B, then they are cancellable.
  return src_dst_match(fanin_transpose, fanout_transpose) &&
         src_dst_match(fanout_transpose, fanin_transpose);
}

inline bool IsCancellableNodePair(
    const utils::MutableNodeView& fanout_transpose,
    const utils::MutableNodeView& fanin_transpose) {
  return IsCancellableConstPermTransposeNodePair(fanout_transpose,
                                                 fanin_transpose) ||
         IsCancellableDataFormatNodePair(fanout_transpose, fanin_transpose);
}

Status EraseCancellableNodes(TransposeContext* context) {
  const int original_num_nodes = context->num_nodes;
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  const int num_nodes = graph_view->NumNodes();

  for (int i = original_num_nodes; i < num_nodes; ++i) {
    auto* node = graph_view->GetNode(i);
    if (node->NumRegularFanins() < 1) {
      continue;
    }
    const auto& regular_fanin_0 = node->GetRegularFanin(0);
    auto* fanin_node = regular_fanin_0.node_view();
    // TODO(lyandy): Lift restriction once original nodes in the graph can be
    // pruned away.
    if (fanin_node->node_index() < original_num_nodes) {
      continue;
    }
    if (!IsCancellableNodePair(*node, *fanin_node)) {
      continue;
    }
    const auto& fanin_to_forward = fanin_node->GetRegularFanin(0);
    TensorId fanin_id_to_forward(fanin_to_forward.node_view()->GetName(),
                                 fanin_to_forward.index());
    for (const auto& regular_fanout : node->GetRegularFanout(0)) {
      mutation->AddOrUpdateRegularFanin(regular_fanout.node_view(),
                                        regular_fanout.index(),
                                        fanin_id_to_forward);
    }
    mutation->RemoveNode(node);
    if (node->NumRegularFanins() > 1) {
      mutation->RemoveNode(node->GetRegularFanin(1).node_view());
    }
    mutation->RemoveNode(fanin_node);
    if (fanin_node->NumRegularFanins() > 1) {
      mutation->RemoveNode(fanin_node->GetRegularFanin(1).node_view());
    }
  }
  return mutation->Apply();
}

// TODO(ezhulenev): This is a temporary workaround for a graph pattern
// in Resnet models. We should be able to push down transpose nodes across Pad
// and many other ops, and then rely on cancellation to remove them.
//
// From: Transpose[NHWC->NCHW] -> Pad[paddings] -> Transpose[NCHW->NHWC]
// To:   Pad[Permute(paddings)]
Status EraseCancellableNodesAroundPad(TransposeContext* context) {
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();

  absl::flat_hash_set<utils::MutableNodeView*> cancelled_transposes;

  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    // Transpose node after Pad.
    auto* transpose_after = graph_view->GetNode(i);
    if (!IsTranspose(*transpose_after->node())) continue;

    // This transpose was already cancelled in previous loop iteration.
    if (cancelled_transposes.contains(transpose_after)) continue;

    // Pad node.
    const auto& transpose_after_fanin = transpose_after->GetRegularFanin(0);
    auto* pad = transpose_after_fanin.node_view();
    if (!IsPad(*pad->node())) continue;

    // Transpose node before Pad.
    const auto& pad_fanin_0 = pad->GetRegularFanin(0);
    auto* transpose_before = pad_fanin_0.node_view();
    if (!IsTranspose(*transpose_before->node())) continue;

    // Transpose before output used once by the Pad node.
    if (transpose_before->NumRegularFanouts() != 1) continue;

    // Transposes are cancellable.
    if (!IsCancellableConstPermTransposeNodePair(*transpose_after,
                                                 *transpose_before))
      continue;

    // Paddings are known constant values.
    Tensor paddings_t;
    if (!GetValueAttrFromConstInputNode(*pad, IsPad, 1, &paddings_t)) continue;

    // Paddings value used once by the pad node only.
    const auto& pad_fanin_1 = pad->GetRegularFanin(1);
    auto* paddings = pad_fanin_1.node_view();
    if (paddings->NumRegularFanouts() != 1) continue;

    // Get permutation after the padding.
    Tensor permute_t;
    if (!GetValueAttrFromConstInputNode(*transpose_after, IsTranspose, 1,
                                        &permute_t))
      continue;

    // Pad output might be used multiple times by different Transpose nodes. If
    // they all have identical permutation, we can cancel all of them.
    std::vector<utils::MutableNodeView*> pad_fanout_transposes;
    pad_fanout_transposes.emplace_back(transpose_after);

    bool pad_has_unsupported_fanout = false;
    for (auto& fanout : pad->GetRegularFanout(0)) {
      auto* extra_transpose = fanout.node_view();
      if (extra_transpose == transpose_after) continue;

      // Check that fanout is a Transpose identical to the transpose_after.
      Tensor extra_permute_t;
      if (!GetValueAttrFromConstInputNode(*extra_transpose, IsTranspose, 1,
                                          &extra_permute_t) ||
          extra_permute_t.tensor_data() != permute_t.tensor_data()) {
        pad_has_unsupported_fanout = true;
        break;
      }

      pad_fanout_transposes.emplace_back(extra_transpose);
    }
    if (pad_has_unsupported_fanout) continue;

    VLOG(0) << "Cancel Transpose nodes around Pad:"
            << " transpose_before=" << transpose_before->node()->name()
            << " pad=" << pad->node()->name() << " transpose_after="
            << absl::StrJoin(pad_fanout_transposes, ",",
                             MutableNodeViewFormatter());

    // Permute paddings in place according to permutation in second transpose.
    auto permutation_s = absl::Span<int32>(permute_t.flat<int32>().data(),
                                           permute_t.NumElements());
    auto paddings_s = absl::Span<int32>(paddings_t.flat<int32>().data(),
                                        paddings_t.NumElements());
    TF_RETURN_IF_ERROR(
        PermuteDouble(absl::StrCat("paddings in ", pad->GetName()),
                      permutation_s, &paddings_s));

    // Update paddings constant value with a permuted tensor.
    AttrValue permuted_paddings_tensor;
    paddings_t.AsProtoTensorContent(permuted_paddings_tensor.mutable_tensor());
    mutation->AddOrUpdateNodeAttr(paddings, "value", permuted_paddings_tensor);

    // Transform Transpose nodes into Identity nodes.
    const auto transpose_to_identity =
        [&cancelled_transposes,
         &mutation](utils::MutableNodeView* transpose) -> void {
      mutation->UpdateNodeOp(transpose, "Identity");
      mutation->RemoveNodeAttr(transpose, "Tperm");
      mutation->RemoveRegularFanin(transpose, 1);
      cancelled_transposes.insert(transpose);
    };

    transpose_to_identity(transpose_before);
    absl::c_for_each(pad_fanout_transposes, transpose_to_identity);
  }

  return mutation->Apply();
}

Status EraseOutputShapeAttrs(TransposeContext* context) {
  utils::MutableGraphView* graph_view = context->graph_view.get();
  utils::Mutation* mutation = graph_view->GetMutationBuilder();
  const int num_nodes = graph_view->NumNodes();
  for (int i = 0; i < num_nodes; ++i) {
    mutation->RemoveNodeAttr(graph_view->GetNode(i), kAttrOutputShape);
    TF_RETURN_IF_ERROR(mutation->Apply());
  }
  return Status::OK();
}

}  // namespace

Status GenericLayoutOptimizer::Optimize(Cluster* cluster,
                                        const GrapplerItem& item,
                                        GraphDef* output) {
  if (cluster == nullptr) {
    LOG(WARNING)
        << "generic layout optimizer was called with cluster == nullptr";
    return errors::Aborted("cluster == nullptr.");
  }
  const auto num_gpus_and_num_volta = GetNumGPUs(*cluster);
  const int num_gpus = num_gpus_and_num_volta.first;
  if (num_gpus < 1) {
    return errors::Aborted(
        "No GPUs found: GenericLayoutOptimizer is currently only tuned for "
        "GPU.");
  }

  const bool is_aggressive = opt_level_ == RewriterConfig::AGGRESSIVE;

  TransposeContext context;
  TF_RETURN_IF_ERROR(
      TransposeContext::InitializeTransposeContext(item, cluster, &context));

  const auto src_dst_formats =
      GetSrcAndDstDataFormats(context, num_gpus, num_gpus_and_num_volta.second);
  context.AssignDeviceAndDataFormats(kGPU, src_dst_formats.first,
                                     src_dst_formats.second);

  TransposerFactory transposer_factory;
  TF_RETURN_IF_ERROR(ExpandLayoutSensitiveOp(&context, &transposer_factory));
  if (context.graph.node_size() > context.num_nodes || is_aggressive) {
    TF_RETURN_IF_ERROR(ExpandLayoutAgnosticOp(&context, &transposer_factory));
    TF_RETURN_IF_ERROR(EraseCancellableNodes(&context));
    TF_RETURN_IF_ERROR(EraseCancellableNodesAroundPad(&context));
    // TODO(lyandy): Remove sorting once other optimizers are migrated to using
    // `utils::GraphView`.
    TF_RETURN_IF_ERROR(
        context.graph_view->SortTopologically(/*ignore_cycles=*/false, {}));
  }
  TF_RETURN_IF_ERROR(EraseOutputShapeAttrs(&context));

  *output = context.graph;
  return Status::OK();
}

void GenericLayoutOptimizer::Feedback(Cluster* cluster,
                                      const GrapplerItem& item,
                                      const GraphDef& optimize_output,
                                      double result) {
  // Takes no feedback.
}

}  // end namespace grappler
}  // end namespace tensorflow
