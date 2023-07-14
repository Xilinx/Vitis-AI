/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "vitis/ai/extra_ops.hpp"

#include "xir/op/op_def.hpp"
namespace xir {
class OpDefFactoryImp : public OpDefFactory {
 public:
  void register_h(const OpDef& def) override;
  const OpDef* create(const std::string& type) const;
  const std::vector<std::string> get_registered_ops() const;

 private:
  std::unordered_map<std::string, OpDef> store_;
};

const OpDefFactoryImp* op_def_factory();

}  // namespace xir

static void replace_shape(xir::Op* op, const std::vector<int>& new_out_shape) {
  auto out = op->get_output_tensor();
  auto output_tensor =
      xir::Tensor::create(out->get_name(), new_out_shape, out->get_data_type());
  output_tensor->set_attrs(out->get_attrs());
  op->replace_output_tensor(std::move(output_tensor));
}
static std::vector<xir::OpDef> my_op_defs() {
  using OpDef = xir ::OpDef;
  using OpArgDef = xir ::OpArgDef;
  using AttrDef = xir::AttrDef;
  auto FLOAT32 = xir::DataType::FLOAT;

  return {
      // TOPK
      OpDef("topk")  //
          .add_input_arg(
              xir::OpArgDef{"input", OpArgDef::REQUIRED, FLOAT32,
                            "An input tensor with shape "
                            "`[batch, in_height, in_width, in_channels]`."})
          .add_attr(xir::AttrDefBuilder<int>::build("K", AttrDef::REQUIRED,
                                                    "`Datatype`: `int`\n\n"
                                                    "top K"))
          .set_shape_infer([](xir::Op* op) {
            auto shape = op->get_input_tensor("input")->get_shape();
            // CHECK_EQ(shape.size(), 2u) << "input shape mismach, must be
            // 2-dimension";
            auto k = op->get_attr<int>("K");
            shape[shape.size() - 1] = 2 * std::min(k, shape[shape.size() - 1]);
            replace_shape(op, shape);
          })
          .set_annotation("usually, topk op is a successor of softmax Op"),
      OpDef("compare")  //
          .add_input_arg(
              xir::OpArgDef{"input", OpArgDef::REQUIRED, FLOAT32,
                            "An input tensor with shape "
                            "`[batch, in_height, in_width, in_channels]`."})
          .add_attr(AttrDef{"baseline", xir::TYPE_INDEX_BYTES_VEC,
                            AttrDef::OPTIONAL, 0,
                            "`Datatype`: `vector<bytes>`\n\n"
                            "reference results",
                            std::vector<std::vector<char>>()})
          .add_attr(xir::AttrDefBuilder<std::vector<std::string>>::build(
              "from_file", AttrDef::OPTIONAL, 0, "`Datatype`: `string`\n\n",
              {}))
          .add_attr(xir::AttrDefBuilder<std::vector<std::string>>::build(
              "md5sum", AttrDef::OPTIONAL, 0, "`Datatype`: `string`\n\n", {}))
          .add_attr(xir::AttrDefBuilder<int>::build(
              "log_limit", AttrDef::OPTIONAL,
              "`Datatype`: `int`\n\n"
              "if the number of errors less than log_limit, print out diff. 0 "
              "means disable log",
              10))
          .add_attr(xir::AttrDefBuilder<bool>::build(
              "save_on_error", AttrDef::OPTIONAL,
              "`Datatype`: `bool`\n\n"
              "whether dump result in case of errors",
              true))
          .add_attr(xir::AttrDefBuilder<std::string>::build(
              "dump_directory", AttrDef::OPTIONAL,
              "`Datatype`: `string`\n\n"
              "the directory for dumping input tensor buffers",
              "dump"))
          .set_shape_infer([](xir::Op* op) {
            auto out = op->get_output_tensor();
            auto shape = out->get_shape();
            auto input = op->get_input_tensor("input", 0);
            auto input_shape = input->get_shape();
            CHECK_GE(input_shape.size(), 1u);
            auto output_tensor =
                xir::Tensor::create(out->get_name(), {input_shape[0], 2, 32},
                                    {xir::DataType::XINT, 8});
            output_tensor->set_attrs(out->get_attrs());
            op->replace_output_tensor(std::move(output_tensor));
          })
          .set_annotation(
              "compare a tensor with a reference result for regression testing")
      //
  };
};
static void add_op_defs() {
  auto safe_add_op = [](const xir::OpDef& def) {
    auto factory = xir::op_def_factory();
    for (auto& t1 : factory->get_registered_ops()) {
      if (def.name() == t1) {
        return;
      }
    }
    const_cast<xir::OpDefFactoryImp*>(factory)->register_h(def);
  };
  for (auto& def : my_op_defs()) {
    safe_add_op(def);
  }
}

namespace vitis {
namespace ai {
void maybe_add_op_defs() {
  static bool init = false;
  if (!init) {
    add_op_defs();
  }
}
}  // namespace ai
}  // namespace vitis
