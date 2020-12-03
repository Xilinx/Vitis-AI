#else
    ret.fixpos = -(int8_t)log2f(scale);
#endif
    ret.height = dim_num <= 2 ? 1 : tensor->get_shape().at(2);
    ret.width = dim_num <= 3 ? 1 : tensor->get_shape().at(3);
    ret.channel = dim_num <= 1 ? 1 : tensor->get_shape().at(1);
    ret.dtype = library::DT_FLOAT;
  }
  ret.name = tensor->get_name();
  auto dims = tensor->get_shape();
  auto size = 0ul;
  // CHECK_LT(dims[0], ret.data.size());
  for (auto batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
    auto idx = std::vector<int32_t>(dims.size());
    idx[0] = batch_idx;
    auto data = tb->data(idx);
    ret.get_data(batch_idx) = (void*)data.first;
    size = data.second;
    CHECK_GE(size, ret.height * ret.width * ret.channel);
  }

  return ret;
}

std::vector<vitis::ai::library::InputTensor> DpuTaskImp::getInputTensor(
    size_t idx) {
  auto dpu_runner_ext = dynamic_cast<vart::RunnerExt*>(runners_[idx].get());
  auto inputs = dpu_runner_ext->get_inputs();
  //# Get the current format
  auto fmt = runners_[idx]->get_tensor_format();
  auto scales = vart::get_input_scale(dpu_runner_ext->get_input_tensors());
  auto ret = std::vector<vitis::ai::library::InputTensor>{};
  ret.reserve(inputs.size());
  int c = 0;
  for (auto& t : inputs) {
    ret.emplace_back(convert_tensor_buffer_to_input_tensor(
        t, scales[c], num_of_inputs_, fmt));
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
        << "input tensor[" << c << "]: " << ret.back();
    c++;
  }
  return ret;
}

std::vector<vitis::ai::library::OutputTensor> DpuTaskImp::getOutputTensor(
    size_t idx) {
  auto dpu_runner_ext = dynamic_cast<vart::RunnerExt*>(runners_[idx].get());
  auto outputs = dpu_runner_ext->get_outputs();
  //# Get the current format
  auto fmt = runners_[idx]->get_tensor_format();
  auto scales = vart::get_output_scale(dpu_runner_ext->get_output_tensors());

  auto ret = std::vector<vitis::ai::library::OutputTensor>{};
  ret.reserve(outputs.size());
  int c = 0;
  for (auto& t : outputs) {
    ret.emplace_back(convert_tensor_buffer_to_output_tensor(
        t, scales[c], num_of_inputs_, fmt));
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
        << "output tensor[" << c << "]: " << ret.back();
    c++;
  }
  return ret;
}

size_t DpuTaskImp::get_input_batch(size_t kernel_idx, size_t node_idx) const {
  return dynamic_cast<vart::RunnerExt*>(runners_[kernel_idx].get())
      ->get_inputs()[node_idx]
      ->get_tensor()
      ->get_shape()
      .at(0);
}

size_t DpuTaskImp::get_num_of_kernels() const {  //
  return runners_.size();
}

const xir::Graph* DpuTaskImp::get_graph() const {
  return graph_holder_->get_graph();
}

std::unique_ptr<xir::Attrs> DpuTaskImp::get_attrs() const {
  return graph_holder_->get_graph()->get_attrs();
}

void DpuTaskImp::set_num_of_inputs(size_t n) {
  // TODO it is too much to call clear_num_of_inputs
  // CHECK_EQ(num_of_inputs_, -1)
  //     << "LOGICAL ERROR. you cannot set num input twices";
  CHECK_LT(n, 100) << "with current DPU design, it is not possible for very "
                      "large batch size.";
  num_of_inputs_ = (int)n;
}

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: undecided-unix
// End:
