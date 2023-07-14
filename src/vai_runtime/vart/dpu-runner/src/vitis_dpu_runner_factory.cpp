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
#include "vart/dpu/vitis_dpu_runner_factory.hpp"

#include <glog/logging.h>

#include <mutex>
#include <xir/graph/subgraph.hpp>

#include "./dpu_session.hpp"
#include "vart/runner.hpp"
#include "vart/runner_ext.hpp"
namespace vart {
namespace dpu {

class DpuRunnerImp : public RunnerExt {
 public:
 public:
  explicit DpuRunnerImp(const std::string& file_name,
                        const std::string& kernel_name);
  explicit DpuRunnerImp(const xir::Subgraph* subgraph, xir::Attrs* attrs);

  DpuRunnerImp(const DpuRunnerImp&) = delete;
  DpuRunnerImp& operator=(const DpuRunnerImp& other) = delete;

  virtual ~DpuRunnerImp() = default;

 private:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() override;
  virtual std::vector<vart::TensorBuffer*> get_inputs() override;
  virtual std::vector<vart::TensorBuffer*> get_outputs() override;

 protected:
  std::unique_ptr<vart::dpu::DpuSession> dpu_session_;
  std::unique_ptr<vart::Runner> real_runner_;

 private:
  // to follow vitis API definition, use the lock for multithread;
  std::mutex mutex_;
};

std::unique_ptr<vart::Runner> DpuRunnerFactory::create_dpu_runner(
    const std::string& file_name, const std::string& kernel_name) {
  return std::make_unique<DpuRunnerImp>(file_name, kernel_name);
}

std::unique_ptr<vart::Runner> DpuRunnerFactory::create_dpu_runner(
    const xir::Subgraph* subgraph, xir::Attrs* attrs) {
  return std::make_unique<DpuRunnerImp>(subgraph, attrs);
}

DpuRunnerImp::DpuRunnerImp(const std::string& file_name,
                           const std::string& kernel_name)
    : vart::RunnerExt{},
      dpu_session_{vart::dpu::DpuSession::create(file_name, kernel_name)},
      real_runner_{dpu_session_->create_runner()} {}
std::pair<uint32_t, int> DpuRunnerImp::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  //
  std::lock_guard<std::mutex> lock(mutex_);
  return real_runner_->execute_async(input, output);
}

DpuRunnerImp::DpuRunnerImp(const xir::Subgraph* subgraph, xir::Attrs* attrs)
    : vart::RunnerExt{},
      dpu_session_{vart::dpu::DpuSession::create(subgraph, attrs)},
      real_runner_{dpu_session_->create_runner()} {}

int DpuRunnerImp::wait(int jobid, int timeout) {
  return real_runner_->wait(jobid, timeout);
}

std::vector<const xir::Tensor*> DpuRunnerImp::get_input_tensors() {
  return real_runner_->get_input_tensors();
}

std::vector<const xir::Tensor*> DpuRunnerImp::get_output_tensors() {
  return real_runner_->get_output_tensors();
}
std::vector<vart::TensorBuffer*> DpuRunnerImp::get_inputs() {
  return dpu_session_->get_inputs();
}
std::vector<vart::TensorBuffer*> DpuRunnerImp::get_outputs() {
  return dpu_session_->get_outputs();
}
}  // namespace dpu
}  // namespace vart

#include <mutex>
extern "C" vart::Runner* create_runner(const xir::Subgraph* subgraph) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  auto ret = vart::dpu::DpuRunnerFactory::create_dpu_runner(subgraph, nullptr);
  return ret.release();
}
extern "C" vart::Runner* create_runner_with_attrs(const xir::Subgraph* subgraph,
                                                  xir::Attrs* attrs) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  auto ret = vart::dpu::DpuRunnerFactory::create_dpu_runner(subgraph, attrs);
  return ret.release();
}
