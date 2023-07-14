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

#include "./async_runner.hpp"

#include <UniLog/UniLog.hpp>
#include <future>
#include <thread>
#include <mutex>
#include <numeric>

#include "../../runner/src/runner_helper.hpp"
#include "./batch_tensor_buffer.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"
#include "xir/graph/graph.hpp"

DEF_ENV_PARAM_2(XLNX_NUM_OF_RUNNER_THREADS, "12", size_t);
DEF_ENV_PARAM(XLNX_MAX_WAITING_TIME_IN_MS, "5");
DEF_ENV_PARAM(DEBUG_ASYNC_RUNNER, "0");
DEF_ENV_PARAM(XLNX_ASYNC_RUNNER_PERF, "0");

namespace {

class AsyncRunner : public vart::Runner {
 public:
  explicit AsyncRunner(std::shared_ptr<vart::Runner> r) : real_runner_{r} {
    LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
        << "AsyncRunner@" << (void*)this << " created.";
  };
  AsyncRunner(const AsyncRunner& other) = delete;

 private:
  virtual ~AsyncRunner() {
    real_runner_.reset();
    LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
        << "AsyncRunner@" << (void*)this << " destroyed.";
  }
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override {
    return real_runner_->execute_async(input, output);
  }
  virtual int wait(int jobid, int timeout) override {
    return real_runner_->wait(jobid, timeout);
  }
  virtual std::vector<const xir::Tensor*> get_input_tensors() override {
    return real_runner_->get_input_tensors();
  }
  virtual std::vector<const xir::Tensor*> get_output_tensors() override {
    return real_runner_->get_output_tensors();
  }

 private:
  std::shared_ptr<vart::Runner> real_runner_;
};

class AsyncRunnerImpl : public vart::Runner {
 public:
  explicit AsyncRunnerImpl(vart::init_function_t f,
                           const xir::Subgraph* subgraph, xir::Attrs* attrs);
  AsyncRunnerImpl(const AsyncRunnerImpl& other) = delete;
  virtual ~AsyncRunnerImpl();

 private:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() override;

 public:
  struct runner_t {
    std::atomic<int> state;
    std::unique_ptr<vart::Runner> runner;
    std::unique_ptr<xir::Attrs> attrs;
    size_t batch_size;
    size_t runner_idx;
  };
  struct queue_element_type_t {
    std::vector<vart::TensorBuffer*> input;
    std::vector<vart::TensorBuffer*> output;
    int job_id;
  };
  struct job_slot_t {
    std::promise<int> promise;
    int job_id;
  };

 private:
  void thread_main();
  void start_one_runner(
      runner_t& runner,
      std::vector<std::unique_ptr<queue_element_type_t>> args);
  int allocate_job_id();
  job_slot_t* find_job_slot(int job_id);
  void delete_job_slot(int job_id);
  size_t num_of_running_runners();
  std::string runners_state_as_string();
  void notify_completion(
      const std::vector<std::unique_ptr<queue_element_type_t>>& args, int ret);

 private:
  static constexpr int IDLE = 0;
  static constexpr int COLLECTING = 1;
  static constexpr int WAITING = 2;
  static constexpr int RUNNING = 3;
  std::vector<runner_t> runners_;
  std::vector<std::unique_ptr<xir::Tensor>> inputs_;
  std::vector<std::unique_ptr<xir::Tensor>> outputs_;
  std::unique_ptr<vitis::ai::ErlMsgBox<queue_element_type_t>> queue_;
  std::unique_ptr<vitis::ai::ErlMsgBox<size_t>> runners_idx_q_;
  std::shared_ptr<vitis::ai::ThreadPool> the_pool_;
  std::thread my_thread_;
  volatile bool running_;
  std::map<int, std::unique_ptr<job_slot_t>> slots_;
  std::mutex mtx_for_slots_;
};
}  // namespace
namespace {
static std::vector<std::unique_ptr<xir::Tensor>>
clone_and_change_dims_for_tensors(const std::vector<const xir::Tensor*>& from) {
  auto ret = std::vector<std::unique_ptr<xir::Tensor>>();
  ret.reserve(from.size());
  for (auto& b : from) {
    auto dims = b->get_shape();
    dims[0] = 1;
    ret.emplace_back(
        xir::Tensor::create(b->get_name(), dims, b->get_data_type()));
  }
  return ret;
}
static size_t get_batch_size(vart::Runner* runner) {
  int batch_size = 0u;
  size_t num_of_dims = 0u;
  for (auto& tensors :
       {runner->get_input_tensors(), runner->get_output_tensors()}) {
    for (auto& b : tensors) {
      auto dims = b->get_shape();
      if (batch_size == 0u) {
        batch_size = dims[0];
        num_of_dims = dims.size();
      } else {
        UNI_LOG_CHECK(dims[0] == batch_size,
                      VART_RUNNER_CONSTRUCTION_FAIL)
            << " all tensor must have the same batch size: "
            << " batch_size=" << batch_size    //
            << " num_of_dims=" << num_of_dims  //
            << " tensor=" << b->get_name()     //
            << " dims[0]=" << dims[0];
      }
    }
  }
  return (size_t)batch_size;
}
AsyncRunnerImpl::AsyncRunnerImpl(vart::Runner* (*init_fun)(const xir::Subgraph*,
                                                           xir::Attrs*),
                                 const xir::Subgraph* subgraph,
                                 xir::Attrs* attrs)
    : inputs_{}, outputs_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
      << "@" << (void*)this << " creating AsyncRunnerImpl for subgraph@"
      << (void*)subgraph << " " << subgraph->get_name();
  size_t num_of_dpu_runners =
      attrs->has_attr("num_of_dpu_runners")
          ? attrs->get_attr<size_t>("num_of_dpu_runners")
          : 10u;
  runners_ = std::vector<AsyncRunnerImpl::runner_t>(num_of_dpu_runners);
  for (auto i = 0u; i < runners_.size(); ++i) {
    // TODO: remove black list
    runners_[i].attrs = xir::Attrs::clone(attrs);
    // it is important not to share attrs for creating real runners,
    // otherwise, they might always use the same device cu index.
    runners_[i].runner = std::unique_ptr<vart::Runner>(
        init_fun(subgraph, runners_[i].attrs.get()));
    runners_[i].state = IDLE;
    runners_[i].batch_size = get_batch_size(runners_[i].runner.get());
    runners_[i].runner_idx = i;
  }
  UNI_LOG_CHECK(!runners_.empty(), VART_RUNNER_CONSTRUCTION_FAIL)
      << " please check attr \"num_of_dpu_runners\"";
  inputs_ = clone_and_change_dims_for_tensors(
      runners_[0].runner->get_input_tensors());
  outputs_ = clone_and_change_dims_for_tensors(
      runners_[0].runner->get_output_tensors());
  queue_ = std::make_unique<vitis::ai::ErlMsgBox<queue_element_type_t>>(
      std::accumulate(runners_.begin(), runners_.end(), 0,
                      [](int s, runner_t& r) { return s + r.batch_size; }));
  runners_idx_q_ =
      std::make_unique<vitis::ai::ErlMsgBox<size_t>>(runners_.size());
  for (auto i = 0u; i < runners_.size(); ++i) {
    runners_idx_q_->emplace_send(i);
  }
  the_pool_ = vitis::ai::WeakStore<std::string, vitis::ai::ThreadPool>::create(
      std::string("async_runner"), ENV_PARAM(XLNX_NUM_OF_RUNNER_THREADS));
  running_ = true;
  // Q: why there is a thread for an async runner?
  //
  // A: it is a tradeoff between the latency and throughput. With a
  // thread, a real runner is able to proactively collect input into a
  // batch within a short period. If it is timeout, e.g. 2ms, the
  // runner start to kick off with avaiable input requests. In this
  // way, we trade throughput for latency.
  //
  // however, under high pressure, the runner can collect enough
  // number of input requests within a very short window, in this way,
  // we trade latency for throughput.
  my_thread_ = std::thread([this]() { thread_main(); });
}

AsyncRunnerImpl::~AsyncRunnerImpl() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
      << "AsyncRunnerImpl@" << (void*)this << " destroying AsyncRunnerImpl";
  running_ = false;
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
      << "async runner is shutting down.";
  my_thread_.join();
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
      << "async runner main thread is terminated.";
  size_t n = 0;
  while ((n = num_of_running_runners()) > 0u) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
        << "busy waiting n= " << n << "states = " << runners_state_as_string();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
      << "size of slots = " << slots_.size()
      << " states: " << runners_state_as_string() << " qlen=" << queue_->size()
      << " qcap=" << queue_->capacity()
      << " if #slots is not zero, there might be some resource leak";
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
      << "AsyncRunnerImpl@" << (void*)this << "  says BYEBYE.";
  the_pool_ = nullptr;  // release the thread pool.
}

std::pair<uint32_t, int> AsyncRunnerImpl::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  auto job_id = allocate_job_id();
  if (!running_) {
    LOG(WARNING) << "runner is shutting down, reject new request";
    std::make_pair(0xFFFFFFFF, -1);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER) >= 2)
      << "job id " << job_id << " is allocated for inputs=" << to_string(input)
      << ",outputs=" << to_string(output);
  queue_->emplace_send(queue_element_type_t{input, output, job_id});
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER) >= 2)
      << "job id " << job_id << " is submitted. qlen=" << queue_->size()
      << " qcap=" << queue_->capacity();
  return std::make_pair((uint32_t)job_id, 0);
}

AsyncRunnerImpl::job_slot_t* AsyncRunnerImpl::find_job_slot(int jobid) {
  std::lock_guard<std::mutex> lock(mtx_for_slots_);
  auto job = slots_.find(jobid);
  if (job == slots_.end()) {
    LOG_IF(WARNING, ENV_PARAM(DEBUG_ASYNC_RUNNER))
        << "job is not found. job_id=" << jobid;
    return nullptr;  // JOB NOT FOUND
  }
  return job->second.get();
}

void AsyncRunnerImpl::delete_job_slot(int jobid) {
  std::lock_guard<std::mutex> lock(mtx_for_slots_);
  slots_.erase(jobid);
}
std::string AsyncRunnerImpl::runners_state_as_string() {
  const char* name[] = {"IDLE", "COLLECTING", "WAITING", "RUNNING"};
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto& r : runners_) {
    if (c++ != 0) {
      str << ",";
    };
    str << name[r.state] << "(" << r.state << ")";
  }
  str << "]";
  return str.str();
}

size_t AsyncRunnerImpl::num_of_running_runners() {
  size_t ret = 0;
  for (auto& r : runners_) {
    if (r.state != IDLE) {
      ret = ret + 1;
    }
  }
  return ret;
}
int AsyncRunnerImpl::wait(int jobid, int timeout) {
  auto job = find_job_slot(jobid);
  auto ret = -1;
  if (job == nullptr) {
    return ret;
  }
  try {
    auto future = job->promise.get_future();
    std::future_status cv = std::future_status::ready;
    if (timeout == -1) {
      future.wait();
    } else {
      cv = future.wait_for(std::chrono::milliseconds(timeout));
    }
    if (cv == std::future_status::ready) {
      ret = future.get();
    }
  } catch (std::exception& e) {
    LOG(WARNING) << "exceptions job_id=" << jobid << " what=" << e.what();
  }
  if (job) {
    delete_job_slot(jobid);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER) >= 3)
      << "wait for job_id=" << jobid << " return ret=" << ret;

  return ret;
}

std::vector<const xir::Tensor*> AsyncRunnerImpl::get_input_tensors() {
  return vitis::ai::vector_unique_ptr_get_const(inputs_);
}
std::vector<const xir::Tensor*> AsyncRunnerImpl::get_output_tensors() {
  return vitis::ai::vector_unique_ptr_get_const(outputs_);
}

int AsyncRunnerImpl::allocate_job_id() {
  std::lock_guard<std::mutex> lock(mtx_for_slots_);
  auto job_id = slots_.empty() ? 0 : slots_.rbegin()->first + 1;
  slots_[job_id] = std::make_unique<job_slot_t>();
  return job_id;
}
static constexpr int INPUT = 0;
static constexpr int OUTPUT = 1;
static std::vector<std::unique_ptr<vart::TensorBuffer>>
create_input_tensor_buffers_from_arg(
    std::vector<std::unique_ptr<AsyncRunnerImpl::queue_element_type_t>>& args,
    int input_or_output) {
  auto batch_size = args.size();
  CHECK_GT(batch_size, 0u);
  auto num_of_tensor_buffers = (input_or_output == INPUT)
                                   ? args[0]->input.size()
                                   : args[0]->output.size();
  auto ret =
      std::vector<std::unique_ptr<vart::TensorBuffer>>(num_of_tensor_buffers);
  for (auto tensor_buffer_idx = 0u; tensor_buffer_idx < num_of_tensor_buffers;
       ++tensor_buffer_idx) {
    auto tensor_buffer_x = std::vector<vart::TensorBuffer*>(batch_size);
    for (auto batch_idx = 0u; batch_idx < batch_size; ++batch_idx) {
      auto n = (input_or_output == INPUT) ? args[batch_idx]->input.size()
                                          : args[0]->output.size();
      CHECK_EQ(n, num_of_tensor_buffers)
          << "all args must have same number of tensor_buffers. "
          << "batch_idx " << batch_idx << " "  //
          ;
      tensor_buffer_x[batch_idx] =
          (input_or_output == INPUT)
              ? args[batch_idx]->input[tensor_buffer_idx]
              : args[batch_idx]->output[tensor_buffer_idx];
    }
    ret[tensor_buffer_idx] =
        std::make_unique<vart::BatchTensorBuffer>(tensor_buffer_x);
  }
  return ret;
}

static int start_one_runner_real(
    vart::Runner* runner,
    std::vector<std::unique_ptr<AsyncRunnerImpl::queue_element_type_t>>& args) {
  auto inputs = create_input_tensor_buffers_from_arg(args, INPUT);
  auto outputs = create_input_tensor_buffers_from_arg(args, OUTPUT);
  auto job = runner->execute_async(vitis::ai::vector_unique_ptr_get(inputs),
                                   vitis::ai::vector_unique_ptr_get(outputs));
  CHECK_EQ(job.second, 0);
  return runner->wait((int)job.first, -1);
}

static std::string jobs_to_string(
    const std::vector<std::unique_ptr<AsyncRunnerImpl::queue_element_type_t>>&
        args) {
  std::ostringstream str;
  str << "[";
  int c = 0;
  for (auto& arg : args) {
    if (c++ != 0) {
      str << ",";
    };
    str << arg->job_id;
  }
  str << "]";
  return str.str();
}

void AsyncRunnerImpl::notify_completion(
    const std::vector<std::unique_ptr<AsyncRunnerImpl::queue_element_type_t>>&
        args,
    int ret) {
  for (auto& arg : args) {
    {
      std::lock_guard<std::mutex> lock(mtx_for_slots_);
      slots_[arg->job_id]->promise.set_value(ret);
    }
  }
}

void AsyncRunnerImpl::start_one_runner(
    AsyncRunnerImpl::runner_t& runner,
    std::vector<std::unique_ptr<queue_element_type_t>> args) {
  LOG_IF(INFO, ENV_PARAM(XLNX_ASYNC_RUNNER_PERF))
      << "batch_perf batch=" << runner.batch_size
      << " requests=" << args.size();
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER) >= 3)
      << " jobs " << jobs_to_string(args) << " are ready for run.";

  runner.state = WAITING;
  the_pool_->async([this, &runner, args = std::move(args)]() mutable {
    LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER) >= 3)
        << " jobs " << jobs_to_string(args) << " are started.";
    runner.state = RUNNING;
    auto ret = start_one_runner_real(runner.runner.get(), args);
    LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER) >= 3)
        << " jobs " << jobs_to_string(args) << " are completed.";
    notify_completion(args, ret);
    runner.state = IDLE;
    runners_idx_q_->emplace_send(runner.runner_idx);
  });
  return;
}

void AsyncRunnerImpl::thread_main() {
  std::unique_ptr<size_t> cur_runner;
  do {
    // round robin
    cur_runner = runners_idx_q_->recv(std::chrono::milliseconds(100));
    if (cur_runner == nullptr) {
      LOG_IF(WARNING, ENV_PARAM(DEBUG_ASYNC_RUNNER))
          << " cannot find an idle runner withing 100ms: states="
          << runners_state_as_string() << " try again."
          << "please check env XLNX_NUM_OF_RUNNER_THREADS and attr "
             "num_of_dpu_runners."
          << ": XLNX_NUM_OF_RUNNER_THREADS="
          << ENV_PARAM(XLNX_NUM_OF_RUNNER_THREADS)
          << " num_of_dpu_runners=" << runners_.size();
      continue;
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER) >= 3)
        << "weakup runner[" << *cur_runner << "]";
    auto& runner = runners_[*cur_runner];
    CHECK_EQ(runner.state, IDLE)
        << " *cur_runner=" << *cur_runner
        << " something wrong. please check queue_.capacity is as same "
           "as sizeof runners. states:"
        << runners_state_as_string();
    auto batch_size = runner.batch_size;
    size_t batch_idx = 0u;
    std::vector<std::unique_ptr<queue_element_type_t>> args;
    args.resize(batch_size);
    do {
      runner.state = COLLECTING;
      for (batch_idx = 0u; batch_idx < batch_size; ++batch_idx) {
        auto arg = queue_->recv(
            std::chrono::milliseconds(ENV_PARAM(XLNX_MAX_WAITING_TIME_IN_MS)));
        if (arg) {
          args[batch_idx] = std::move(arg);
        } else {
          break;
        }
      }
      if (ENV_PARAM(DEBUG_ASYNC_RUNNER)) {
        if (batch_idx != 0) {
          LOG_IF(INFO, batch_idx != batch_size)
              << " throughput might be degraded. "      //
              << " we need increase the queue length."  //
              << " batch_size " << batch_size << " "    //
              << " batch_idx = " << batch_idx << " *cur_runner= " << *cur_runner
              << " running_= " << running_
              << " queue_.capacity = " << queue_->capacity()
              << " queue_.size() = " << queue_->size();
        }
      }
    } while (batch_idx == 0u &&
             // thread is alive if it is running or there are still some
             // quests in the queue.
             (running_ || queue_->size() != 0));
    if (batch_idx > 0u) {
      args.resize(batch_idx);
      start_one_runner(runner, std::move(args));
    } else {
      runner.state = IDLE;
      runners_idx_q_->send_ptr(std::move(cur_runner));
    }
  } while ((running_ || queue_->size() != 0));
  LOG_IF(INFO, ENV_PARAM(DEBUG_ASYNC_RUNNER))
      << "async runner main_thread say good bye";
  return;
}
}  // namespace

// main entry
extern "C" vart::Runner* create_runner_with_attrs(vart::init_function_t f,
                                                  const xir::Subgraph* subgraph,
                                                  xir::Attrs* attrs) {
  auto r = vitis::ai::WeakStore<const xir::Subgraph*, AsyncRunnerImpl>::create(
      subgraph, f, subgraph, attrs);
  return std::unique_ptr<vart::Runner>(new AsyncRunner(r)).release();
}
