
/*
 * Copyright 2019 Xilinx Inc.
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

#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <vector>

#include "./erl_msg_box.hpp"
namespace vitis {
namespace ai {
class ThreadPool {
 public:
  static std::unique_ptr<ThreadPool> create(size_t num_of_threads);
#if defined(__cpp_lib_result_of_sfinae)
  template <class Function, class... Args>
  using result_t =
      std::result_of_t<std::decay_t<Function>(std::decay_t<Args>...)>;
#elif defined(__cpp_lib_is_invocable)
  template <class Function, class... Args>
  using result_t =
      std::invoke_result_t<std::decay_t<Function>, std::decay_t<Args>...>;
#else
#error "not supported, c++14 or c++17 is needed."
#endif

  template <class Function, class... Args>
  std::future<result_t<Function, Args...>> async(Function&& f, Args&&... args) {
    std::packaged_task<result_t<Function, Args...>()> task(
        std::bind(std::forward<Function>(f), std::forward<Args>(args)...));
    std::future<result_t<Function, Args...>> ret = task.get_future();
    queue_.emplace_send(std::move(task));
    return ret;
  }
  ~ThreadPool();

 private:
  explicit ThreadPool(size_t num_of_thread);

 private:
  static void thread_main(ThreadPool* self);

 private:
  std::vector<std::thread> pool_;
  vitis::ai::ErlMsgBox<std::packaged_task<void()>> queue_;
  int running_;

 private:
  /// thanks akk
  /// https://stackoverflow.com/questions/20843271/passing-a-non-copyable-closure-object-to-stdfunction-parameter
  template <class F>
  auto make_copyable_function(F&& f) {
    using dF = std::decay_t<F>;
    auto spf = std::make_shared<dF>(std::forward<F>(f));
    return [spf](auto&&... args) -> decltype(auto) {
      return (*spf)(decltype(args)(args)...);
    };
  }
};
}  // namespace ai
}  // namespace vitis
