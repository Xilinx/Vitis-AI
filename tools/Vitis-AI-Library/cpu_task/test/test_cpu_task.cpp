#include <iostream>
using namespace std;
#include <chrono>
#include <thread>
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>

#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"

int main(int argc, char* argv[]) {
  LOG(INFO) << "HELLO";
  auto graph = xir::Graph::deserialize(argv[1]);
  auto root = graph->get_root_subgraph();
  xir::Subgraph* s = nullptr;
  for (auto c : root->get_children()) {
    if (c->get_attr<std::string>("device") == "CPU") {
      s = c;
      break;
    }
  }
  auto attrs = xir::Attrs::create();
  attrs->set_attr("lib", std::map<std::string, std::string>{
                             {"CPU", "libvitis_ai_library-cpu_task.so"}});
  auto runner = vart::Runner::create_runner_with_attrs(s, attrs.get());
  auto inputs =
      vart::alloc_cpu_flat_tensor_buffers(runner->get_input_tensors());
  auto outputs =
      vart::alloc_cpu_flat_tensor_buffers(runner->get_output_tensors());
  auto job = runner->execute_async(vitis::ai::vector_unique_ptr_get(inputs),
                                   vitis::ai::vector_unique_ptr_get(outputs));
  runner->wait((int)job.first, -1);
  LOG(INFO) << "BYEBYE";
  return 0;
}
