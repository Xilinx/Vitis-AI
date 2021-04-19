/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <thread>
#include "tests.hpp"

std::vector<const xir::Subgraph*> get_dpu_subgraph(
            const xir::Graph* graph) {
    auto root = graph->get_root_subgraph();
    auto children = root->children_topological_sort();
    auto ret = std::vector<const xir::Subgraph*>();
    for (auto c : children) {
        CHECK(c->has_attr("device"));
        auto device = c->get_attr<std::string>("device");
        if (device == "DPU") {
            ret.emplace_back(c);
        }
    }
    return ret;
}

TestClassifyMultiThread::TestClassifyMultiThread(
  std::string runner_dir, unsigned nqueries, unsigned nthreads, unsigned nrunners, std::string img_dir, const bool goldenAvailable, const bool verbose) 
 : num_queries_(nqueries), num_threads_(nthreads), num_runners_(nrunners), 
   runner_dir_(runner_dir), runners_(nrunners)
{
  
  cpuUtilobj_.reset(new cpuUtil(runner_dir, goldenAvailable, verbose, img_dir, num_queries_));
  
  std::cout<<"Executing test for "<<runner_dir<<std::endl;
  for (unsigned i=0; i < num_runners_; i++)
    init_thread(i);

  std::cout<<"********************************"<<std::endl;
  std::cout<<"Loading "<<num_queries_*4<<" Images ..."<<std::endl;

}

void TestClassifyMultiThread::run() {
  std::vector<std::thread> threads(num_threads_);

  auto t1 = std::chrono::high_resolution_clock::now();

  for (unsigned ti=0; ti < threads.size(); ti++)
  {
    auto ri = ti % runners_.size();
    threads[ti] = std::thread([this,ti,ri]{run_thread(ti, ri, num_queries_/num_threads_);});
  }

  for (unsigned ti=0; ti < threads.size(); ti++)
    threads[ti].join();

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = t2-t1;
  std::cout << "Elapsed Threads: " << elapsed.count() << std::endl;
  std::cout<<"NumThreads: "<<num_threads_<<std::endl;
  std::cout<<"NumRunners: "<<runners_.size()<<std::endl;
  cpuUtilobj_->printtop1top5(num_queries_);

}

void TestClassifyMultiThread::init_thread(unsigned ridx)
{
    std::unique_ptr<xir::Graph> graph0 = xir::Graph::deserialize(runner_dir_);
    auto subgraph0 = graph0->get_root_subgraph();
    std::map<std::string, std::string> runset;
    runset.emplace("run","librt-engine.so");
    subgraph0->children_topological_sort()[1]->set_attr("runner", runset);
    graph0->serialize(runner_dir_);
    std::unique_ptr<xir::Graph> graph = xir::Graph::deserialize(runner_dir_);
    auto subgraph = get_dpu_subgraph(graph.get());
    auto r = vart::Runner::create_runner(subgraph[0], "run");
    runners_[ridx]=std::move(r);
}

void TestClassifyMultiThread::run_thread(unsigned tidx, unsigned ridx, unsigned n) {
  auto runner = runners_[ridx].get();
  auto inputs = dynamic_cast<vart::RunnerExt*>(runner)->get_inputs();
  auto outputs = dynamic_cast<vart::RunnerExt*>(runner)->get_outputs();
  
  for (unsigned i=tidx*n; i < (tidx+1)*n; i++)
  {
    cpuUtilobj_->fillInData(i, inputs);
    auto ret = (runner)->execute_async(inputs, outputs);
    (runner)->wait(uint32_t(ret.first), 20000);
    cpuUtilobj_->postProcess(outputs, i);
  }

}


