
#include <vitis/ai/benchmark.hpp>
#include <vitis/ai/brtseg.hpp>
int main(int argc, char *argv[]) {
  std::string model = argv[1];
  return vitis::ai::main_for_performance(argc, argv, [model] {
    { return vitis::ai::Brtseg::create(model); }
  });
}

