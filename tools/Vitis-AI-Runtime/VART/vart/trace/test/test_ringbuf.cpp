#include "event.hpp"
#include "ringbuf.hpp"

using namespace vitis::ai;
using namespace std;

int main(void) {
  RingBuf buf(8);

  for (auto i = 0; i < 1000000; i++) {
    // auto e1 = new traceEvent<uint64_t>(VAI_EVENT_PY_FUNC_END, "XXX", i,
    // string("hels"), 123123123);
    auto e1 = new traceEvent<uint64_t>(VAI_EVENT_PY_FUNC_END, "XXX", i,
                                       "run_time", 123123123);
    buf.push(e1);

    if (i % 100000 == 0) {
      std::cout << "input test " << i << " times" << std::endl;
    }
  }

  buf.dump();

  return 0;
}
