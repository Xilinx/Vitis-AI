// This example requires an empty, local redis-server instance running with
// default settings.
#include <glog/logging.h>
#include <unistd.h>

#include <iostream>
#include <mutex>
#include <sstream>
#include <vitis/ai/lock.hpp>

using namespace std;
int main(int argc, char* argv[]) {
  auto device_name = string(argv[1]);
  {
    cout << "trying to lock..." << endl;
    auto mtx = vitis::ai::Lock::create(device_name);
    auto lock =
        std::unique_lock<vitis::ai::Lock>(*(mtx.get()), std::try_to_lock_t());
    if (!lock.owns_lock()) {
      cout << "waiting for other process to release the resource:"
           << device_name << endl;
      lock.lock();
    }
    cout << device_name << " is lock. presss any key to release the lock..."
         << endl;
    char c;
    cin >> c;
    cout << "lock is released" << endl;
  }
  return 0;
}
