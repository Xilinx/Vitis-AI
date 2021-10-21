// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_OS_SPECIFIC_H_
#define PIK_OS_SPECIFIC_H_

// OS-specific functions (e.g. timing and thread affinity)

#include <string>
#include <vector>

#include "pik/status.h"

namespace pik {

// Returns current time [seconds] from a monotonic clock with unspecified
// starting point - only suitable for computing elapsed time.
double Now();

// Returns CPU numbers in [0, N), where N is the number of bits in the
// thread's initial affinity (unaffected by any SetThreadAffinity).
std::vector<int> AvailableCPUs();

// Opaque.
struct ThreadAffinity;

// Caller must free() the return value.
ThreadAffinity* GetThreadAffinity();

// Restores a previous affinity returned by GetThreadAffinity.
Status SetThreadAffinity(ThreadAffinity* affinity);

// Ensures the thread is running on the specified cpu, and no others.
// Useful for reducing nanobenchmark variability (fewer context switches).
// Uses SetThreadAffinity.
Status PinThreadToCPU(const int cpu);

// Random choice of CPU avoids overloading any one core.
// Uses SetThreadAffinity.
Status PinThreadToRandomCPU();

// Executes a command in a subprocess.
Status RunCommand(const std::vector<std::string>& args);

}  // namespace pik

#endif  // PIK_OS_SPECIFIC_H_
