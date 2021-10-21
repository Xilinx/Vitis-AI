// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/os_specific.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>
#include <sstream>
#include <iterator>

#include "pik/arch_specific.h"
#include "pik/compiler_specific.h"

#if defined(_WIN32) || defined(_WIN64)
#define OS_WIN 1
#define NOMINMAX
#include <windows.h>
#else
#define OS_WIN 0
#endif

#ifdef __linux__
#define OS_LINUX 1
#include <sched.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#define OS_LINUX 0
#endif

#ifdef __MACH__
#define OS_MAC 1
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#define OS_MAC 0
#endif

#ifdef __FreeBSD__
#define OS_FREEBSD 1
#include <sys/cpuset.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#define OS_FREEBSD 0
#endif

namespace pik {

double Now() {
#if OS_WIN
  LARGE_INTEGER counter;
  (void)QueryPerformanceCounter(&counter);
  LARGE_INTEGER freq;
  (void)QueryPerformanceFrequency(&freq);
  return double(counter.QuadPart) / freq.QuadPart;
#elif OS_MAC
  const auto t = mach_absolute_time();
  // On OSX/iOS platform the elapsed time is cpu time unit
  // We have to query the time base information to convert it back
  // See https://developer.apple.com/library/mac/qa/qa1398/_index.html
  static mach_timebase_info_data_t timebase;
  if (timebase.denom == 0) {
    (void)mach_timebase_info(&timebase);
  }
  return double(t) * timebase.numer / timebase.denom * 1E-9;
#else
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1E-9;
#endif
}

struct ThreadAffinity {
#if OS_WIN
  DWORD_PTR mask;
#elif OS_LINUX
  cpu_set_t set;
#elif OS_FREEBSD
  cpuset_t set;
#endif
};

ThreadAffinity* GetThreadAffinity() {
  ThreadAffinity* affinity =
      static_cast<ThreadAffinity*>(malloc(sizeof(ThreadAffinity)));
#if OS_WIN
  DWORD_PTR system_affinity;
  const BOOL ok = GetProcessAffinityMask(GetCurrentProcess(), &affinity->mask,
                                         &system_affinity);
  PIK_CHECK(ok);
#elif OS_LINUX
  const pid_t pid = 0;  // current thread
  const int err = sched_getaffinity(pid, sizeof(cpu_set_t), &affinity->set);
  PIK_CHECK(err == 0);
#elif OS_FREEBSD
  const pid_t pid = getpid();  // current thread
  const int err = cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, pid,
                                     sizeof(cpuset_t), &affinity->set);
  PIK_CHECK(err == 0);

#endif
  return affinity;
}

namespace {

ThreadAffinity* OriginalThreadAffinity() {
  static ThreadAffinity* original = GetThreadAffinity();
  return original;
}

}  // namespace

Status SetThreadAffinity(ThreadAffinity* affinity) {
  // Ensure original is initialized before changing.
  const ThreadAffinity* const original = OriginalThreadAffinity();
  PIK_CHECK(original != nullptr);

#if OS_WIN
  const HANDLE hThread = GetCurrentThread();
  const DWORD_PTR prev = SetThreadAffinityMask(hThread, affinity->mask);
  if (prev == 0) return PIK_FAILURE("SetThreadAffinityMask failed");
#elif OS_LINUX
  const pid_t pid = 0;  // current thread
  const int err = sched_setaffinity(pid, sizeof(cpu_set_t), &affinity->set);
  if (err != 0) return PIK_FAILURE("sched_setaffinity failed");
#elif OS_FREEBSD
  const pid_t pid = getpid();  // current thread
  const int err = cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, pid,
                                     sizeof(cpuset_t), &affinity->set);
  if (err != 0) return PIK_FAILURE("cpuset_setaffinity failed");
#else
  printf("Don't know how to SetThreadAffinity on this platform.\n");
  return false;
#endif
  return true;
}

std::vector<int> AvailableCPUs() {
  std::vector<int> cpus;
  cpus.reserve(64);
#if OS_WIN
  const ThreadAffinity* const affinity = OriginalThreadAffinity();
  for (int cpu = 0; cpu < 64; ++cpu) {
    if (affinity->mask & (1ULL << cpu)) {
      cpus.push_back(cpu);
    }
  }
#elif OS_LINUX
  const ThreadAffinity* const affinity = OriginalThreadAffinity();
  for (size_t cpu = 0; cpu < sizeof(cpu_set_t) * 8; ++cpu) {
    if (CPU_ISSET(cpu, &affinity->set)) {
      cpus.push_back(cpu);
    }
  }
#elif OS_FREEBSD
  const ThreadAffinity* const affinity = OriginalThreadAffinity();
  for (size_t cpu = 0; cpu < sizeof(cpuset_t) * 8; ++cpu) {
    if (CPU_ISSET(cpu, &affinity->set)) {
      cpus.push_back(cpu);
    }
  }
#else
  cpus.push_back(0);
#endif
  return cpus;
}

Status PinThreadToCPU(const int cpu) {
  ThreadAffinity affinity;
#if OS_WIN
  affinity.mask = 1ULL << cpu;
#elif OS_LINUX
  CPU_ZERO(&affinity.set);
  CPU_SET(cpu, &affinity.set);
#elif OS_FREEBSD
  CPU_ZERO(&affinity.set);
  CPU_SET(cpu, &affinity.set);
#endif
  return SetThreadAffinity(&affinity);
}

Status PinThreadToRandomCPU() {
  std::vector<int> cpus = AvailableCPUs();

  // Remove first two CPUs because interrupts are often pinned to them.
  PIK_CHECK(cpus.size() > 2);
  cpus.erase(cpus.begin(), cpus.begin() + 2);

  // Random choice to prevent burning up the same core.
  std::random_device device;
  std::ranlux48 generator(device());
  std::shuffle(cpus.begin(), cpus.end(), generator);
  const int cpu = cpus.front();

  PIK_RETURN_IF_ERROR(PinThreadToCPU(cpu));

  // After setting affinity, we should be running on the desired CPU.
#if PIK_ARCH_X64
  printf("Running on CPU #%d, APIC ID %02x\n", cpu, ApicId());
#else
  printf("Running on CPU #%d\n", cpu);
#endif
  return true;
}

Status RunCommand(const std::vector<std::string>& args) {
#if _POSIX_VERSION >= 200112L
  // Avoid system(), but do not try to be over-zealous about not passing along
  // some special resources further (such as: inherited-not-marked-FD_CLOEXEC
  // file descriptors).
  std::vector<const char*> c_args;
  c_args.reserve(args.size() + 1);
  for (size_t i = 0; i < args.size(); ++i) {
    c_args.push_back(args[i].c_str());
  }
  c_args.push_back(nullptr);
  const pid_t pid = fork();
  if (pid == -1)  // fork() failed.
    return false;
  if (pid != 0) {  // Parent process.
    int ret_status;
    if (pid != waitpid(pid, &ret_status, 0)) {
      return false;  // waitpid() error.
    }
    return ret_status == 0;
  } else {  // Child process.
    execvp(c_args[0],
           // Address benign-but-annoying execvp() signature weirdness.
           const_cast<char * const *>(c_args.data()));
    fprintf(stderr, "execvp() failed. Exiting child process.\n");
    exit(EXIT_FAILURE);
  }
#elif OS_WIN
  // Synthesize a string for system(). And warn about it.
  // TODO(user): Fix this - research the safe way to run a command on Windows.
  // Likely, the solution is along these lines:
  // https://docs.microsoft.com/en-us/windows/desktop/ProcThread/creating-processes
  std::ostringstream cmd;
  std::copy(args.begin(), args.end(),
           std::ostream_iterator<std::string>(cmd, " "));
  printf(stderr, "Warning: Using system() on string: %s\n", cmd.str.c_str());
  int ret = system(cmd.str.c_str());
  if (errno != ENOENT &&  // Windows: Command interpreter not found.
      ret == 0) {
    return true;
  }
  return false;
#else
#error Neither a POSIX-1.2001 nor a Windows System.
#endif
}

}  // namespace pik
