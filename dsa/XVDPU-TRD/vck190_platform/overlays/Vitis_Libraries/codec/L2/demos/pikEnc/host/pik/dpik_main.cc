// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/cache_aligned.h"
#include "pik/cmdline.h"
#include "pik/dpik.h"
#include "pik/file_io.h"
#include "pik/os_specific.h"
#include "pik/padded_bytes.h"
#include "pik/profiler.h"

namespace pik {
namespace {

int DecompressMain(int argc, const char* argv[]) {
  DecompressArgs args;
  tools::CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);
  if (!cmdline.Parse(argc, argv) || !args.ValidateArgs()) {
    cmdline.PrintHelp();
    return 1;
  }

  const int bits = TargetBitfield().Bits();
  if ((bits & SIMD_ENABLE) != SIMD_ENABLE) {
    fprintf(stderr, "CPU does not support all enabled targets => exiting.\n");
    return 1;
  }

  PaddedBytes compressed;
  if (!ReadFile(args.file_in, &compressed)) return 1;
  fprintf(stderr, "Read %zu compressed bytes\n", compressed.size());

  CodecContext codec_context;
  ThreadPool pool(args.num_threads);
  DecompressStats stats;

  const std::vector<int> cpus = AvailableCPUs();
  pool.RunOnEachThread([&cpus](const int task, const int thread) {
    // 1.1-1.2x speedup (36 cores) from pinning.
    if (thread < cpus.size()) {
      if (!PinThreadToCPU(cpus[thread])) {
        fprintf(stderr, "WARNING: failed to pin thread %d.\n", thread);
      }
    }
  });

  CodecInOut io(&codec_context);
  size_t downsampling = 1;
  for (size_t i = 0; i < args.num_reps; ++i) {
    if (!Decompress(compressed, args.params, &pool, &io, &downsampling,
                    &stats)) {
      return 1;
    }
  }

  if (!WriteOutput(args, io)) return 1;

  (void)stats.Print(io, downsampling, &pool);

  if (args.print_profile == Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  CacheAligned::PrintStats();
  return 0;
}

}  // namespace
}  // namespace pik

int main(int argc, const char* argv[]) {
  return pik::DecompressMain(argc, argv);
}
