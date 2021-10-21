// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/cpik.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1

#include "pik/cmdline.h"
#include "pik/file_io.h"
#include "pik/os_specific.h"
#include "pik/padded_bytes.h"
#include "pik/profiler.h"

namespace pik {
namespace {

int CompressMain(int argc, const char **argv) {
  CompressArgs args;
  tools::CommandLineParser cmdline;
  PIK_ASSERT(args.AddCommandLineOptions(&cmdline));
  if (!cmdline.Parse(argc, argv) || !args.ValidateArgs(cmdline)) {
    cmdline.PrintHelp();
    return 1;
  }

  const int bits = TargetBitfield().Bits();
  if ((bits & SIMD_ENABLE) != SIMD_ENABLE) {
    fprintf(stderr, "CPU does not support all enabled targets => exiting.\n");
    return 1;
  }

  ThreadPool pool(args.num_threads);

  PaddedBytes compressed;
  if (!Compress(&pool, args.params.xclbinPath, args, &compressed))
    return 1;

  if (args.params.file_out) {
    if (!WriteFile(compressed, args.params.file_out))
      return 1;
  }

  if (args.print_profile == Override::kOn) {
    PROFILER_PRINT_RESULTS();
  }
  return 0;
}

} // namespace
} // namespace pik

int main(int argc, const char **argv) { return pik::CompressMain(argc, argv); }
