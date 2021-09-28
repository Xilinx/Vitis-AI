// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_DPIK_H_
#define PIK_DPIK_H_

#include <stddef.h>
#include <stdint.h>

#include "pik/cmdline.h"
#include "pik/codec.h"
#include "pik/data_parallel.h"
#include "pik/padded_bytes.h"
#include "pik/pik_params.h"
#include "pik/status.h"

namespace pik {

struct DecompressArgs {
  // Initialize non-static default options.
  DecompressArgs();

  // Add all the command line options to the CommandLineParser. Note that the
  // options are tied to the instance that this was called on.
  void AddCommandLineOptions(tools::CommandLineParser* cmdline);

  // Validate the passed arguments, checking whether all passed options are
  // compatible. Returns whether the validation was successful.
  Status ValidateArgs();

  // The parameters.
  const char* file_in = nullptr;
  const char* file_out = nullptr;
  size_t bits_per_sample = 0;
  size_t num_threads;
  std::string color_space;  // description
  DecompressParams params;
  size_t num_reps = 1;
  Override print_profile = Override::kDefault;
};

class DecompressStats {
 public:
  void NotifyElapsed(double elapsed_seconds);

  Status Print(const CodecInOut& io, size_t downsampling, ThreadPool* pool);

 private:
  struct ElapsedStats {
    double central_tendency;
    double min;
    double max;
    double variability;
    const char* type;
  };

  Status SummarizeElapsed(ElapsedStats* s);

  std::vector<double> elapsed_;
};

Status Decompress(const PaddedBytes& compressed, const DecompressParams& params,
                  ThreadPool* pool, CodecInOut* PIK_RESTRICT io,
                  size_t* downsampling, DecompressStats* PIK_RESTRICT stats);

Status WriteOutput(const DecompressArgs& args, const CodecInOut& io);

}  // namespace pik

#endif  // PIK_DPIK_H_
