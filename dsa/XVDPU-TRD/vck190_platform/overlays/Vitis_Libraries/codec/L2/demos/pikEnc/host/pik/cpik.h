// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_CPIK_H_
#define PIK_CPIK_H_

#include <utility>

#include "pik/cmdline.h"
#include "pik/codec.h"
#include "pik/padded_bytes.h"
#include "pik/pik_inspection.h"
#include "pik/pik_params.h"
#include "pik/status.h"

namespace pik {

struct CompressArgs {
  // Initialize non-static default options.
  CompressArgs();

  void SetInspectorImage3F(InspectorImage3F inspector) {
    inspector_image3f = inspector;
  }

  // Add all the command line options to the CommandLineParser. Note that the
  // options are tied to the instance that this was called on.
  Status AddCommandLineOptions(tools::CommandLineParser *cmdline);

  // Validate the passed arguments, checking whether all passed options are
  // compatible. Returns whether the validation was successful.
  Status ValidateArgs(const tools::CommandLineParser &cmdline);

  DecoderHints dec_hints;
  CompressParams params;
  size_t num_threads = 0;
  bool got_num_threads = false;
  Override print_profile = Override::kDefault;

  // Will get passed on to PikInfo.
  InspectorImage3F inspector_image3f;

  // References (ids) of specific options to check if they were matched.
  tools::CommandLineParser::OptionId opt_distance_id = -1;
  tools::CommandLineParser::OptionId opt_target_size_id = -1;
  tools::CommandLineParser::OptionId opt_target_bpp_id = -1;
};

Status Compress(ThreadPool *pool, std::string xclbinPath, CompressArgs &args,
                PaddedBytes *compressed);

} // namespace pik

#endif // PIK_CPIK_H_
