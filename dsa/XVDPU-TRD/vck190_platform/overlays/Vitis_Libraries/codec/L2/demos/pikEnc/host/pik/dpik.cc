// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/dpik.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "pik/data_parallel.h"

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/arch_specific.h"
#include "pik/args.h"
#include "pik/common.h"
#include "pik/file_io.h"
#include "pik/image.h"
#include "pik/os_specific.h"
#include "pik/padded_bytes.h"
#include "pik/pik.h"
#include "pik/pik_info.h"
#include "pik/profiler.h"
#include "pik/robust_statistics.h"
#include "pik/simd/targets.h"

namespace pik {

DecompressArgs::DecompressArgs() {
  // TODO(janwas): differentiate between cores/HT
  num_threads = AvailableCPUs().size() / 2;
}

void DecompressArgs::AddCommandLineOptions(
    tools::CommandLineParser* cmdline) {
  // Positional arguments.
  cmdline->AddPositionalOption("INPUT", "the compressed input file", &file_in);

  cmdline->AddPositionalOption(
      "OUTPUT", "the output can be PNG with ICC, or PPM/PFM.", &file_out);

  // Flags.
  cmdline->AddOptionValue('\0', "bits_per_sample", "N",
                          "defaults to original (input) bit depth",
                          &bits_per_sample, &ParseUnsigned);

  cmdline->AddOptionValue('\0', "num_threads", "N",
                          "The number of threads to use", &num_threads,
                          &ParseUnsigned);

  cmdline->AddOptionValue('\0', "color_space", "RGB_D65_SRG_Rel_Lin",
                          "defaults to original (input) color space",
                          &color_space, &ParseString);

  cmdline->AddOptionValue('\0', "num_reps", "N", nullptr, &num_reps,
                          &ParseUnsigned);

  cmdline->AddOptionValue('\0', "noise", "0", "disables noise generation",
                          &params.noise, &ParseOverride);

  cmdline->AddOptionValue('\0', "gradient", "0",
                          "disables the extra gradient map", &params.gradient,
                          &ParseOverride);

  cmdline->AddOptionValue('\0', "adaptive_reconstruction", "0|1",
                          "disables/enables extra filtering",
                          &params.adaptive_reconstruction, &ParseOverride);

  cmdline->AddOptionValue('\0', "gaborish", "0..7",
                          "chooses deblocking strength (4=normal).",
                          &params.gaborish, &ParseGaborishStrength);

  cmdline->AddOptionValue('s', "downsampling", "1,2,4,8",
                          "maximum permissible downsampling factor",
                          &params.max_downsampling, &ParseUnsigned);

  cmdline->AddOptionValue('\0', "print_profile", "0|1",
                          "print timing information before exiting",
                          &print_profile, &ParseOverride);
}

Status DecompressArgs::ValidateArgs() {
  if (params.noise == Override::kOn) {
    fprintf(stderr, "Noise can only be enabled by the encoder.\n");
    return PIK_FAILURE("Cannot force noise on");
  }
  if (params.gradient == Override::kOn) {
    fprintf(stderr, "Gradient can only be enabled by the encoder.\n");
    return PIK_FAILURE("Cannot force gradient on");
  }

  if (file_in == nullptr) {
    fprintf(stderr, "Missing INPUT filename.\n");
    return false;
  }
  return true;
}

void DecompressStats::NotifyElapsed(double elapsed_seconds) {
  PIK_ASSERT(elapsed_seconds > 0.0);
  elapsed_.push_back(elapsed_seconds);
}

Status DecompressStats::Print(const CodecInOut& io, size_t downsampling,
                              ThreadPool* pool) {
  ElapsedStats s;
  PIK_RETURN_IF_ERROR(SummarizeElapsed(&s));
  char variability[20] = {'\0'};
  if (s.variability != 0.0) {
    snprintf(variability, sizeof(variability), " (var %.2f)", s.variability);
  }

  const size_t xsize = io.xsize();
  const size_t ysize = io.ysize();
  const size_t channels = io.c_current().Channels() + io.HasAlpha();
  const size_t bytes = downsampling * downsampling * xsize * ysize * channels *
                       DivCeil(io.original_bits_per_sample(), kBitsPerByte);
  const auto mb_per_sec = [bytes](const double elapsed) {
    return bytes * 1E-6 / elapsed;
  };
  const double mbps = mb_per_sec(s.central_tendency);
  // Note flipped order: higher elapsed = lower mbps.
  const double mbps_min = mb_per_sec(s.max);
  const double mbps_max = mb_per_sec(s.min);

  fprintf(stderr,
          "%zu x %zu, %s%.2f MB/s [%.2f, %.2f]%s, %zu reps, %zu threads).\n",
          xsize, ysize, s.type, mbps, mbps_min, mbps_max, variability,
          elapsed_.size(), NumWorkerThreads(pool));
  return true;
}

Status DecompressStats::SummarizeElapsed(ElapsedStats* s) {
  // type depends on #reps.
  if (elapsed_.empty()) return PIK_FAILURE("Didn't call NotifyElapsed");

  s->min = *std::min_element(elapsed_.begin(), elapsed_.end());
  s->max = *std::max_element(elapsed_.begin(), elapsed_.end());

  // Single rep
  if (elapsed_.size() == 1) {
    s->central_tendency = elapsed_[0];
    s->variability = 0.0;
    s->type = "";
    return true;
  }

  // Two: skip first (noisier)
  if (elapsed_.size() == 2) {
    s->central_tendency = elapsed_[1];
    s->variability = 0.0;
    s->type = "second: ";
    return true;
  }

  // Prefer geomean unless numerically unreliable (too many reps)
  if (std::pow(elapsed_[0], elapsed_.size()) < 1E100) {
    double product = 1.0;
    for (size_t i = 1; i < elapsed_.size(); ++i) {
      product *= elapsed_[i];
    }

    s->central_tendency = std::pow(product, 1.0 / (elapsed_.size() - 1));
    s->variability = 0.0;
    s->type = "geomean: ";
    return true;
  }

  // Else: mode
  std::sort(elapsed_.begin(), elapsed_.end());
  s->central_tendency = HalfSampleMode()(elapsed_.data(), elapsed_.size());
  s->variability = MedianAbsoluteDeviation(elapsed_, s->central_tendency);
  s->type = "mode: ";
  return true;
}

// Called num_reps times.
Status Decompress(const PaddedBytes& compressed, const DecompressParams& params,
                  ThreadPool* pool, CodecInOut* PIK_RESTRICT io,
                  size_t* downsampling, DecompressStats* PIK_RESTRICT stats) {
  PikInfo info;
  const double t0 = Now();
  if (!PikToPixels(params, compressed, io, &info, pool)) {
    fprintf(stderr, "Failed to decompress.\n");
    return false;
  }
  *downsampling = info.downsampling;
  const double t1 = Now();
  stats->NotifyElapsed(t1 - t0);
  return true;
}

Status WriteOutput(const DecompressArgs& args, const CodecInOut& io) {
  // Can only write if we decoded and have an output filename.
  // (Writing large PNGs is slow, so allow skipping it for benchmarks.)
  if (args.num_reps == 0 || args.file_out == nullptr) return true;

  // Override original color space with arg if specified.
  ColorEncoding c_out = io.dec_c_original;
  if (!args.color_space.empty()) {
    ProfileParams pp;
    if (!ParseDescription(args.color_space, &pp) ||
        !ColorManagement::SetFromParams(pp, &c_out)) {
      fprintf(stderr, "Failed to apply color_space.\n");
      return false;
    }
  }

  // Override original #bits with arg if specified.
  const size_t bits_per_sample = args.bits_per_sample == 0
                                     ? io.original_bits_per_sample()
                                     : args.bits_per_sample;

  if (!io.EncodeToFile(c_out, bits_per_sample, args.file_out)) {
    fprintf(stderr, "Failed to write decoded image.\n");
    return false;
  }
  fprintf(stderr, "Wrote %zu bytes; done.\n", io.enc_size);
  return true;
}

}  // namespace pik
