// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <unistd.h>

#include <cstdio>
#include <string>

#include "pik/saliency_map.h"

#include "pik/bit_reader.h"
#include "pik/headers.h"
#include "pik/os_specific.h"
#include "pik/pik_frame.h"
#include "pik/pik_info.h"
#include "pik/single_image_handler.h"

namespace pik {

const char* const kPartialSuffix = ".partial.png";
const char* const kHeatmapSuffix = ".heatmap.pgm";

namespace {

Status ProduceSaliencyMapWithoutCleanup(const CompressParams& cparams,
                                        const PaddedBytes* compressed,
                                        const CodecInOut* io, ThreadPool* pool,
                                        std::shared_ptr<ImageF>* out_heatmap) {
  DecompressParams dparams;
  BitReader reader(compressed->data(), compressed->size());
  FileHeader container;
  PikInfo aux_out;
  CodecInOut io_partial(io->Context());
  PIK_RETURN_IF_ERROR(ReadFileHeader(&reader, &container));
  SingleImageManager transform;
  // TODO(user): Replace resynthesis below with using a GetDecodedPass()
  // method.
  // Cannot rely on transform.IsLastPass() here, since we process a
  // partially-compressed image.
  const int kNumStepsAvailable = 2;  // DC and Low frequency.
  for (int num_pass = 0; num_pass < kNumStepsAvailable; num_pass++) {
    PIK_RETURN_IF_ERROR(PikPassToPixels(dparams, *compressed, container, pool,
                                        &reader, &io_partial, &aux_out,
                                        &transform));
  }
  const std::string filename_partially_constructed_image =
      std::string(cparams.file_out) + kPartialSuffix;
  const std::string filename_heatmap =
      std::string(cparams.file_out) + kHeatmapSuffix;

  if (!io_partial.EncodeToFile(io->dec_c_original,
                               io->original_bits_per_sample(),
                               filename_partially_constructed_image, pool))
    return false;
  if (!RunCommand({cparams.saliency_extractor_for_progressive_mode,
                   std::to_string(kBlockDim),
                   cparams.file_in,
                   filename_partially_constructed_image,
                   filename_heatmap}))
    return false;

  CodecInOut io_heatmap(io->Context());
  if (!io_heatmap.SetFromFile(filename_heatmap, pool)) {
    fprintf(stderr, "Failed to read heatmap: %s\n", filename_heatmap.c_str());
    return false;
  }
  if (cparams.verbose) {
    printf("Read heatmap: xsize=%zu ysize=%zu is_gray=%d\n",
           io_heatmap.xsize(), io_heatmap.ysize(), io_heatmap.IsGray());
  }
  out_heatmap->reset(new ImageF(io_heatmap.xsize(),
                                io_heatmap.ysize()));
  for (size_t num_row = 0; num_row < io_heatmap.ysize(); num_row++) {
    const auto row_src = io_heatmap.color().PlaneRow(0, num_row);
    const auto row_dst = (*out_heatmap)->Row(num_row);
    for (size_t num_col = 0; num_col < io_heatmap.xsize(); num_col++) {
      row_dst[num_col] = row_src[num_col] / 255.0;
    }
  }
  return true;
}

}  // namespace

Status ProduceSaliencyMap(const CompressParams& cparams,
                          const PaddedBytes* compressed, const CodecInOut* io,
                          ThreadPool* pool,
                          std::shared_ptr<ImageF>* out_heatmap) {
  Status ret = ProduceSaliencyMapWithoutCleanup(cparams, compressed, io, pool,
                                                out_heatmap);
  if (!cparams.keep_tempfiles) {
    // Ignore (benign) failures.
    unlink((std::string(cparams.file_out) + kPartialSuffix).c_str());
    unlink((std::string(cparams.file_out) + kHeatmapSuffix).c_str());
  }
  return ret;
}

}  // namespace pik
