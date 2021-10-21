// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/pik.h"

#include <string>
#include <vector>

#undef PROFILER_ENABLED
#define PROFILER_ENABLED 1
#include "pik/adaptive_quantization.h"
#include "pik/common.h"
#include "pik/compressed_image.h"
#include "pik/headers.h"
#include "pik/image.h"
#include "pik/multipass_handler.h"
#include "pik/noise.h"
#include "pik/os_specific.h"
#include "pik/pik_frame.h"
#include "pik/pik_params.h"
#include "pik/profiler.h"
#include "pik/quantizer.h"
#include "pik/saliency_map.h"
#include "pik/single_image_handler.h"

namespace pik {

namespace {
static const uint8_t kBrunsliMagic[] = {0x0A, 0x04, 'B', 0xd2, 0xd5, 'N', 0x12};

// TODO(user): use VerifySignature, when brunsli codebase is attached.
bool IsBrunsliFile(const PaddedBytes &compressed) {
  const size_t magic_size = sizeof(kBrunsliMagic);
  if (compressed.size() < magic_size) {
    return false;
  }
  if (memcmp(compressed.data(), kBrunsliMagic, magic_size) != 0) {
    return false;
  }
  return true;
}

Status BrunsliToPixels(const DecompressParams &dparams,
                       const PaddedBytes &compressed, CodecInOut *io,
                       PikInfo *aux_out, ThreadPool *pool) {
  return PIK_FAILURE("Brunsli decoding is not implemented yet.");
}

} // namespace

Status PixelsToPik(const CompressParams &cparams, std::string xclbinPath,
                   const CodecInOut *io, PaddedBytes *compressed,
                   PikInfo *aux_out, ThreadPool *pool) {
  if (io->xsize() == 0 || io->ysize() == 0) {
    return PIK_FAILURE("Empty image");
  }
  if (!io->HasOriginalBitsPerSample()) {
    return PIK_FAILURE("Pik requires specifying original bit depth "
                       "of the pixels to encode as metadata.");
  }
  FileHeader container;
  MakeFileHeader(cparams, io, &container);

  size_t extension_bits, total_bits;
  PIK_CHECK(CanEncode(container, &extension_bits, &total_bits));

  compressed->resize(DivCeil(total_bits, kBitsPerByte));
  size_t pos = 0;
  PIK_RETURN_IF_ERROR(
      WriteFileHeader(container, extension_bits, &pos, compressed->data()));
  FrameParams frame_params;
  SingleImageManager transform;
  if (cparams.progressive_mode) {
    // TODO(veluca): re-enable saliency.
    PassDefinition pass_definition[] = {
        {/*num_coefficients=*/2, /*salient_only=*/false,
         /*suitable_for_downsampling_factor_of_at_least=*/4},
        {/*num_coefficients=*/3, /*salient_only=*/false,
         /*suitable_for_downsampling_factor_of_at_least=*/2},
        {/*num_coefficients=*/8, /*salient_only=*/false}};
    transform.SetProgressiveMode(ProgressiveMode{pass_definition});
  }
  PIK_RETURN_IF_ERROR(hls_PixelsToPikPass(cparams, xclbinPath, frame_params, io,
                                          pool, compressed, pos, aux_out,
                                          &transform));
  return true;
}

Status PikToPixels(const DecompressParams &dparams,
                   const PaddedBytes &compressed, CodecInOut *io,
                   PikInfo *aux_out, ThreadPool *pool) {
  PROFILER_ZONE("PikToPixels uninstrumented");

  if (IsBrunsliFile(compressed)) {
    return BrunsliToPixels(dparams, compressed, io, aux_out, pool);
  }

  // To avoid the complexity of file I/O and buffering, we assume the bitstream
  // is loaded (or for large images/sequences: mapped into) memory.
  BitReader reader(compressed.data(), compressed.size());
  FileHeader container;
  PIK_RETURN_IF_ERROR(ReadFileHeader(&reader, &container));

  // Preview is discardable, i.e. content image does not rely on decoded preview
  // pixels; just skip it, if any.
  size_t preview_size_bits = container.preview.size_bits;
  if (preview_size_bits != 0) {
    reader.SkipBits(preview_size_bits);
  }

  SingleImageManager transform;
  PIK_RETURN_IF_ERROR(PikPassToPixels(dparams, compressed, container, pool,
                                      &reader, io, aux_out, &transform));

  if (dparams.check_decompressed_size &&
      reader.Position() != compressed.size()) {
    return PIK_FAILURE("Pik compressed data size mismatch.");
  }

  io->enc_size = compressed.size();

  return true;
}

} // namespace pik
