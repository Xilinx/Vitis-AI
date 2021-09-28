// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_NOISE_H_
#define PIK_NOISE_H_

// Noise synthesis. Currently disabled.

#include "pik/bit_reader.h"
#include "pik/image.h"

namespace pik {

struct NoiseParams {
  // Parameters of the fitted noise curve.
  // alpha * x ^ gamma + beta,
  // where x is an intensity of pixel / mean intensity of patch
  float alpha = 0.0f;
  float gamma = 0.0f;
  float beta = 0.0f;
};

struct NoiseLevel {
  float noise_level;
  float intensity;
};

// Add a noise to Opsin image
void AddNoise(const NoiseParams& noise_params, Image3F* opsin);

// Get parameters of the noise for NoiseParams model
void GetNoiseParameter(const Image3F& opsin, NoiseParams* noise_params,
                       float quality_coef);

std::string EncodeNoise(const NoiseParams& noise_params);

bool DecodeNoise(BitReader* br, NoiseParams* noise_params);

// Texture Strength is defined as tr(A), A = [Gh, Gv]^T[[Gh, Gv]]
std::vector<float> GetTextureStrength(const Image3F& opsin, const int block_s);

float GetThresholdFlatIndices(const std::vector<float>& texture_strength,
                              const int n_patches);

std::vector<NoiseLevel> GetNoiseLevel(
    const Image3F& opsin, const std::vector<float>& texture_strength,
    const float threshold, const int block_s);

void OptimizeNoiseParameters(const std::vector<NoiseLevel>& noise_level,
                             NoiseParams* noise_params);
}  // namespace pik

#endif  // PIK_NOISE_H_
