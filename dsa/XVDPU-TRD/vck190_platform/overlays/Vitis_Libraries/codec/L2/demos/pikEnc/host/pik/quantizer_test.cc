// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/quantizer.h"

#include <stdint.h>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "pik/bit_reader.h"
#include "pik/dct.h"
#include "pik/entropy_coder.h"

namespace pik {
namespace {

void TestEquivalence(int qxsize, int qysize, const Quantizer& quantizer1,
                     const Quantizer& quantizer2) {
  ASSERT_NEAR(quantizer1.inv_quant_dc(), quantizer2.inv_quant_dc(), 1e-7);
}

TEST(QuantizerTest, BitStreamRoundtripSameQuant) {
  const int qxsize = 8;
  const int qysize = 8;
  DequantMatrices dequant(/*need_inv_matrices=*/false);
  Quantizer quantizer1(&dequant, qxsize, qysize);
  quantizer1.SetQuant(0.17);
  std::string data = quantizer1.Encode(nullptr);
  size_t data_size = data.size();
  data.resize((data_size + 3) & ~3);
  Quantizer quantizer2(&dequant, qxsize, qysize);
  BitReader br(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  EXPECT_TRUE(quantizer2.Decode(&br));
  EXPECT_EQ(br.Position(), data_size);
  TestEquivalence(qxsize, qysize, quantizer1, quantizer2);
}

TEST(QuantizerTest, BitStreamRoundtripRandomQuant) {
  const int qxsize = 8;
  const int qysize = 8;
  DequantMatrices dequant(/*need_inv_matrices=*/false);
  Quantizer quantizer1(&dequant, qxsize, qysize);
  std::mt19937_64 rng;
  std::uniform_int_distribution<> uniform(1, 256);
  float quant_dc = 0.17f;
  ImageF qf(qxsize, qysize);
  RandomFillImage(&qf, 1.0f);
  quantizer1.SetQuantField(quant_dc, QuantField(qf));
  std::string data = quantizer1.Encode(nullptr);
  size_t data_size = data.size();
  data.resize((data_size + 3) & ~3);
  Quantizer quantizer2(&dequant, qxsize, qysize);
  BitReader br(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  EXPECT_TRUE(quantizer2.Decode(&br));
  EXPECT_EQ(br.Position(), data_size);
  TestEquivalence(qxsize, qysize, quantizer1, quantizer2);
}
}  // namespace
}  // namespace pik
