// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/ans_decode.h"
#include "pik/ans_encode.h"

#include <random>
#include <vector>

#include "gtest/gtest.h"

namespace pik {
namespace {

struct Symbol {
  int context;
  int value;
  int nbits;
};

void RoundtripTestcase(int n_histograms, int alphabet_size,
                       const std::vector<Symbol>& input_values) {
  constexpr uint16_t kMagic1 = 0x9e33;
  constexpr uint16_t kMagic2 = 0x8b04;

  std::string compressed_data(
      4 * n_histograms * alphabet_size + 4 * input_values.size() + 4, 0);
  uint8_t* storage = reinterpret_cast<uint8_t*>(&compressed_data[0]);
  size_t storage_ix = 0;

  // Create and store codes.
  std::vector<ANSEncodingData> encoding_codes(n_histograms);
  for (int i = 0; i < n_histograms; i++) {
    std::vector<uint32_t> counts(alphabet_size);
    for (int j = 0; j < counts.size(); j++) {
      counts[j] = 1 + (j == 0 ? 100000 : 0);
    }
    encoding_codes[i].BuildAndStore(&counts[0], counts.size(), &storage_ix,
                                    storage);
  }

  WriteBits(16, kMagic1, &storage_ix, storage);

  // Store the symbol stream using the codes.
  std::vector<uint8_t> dummy_context_map;
  for (int i = 0; i < n_histograms; i++) {
    dummy_context_map.push_back(i);
  }
  ANSSymbolWriter writer(encoding_codes, dummy_context_map, &storage_ix,
                         storage);
  for (const Symbol& symbol : input_values) {
    writer.VisitSymbol(symbol.value, symbol.context);
    writer.VisitBits(symbol.nbits, 0, 0);
  }
  writer.FlushToBitStream();

  WriteBits(16, kMagic2, &storage_ix, storage);

  PIK_ASSERT(storage_ix + 4 * 8 <= 8 * compressed_data.size());
  // We do not truncate the output. Reading past the end reads out zeroes
  // anyway.
  BitReader br(storage, (storage_ix + 7) / 8);
  ANSCode decoded_codes;
  ASSERT_TRUE(DecodeANSCodes(n_histograms, alphabet_size, &br, &decoded_codes));

  ASSERT_EQ(br.ReadBits(16), kMagic1);

  ANSSymbolReader reader(&decoded_codes);
  for (const Symbol& symbol : input_values) {
    int read_symbol = reader.ReadSymbol(symbol.context, &br);
    ASSERT_EQ(read_symbol, symbol.value);
    ASSERT_EQ(br.ReadBits(symbol.nbits), 0);
  }
  ASSERT_TRUE(reader.CheckANSFinalState());

  ASSERT_EQ(br.ReadBits(16), kMagic2);
}

TEST(ANSTest, EmptyRoundtrip) {
  RoundtripTestcase(2, 256, std::vector<Symbol>());
}

TEST(ANSTest, SingleSymbolRoundtrip) {
  for (int i = 0; i < 256; i++) RoundtripTestcase(2, 256, {{0, i, 0}});
}

void RoundtripRandomStream(int alphabet_size) {
  constexpr int kNumHistograms = 3;
  std::mt19937_64 rng;
  for (int i = 0; i < 100; i++) {
    std::vector<Symbol> symbols;
    for (int j = 0; j < 1000; j++) {
      Symbol s;
      s.context = std::uniform_int_distribution<>(0, kNumHistograms - 1)(rng);
      s.value = std::uniform_int_distribution<>(0, alphabet_size - 1)(rng);
      s.nbits = std::uniform_int_distribution<>(0, 16)(rng);
      symbols.push_back(s);
    }
    RoundtripTestcase(kNumHistograms, alphabet_size, symbols);
  }
}

TEST(ANSTest, RandomStreamRoundtrip3) { RoundtripRandomStream(3); }

TEST(ANSTest, RandomStreamRoundtrip256) { RoundtripRandomStream(256); }

TEST(ANSTest, RandomStreamRoundtrip1023) { RoundtripRandomStream(1023); }

}  // namespace
}  // namespace pik
