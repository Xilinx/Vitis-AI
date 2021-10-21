// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/lossless16.h"


#include "gtest/gtest.h"

namespace pik {
namespace {

ImageU GenerateGray(size_t xsize, size_t ysize) {
  ImageU image(xsize, ysize);

  for (size_t y = 0; y < ysize; ++y) {
    auto* row = image.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row[x] = x * y;
    }
  }

  return image;
}

Image3U GenerateColor(size_t xsize, size_t ysize) {
  Image3U image(xsize, ysize);

  for (int c = 0; c < 3; c++) {
    for (size_t y = 0; y < ysize; ++y) {
      auto* row = image.PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row[x] = x * y * (c + 1);
      }
    }
  }

  return image;
}

void ExpectImagesEqual(const ImageU& a, const ImageU& b) {
  EXPECT_EQ(a.xsize(), b.xsize());
  EXPECT_EQ(a.ysize(), b.ysize());
  size_t numdiff = 0;
  for (size_t y = 0; y < a.ysize(); ++y) {
    const auto* rowa = a.Row(y);
    const auto* rowb = b.Row(y);
    for (size_t x = 0; x < b.xsize(); ++x) {
      if (rowa[x] != rowb[x]) {
        numdiff++;
      }
    }
  }
  EXPECT_EQ(0, numdiff);
}

void ExpectImagesEqual(const Image3U& a, const Image3U& b) {
  EXPECT_EQ(a.xsize(), b.xsize());
  EXPECT_EQ(a.ysize(), b.ysize());
  size_t numdiff = 0;
  for (int c = 0; c < 3; c++) {
    for (size_t y = 0; y < a.ysize(); ++y) {
      const auto* rowa = a.PlaneRow(c, y);
      const auto* rowb = b.PlaneRow(c, y);
      for (size_t x = 0; x < b.xsize(); ++x) {
        if (rowa[x] != rowb[x]) {
          numdiff++;
        }
      }
    }
  }
  EXPECT_EQ(0, numdiff);
}

void TestGray(size_t xsize, size_t ysize) {
  ImageU image = GenerateGray(xsize, ysize);
  PaddedBytes bytes;

  EXPECT_TRUE(Grayscale16bit_compress(image, &bytes));

  ImageU image2;
  size_t pos = 0;
  EXPECT_TRUE(Grayscale16bit_decompress(bytes, &pos, &image2));

  ExpectImagesEqual(image, image2);
}

void TestColor(size_t xsize, size_t ysize) {
  Image3U image = GenerateColor(xsize, ysize);
  PaddedBytes bytes;

  EXPECT_TRUE(Colorful16bit_compress(image, &bytes));
  std::cout << "bytes.size(): " << bytes.size() << std::endl;

  Image3U image2;
  size_t pos = 0;
  EXPECT_TRUE(Colorful16bit_decompress(bytes, &pos, &image2));

  ExpectImagesEqual(image, image2);
}

TEST(Lossless16Test, Grayscale) {
  TestGray(1, 1);
  TestGray(3, 5);
  TestGray(256, 256);
  TestGray(555, 555);
  TestGray(5, 800);
}

TEST(Lossless16Test, Color) {
  TestColor(1, 1);
  TestColor(3, 5);
  TestColor(256, 256);
  TestColor(555, 555);
  TestColor(5, 800);
}

}  // namespace
}  // namespace pik

