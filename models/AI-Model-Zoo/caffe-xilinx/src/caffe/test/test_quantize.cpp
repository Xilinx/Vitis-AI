#include "gtest/gtest.h"

#include "caffe/util/quantize.hpp"

namespace caffe {

TEST(DPUShiftTest, ZeroInput) {
  EXPECT_EQ(0, dpu_shift(0, 1));
  EXPECT_EQ(0, dpu_shift(0, 0));
  EXPECT_EQ(0, dpu_shift(0, -1));
}

TEST(DPUShiftTest, ZeroShift) {
  EXPECT_EQ(1, dpu_shift(1, 0));
  EXPECT_EQ(-1, dpu_shift(-1, 0));
}

TEST(DPUShiftTest, PosShift) {
  EXPECT_EQ(12, dpu_shift(3, 2));
  EXPECT_EQ(-8, dpu_shift(-2, 2));
}

TEST(DPUShiftTest, PosInputNegShift) {
  EXPECT_EQ(0, dpu_shift(0, -2));
  EXPECT_EQ(0, dpu_shift(1, -2));
  EXPECT_EQ(1, dpu_shift(2, -2));
  EXPECT_EQ(1, dpu_shift(3, -2));
  EXPECT_EQ(1, dpu_shift(4, -2));
  EXPECT_EQ(1, dpu_shift(5, -2));
  EXPECT_EQ(2, dpu_shift(6, -2));
}

TEST(DPUShiftTest, NegInputNegShift) {
  EXPECT_EQ(0, dpu_shift(-1, -2));
  EXPECT_EQ(0, dpu_shift(-2, -2));
  EXPECT_EQ(-1, dpu_shift(-3, -2));
  EXPECT_EQ(-1, dpu_shift(-4, -2));
  EXPECT_EQ(-1, dpu_shift(-5, -2));
  EXPECT_EQ(-1, dpu_shift(-6, -2));
  EXPECT_EQ(-2, dpu_shift(-7, -2));
  EXPECT_EQ(-2, dpu_shift(-8, -2));
  EXPECT_EQ(-1, dpu_shift(-10, -3));
  EXPECT_EQ(-1, dpu_shift(-11, -3));
  EXPECT_EQ(-1, dpu_shift(-12, -3));
  EXPECT_EQ(-2, dpu_shift(-13, -3));
  EXPECT_EQ(-2, dpu_shift(-14, -3));
  EXPECT_EQ(-2, dpu_shift(-15, -3));
  EXPECT_EQ(-2, dpu_shift(-16, -3));
}

TEST(FixTest, ZeroInput) {
  EXPECT_EQ(0.0, fix_data<float>(0, 0.5, -64, 63.5));
  EXPECT_EQ(0.0, fix_data<float>(0, 1, -128, 127));
  EXPECT_EQ(0.0, fix_data<float>(0, 2, -256, 254));
}

TEST(FixTest, PosInput) {
  EXPECT_EQ(1.0, fix_data<float>(1, 0.5, -64, 63.5));
  EXPECT_EQ(1.0, fix_data<float>(1, 1, -128, 127));
  EXPECT_EQ(0.0, fix_data<float>(0.99, 2, -256, 254));
  EXPECT_EQ(2.0, fix_data<float>(1, 2, -256, 254));
  EXPECT_EQ(2.0, fix_data<float>(1.01, 2, -256, 254));
}

TEST(FixTest, NegInput) {
  EXPECT_EQ(-1.0, fix_data<float>(-1, 0.5, -64, 63.5));
  EXPECT_EQ(-1.0, fix_data<float>(-1, 1, -128, 127));
  EXPECT_EQ(0.0, fix_data<float>(-0.99, 2, -256, 254));
  EXPECT_EQ(-2.0, fix_data<float>(-1, 2, -256, 254));
  EXPECT_EQ(-2.0, fix_data<float>(-1.01, 2, -256, 254));
}

TEST(FixTest, SaturationHandle) {
  EXPECT_EQ(-64.0, fix_data<float>(-64.1, 0.5, -64, 63.5));
  EXPECT_EQ(-128, fix_data<float>(-130, 1, -128, 127));
  EXPECT_EQ(254.0, fix_data<float>(255, 2, -256, 254));
  EXPECT_EQ(254.0, fix_data<float>(380, 2, -256, 254));
  EXPECT_EQ(254.0, fix_data<float>(253, 2, -256, 254));
}

}
