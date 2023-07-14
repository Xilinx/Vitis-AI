/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

string DecIntToHexStr(uint8_t num) {
  string str;
  int Temp = num / 16;
  int left = num % 16;
  if (Temp > 0) str += DecIntToHexStr(Temp);

  if (left < 10)
    str += (left + '0');
  else
    str += ('A' + left - 10);

  return str;
}

class ExpTable {
 public:
  ExpTable(float scale);
  virtual ~ExpTable() {}
  ExpTable(const ExpTable &rhs) = delete;

 public:
  std::vector<uint8_t> getCharTable() { return char_table_; }
  float getMiny() { return miny_; }
  float getStep() { return stepy_; }

 private:
  const float scale_;
  const int fixpos_;
  int step_;
  float miny_;
  float maxy_;
  float stepy_;
  std::vector<uint8_t> char_table_;

 private:
  unsigned char float2char(float x) {
    auto y = (unsigned int)((x - miny_) / stepy_);
    y = std::min(y, 255u);
    if (0)
      std::cout << "x " << x << " "  //
                << "y " << y << " "  //
                << std::endl;
    return y;
  }

  uint8_t exp1(int8_t x) {
    int x1 = x;
    float fx = (float)x1 * scale_;
    float fy = expf(fx);
    return float2char(fy);
  }

  void generate_char_table_8() {
    // 5bits fix point 0x80 --  0x80
    // -128 -- 120, step = 8;
    // float range = (-1, 1.0)
    char_table_.reserve(32);
    for (int i = 0; i < 0x78 + 8; i = i + 8) {
      unsigned int y = exp1(i);
      char_table_.push_back(y);
    }
    for (int i = -128; i < 0; i = i + 8) {
      unsigned int y = exp1(i);
      char_table_.push_back(y);
    }
  }
  void generate_char_table_4() {
    // 5bits fix point 0x80 --  0x80
    // -128 -- 120, step = 8;
    // float range = (-1, 1.0)
    char_table_.reserve(32);
    for (int i = -64; i < 64; i = i + 4) {
      unsigned int y = exp1(i);
      char_table_.push_back(y);
    }
  }
  void generate_char_table_2() {
    // 5bits fix point 0x80 --  0x80
    // -128 -- 120, step = 8;
    // float range = (-1, 1.0)
    char_table_.reserve(32);
    for (int i = -32; i < 32; i = i + 2) {
      unsigned int y = exp1(i);
      char_table_.push_back(y);
    }
  }
  void generate_char_table_1() {
    // 5bits fix point 0x80 --  0x80
    // -128 -- 120, step = 8;
    // float range = (-1, 1.0)
    char_table_.reserve(32);
    for (int i = -16; i < 16; i = i + 1) {
      unsigned int y = exp1(i);
      char_table_.push_back(y);
    }
  }
};

ExpTable::ExpTable(float scale)
    : scale_(scale),
      fixpos_(std::abs((int)log2(scale)))
// miny_(expf(-128 * scale_)),
// maxy_(expf(127 * scale_)),
// stepy_((maxy_ - miny_) / 256.0f)
{
  switch (fixpos_) {
    case 8:
    case 7:
    case 6:
      miny_ = expf(-128 * scale_);
      maxy_ = expf(120 * scale_);
      stepy_ = ((maxy_ - miny_) / 255.0f);
      generate_char_table_8();
      break;
    case 5:
    case 4:
      miny_ = expf(-64 * scale_);
      maxy_ = expf(60 * scale_);
      stepy_ = ((maxy_ - miny_) / 255.0f);
      generate_char_table_4();
      break;
    case 3:
    case 2:
      miny_ = expf(-32 * scale_);
      maxy_ = expf(30 * scale_);
      stepy_ = ((maxy_ - miny_) / 255.0f);
      generate_char_table_2();

      break;
    case 1:
      miny_ = expf(-16 * scale_);
      maxy_ = expf(15 * scale_);
      stepy_ = ((maxy_ - miny_) / 255.0f);
      generate_char_table_1();

      break;
    default:
      break;
  }

  if (0) {
    std::cout << "miny_ " << miny_ << " "    //
              << "maxy_ " << maxy_ << " "    //
              << "stepy_ " << stepy_ << " "  //
              << std::endl;
    for (unsigned int t : char_table_) {
      std::cout << t << " "  //
          ;
    }
    std::cout << std::endl;
  }
}

void printTable2Hex(std::vector<uint8_t> table) {
  // 8个一组倒序输出
  for (int i = 0; i < 4; ++i) {
    for (int j = 7; j >= 0; --j) {
      uint8_t t = table[i * 8 + j];
      if (t < 16) {
        std::cout << "0" << DecIntToHexStr(t);
      } else {
        std::cout << DecIntToHexStr(t);
      }
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  ExpTable expTable_8(0.00390625);
  ExpTable expTable_7(0.0078125);
  ExpTable expTable_6(0.015625);
  ExpTable expTable_5(0.03125);
  ExpTable expTable_4(0.0625);
  ExpTable expTable_3(0.125);
  ExpTable expTable_2(0.25);
  ExpTable expTable_1(0.5);

  std::cout << "miny " << expTable_8.getMiny() << " "
            << "step " << expTable_8.getStep() << " " << std::endl;
  printTable2Hex(expTable_8.getCharTable());

  std::cout << "miny " << expTable_7.getMiny() << " "
            << "step " << expTable_7.getStep() << " " << std::endl;
  printTable2Hex(expTable_7.getCharTable());

  std::cout << "miny " << expTable_6.getMiny() << " "
            << "step " << expTable_6.getStep() << " " << std::endl;
  printTable2Hex(expTable_6.getCharTable());

  std::cout << "miny " << expTable_5.getMiny() << " "
            << "step " << expTable_5.getStep() << " " << std::endl;
  printTable2Hex(expTable_5.getCharTable());

  std::cout << "miny " << expTable_4.getMiny() << " "
            << "step " << expTable_4.getStep() << " " << std::endl;
  printTable2Hex(expTable_4.getCharTable());

  std::cout << "miny " << expTable_3.getMiny() << " "
            << "step " << expTable_3.getStep() << " " << std::endl;
  printTable2Hex(expTable_3.getCharTable());

  std::cout << "miny " << expTable_2.getMiny() << " "
            << "step " << expTable_2.getStep() << " " << std::endl;
  printTable2Hex(expTable_2.getCharTable());

  std::cout << "miny " << expTable_1.getMiny() << " "
            << "step " << expTable_1.getStep() << " " << std::endl;
  printTable2Hex(expTable_1.getCharTable());

  return 0;
}
