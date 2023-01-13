/*
 * Copyright 2019 Xilinx Inc.
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
#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace vart {

class Bit4 {
 public:
  enum class Pos { LOW, HIGH };

 protected:
  Bit4() : data_{}, pos_(Pos::LOW), if_signed_(true) {
    data_.i = int8_t(0);
  }
  ~Bit4() = default;

 public:
  void set_valid_pos(Pos pos) { pos_ = pos; }
  Pos get_valid_pos() const { return pos_; }

  template <typename T,
            typename std::enable_if<std::is_integral<T>::value &&
                                        !std::is_same<bool, T>::value,
                                    int>::type = 0>
  void set_value(T t, Pos pos = Pos::LOW) {
    data_.i = *(int8_t*)(&t);
    set_valid_pos(pos);
  }

  int8_t get_origin_int8() const { return data_.i; }
  uint8_t get_origin_uint8() const { return data_.u; }

  // funcs for output
  void print() const { std::cout << get_dbg_str() << std::endl; }

  friend std::ostream& operator<<(std::ostream& out, const Bit4& b) {
    out << b.get_dbg_str();
    return out;
  }

 protected:
  std::string get_dbg_str() const {
    std::string s;

    // signed flag
    if (if_signed_)
      s = "signed";
    else
      s = "unsigned";

    // origin hex
    s += ", origin hex: 0x";
    std::ostringstream ostr;
    ostr << std::hex << std::setw(2) << std::setfill('0')
         << (uint32_t)(data_.u);
    s += ostr.str();
    ostr.str("");

    // valid pos
    s += ", valid pos: ";
    if (pos_ == Pos::LOW)
      s += "LOW";
    else
      s += "HIGH";

    // value
    s += ", value: ";
    if (if_signed_) {
      if (pos_ == Pos::LOW) {
        ostr << std::dec << (int)data_.ib.first << "(0x";
        ostr << std::hex << std::setw(1) << std::setfill('0')
             << (uint32_t)(data_.ub.first);
        ostr << ")";
      } else {
        ostr << std::dec << (int)data_.ib.second << "(0x";
        ostr << std::hex << std::setw(1) << std::setfill('0')
             << (uint32_t)(data_.ub.second);
        ostr << ")";
      }
    } else {
      if (pos_ == Pos::LOW) {
        ostr << std::dec << (uint32_t)data_.ub.first << "(0x";
        ostr << std::hex << std::setw(1) << std::setfill('0')
             << (uint32_t)(data_.ub.first);
        ostr << ")";
      } else {
        ostr << std::dec << (uint32_t)data_.ub.second << "(0x";
        ostr << std::hex << std::setw(1) << std::setfill('0')
             << (uint32_t)(data_.ub.second);
        ostr << ")";
      }
    }
    s += ostr.str();
    ostr.str("");

    return s;
  }

 protected:
  struct InternalBit4 {
    int8_t first : 4, second : 4;
  };
  struct InternalUBit4 {
    uint8_t first : 4, second : 4;
  };

  union InternalData {
    int8_t i;
    uint8_t u;
    InternalBit4 ib;
    InternalUBit4 ub;
  };

  InternalData data_;
  Pos pos_;
  bool if_signed_;
};

class SignedBit4 : public Bit4 {
 public:
  SignedBit4() : Bit4() { if_signed_ = true; }
  ~SignedBit4() = default;
  SignedBit4(const SignedBit4&) = default;
  SignedBit4& operator=(const SignedBit4&) = default;
  SignedBit4(SignedBit4&&) = default;
  SignedBit4& operator=(SignedBit4&&) = default;

  template <typename T,
            typename std::enable_if<std::is_integral<T>::value
                                        // && std::is_signed<T>::value
                                        && !std::is_same<bool, T>::value,
                                    int>::type = 0>
  SignedBit4(T i, Pos pos = Pos::LOW) {
    if_signed_ = true;
    data_.i = *(int8_t*)(&i);
    set_valid_pos(pos);
  }

 public:
  int8_t get_value() const {
    if (pos_ == Pos::LOW)
      return data_.ib.first;
    else
      return data_.ib.second;
  }
};

class UnsignedBit4 : public Bit4 {
 public:
  UnsignedBit4() : Bit4() { if_signed_ = false; }
  ~UnsignedBit4() = default;
  UnsignedBit4(const UnsignedBit4&) = default;
  UnsignedBit4& operator=(const UnsignedBit4&) = default;
  UnsignedBit4(UnsignedBit4&&) = default;
  UnsignedBit4& operator=(UnsignedBit4&&) = default;

  template <typename T,
            typename std::enable_if<std::is_integral<T>::value
                                        // && std::is_unsigned<T>::value
                                        && !std::is_same<bool, T>::value,
                                    int>::type = 0>
  UnsignedBit4(T u, Pos pos = Pos::LOW) {
    if_signed_ = false;
    data_.u = *(uint8_t*)(&u);
    set_valid_pos(pos);
  }

 public:
  uint8_t get_value() const {
    if (pos_ == Pos::LOW)
      return data_.ub.first;
    else
      return data_.ub.second;
  }
};

// overload operator ==
inline bool operator==(const SignedBit4& a, const SignedBit4& b) {
  return a.get_value() == b.get_value();
}
inline bool operator==(const SignedBit4& a, const UnsignedBit4& b) {
  return a.get_value() == (int8_t)b.get_value();
}
inline bool operator==(const UnsignedBit4& a, const SignedBit4& b) {
  return (int8_t)a.get_value() == b.get_value();
}
inline bool operator==(const UnsignedBit4& a, const UnsignedBit4& b) {
  return a.get_value() == b.get_value();
}

// overload operator !=
inline bool operator!=(const SignedBit4& a, const SignedBit4& b) {
  return !operator==(a, b);
}
inline bool operator!=(const SignedBit4& a, const UnsignedBit4& b) {
  return !operator==(a, b);
}
inline bool operator!=(const UnsignedBit4& a, const SignedBit4& b) {
  return !operator==(a, b);
}
inline bool operator!=(const UnsignedBit4& a, const UnsignedBit4& b) {
  return !operator==(a, b);
}

// overload operator +
inline int8_t operator+(const SignedBit4& a, const SignedBit4& b) {
  return a.get_value() + b.get_value();
}
inline int8_t operator+(const SignedBit4& a, const UnsignedBit4& b) {
  return a.get_value() + (int8_t)b.get_value();
}
inline int8_t operator+(const UnsignedBit4& a, const SignedBit4& b) {
  return (int8_t)a.get_value() + b.get_value();
}
inline int8_t operator+(const UnsignedBit4& a, const UnsignedBit4& b) {
  return (int8_t)a.get_value() + (int8_t)b.get_value();
}

// overload operator *
inline int8_t operator*(const SignedBit4& a, const SignedBit4& b) {
  return a.get_value() * b.get_value();
}
inline int8_t operator*(const SignedBit4& a, const UnsignedBit4& b) {
  return a.get_value() * (int8_t)b.get_value();
}
inline int8_t operator*(const UnsignedBit4& a, const SignedBit4& b) {
  return (int8_t)a.get_value() * b.get_value();
}
inline int8_t operator*(const UnsignedBit4& a, const UnsignedBit4& b) {
  return (int8_t)a.get_value() * (int8_t)b.get_value();
}

void bump_idx(std::vector<int32_t>& idx, const std::vector<int32_t>& shape);
void bump_idx(std::vector<int32_t>& idx, const std::vector<uint32_t>& shape);

template <typename T>
void dt_2_unsigned4bit(T* src, uint8_t* dst, int num,
                       std::vector<int32_t> shape,
                       std::vector<int32_t> stride) {
  auto idx = std::vector<int32_t>(shape.size(), 0);
  for (auto i = 0; i < num; i++) {
    auto offset = 0U;
    for (auto k = 0U; k < idx.size(); k++) {
      offset += idx[k] * stride[k] * 4;
    }
    dst[offset / 8] =
        (offset % 8 == 0)
            ? (dst[offset / 8] & 0xf0) | UnsignedBit4(src[i]).get_value()
            : (dst[offset / 8] & 0x0f) |
                  (UnsignedBit4(src[i]).get_value() << 4);
    bump_idx(idx, shape);
  }
}

template <typename T>
void dt_2_signed4bit(T* src, int8_t* dst, int num, std::vector<int32_t> shape,
                     std::vector<int32_t> stride) {
  dt_2_unsigned4bit(src, reinterpret_cast<uint8_t*>(dst), num, shape, stride);
}

template <typename T>
void signed4bit_2_dt(int8_t* src, T* dst, int32_t num,
                     std::vector<int32_t> shape, std::vector<int32_t> stride) {
  auto idx = std::vector<int32_t>(shape.size(), 0);
  for (auto i = 0; i < num; i++) {
    auto offset = 0U;
    for (auto k = 0U; k < idx.size(); k++) {
      offset += idx[k] * stride[k] * 4;
    }
    auto pos = offset % 8 == 0 ? Bit4::Pos::LOW : Bit4::Pos::HIGH;
    dst[i] = static_cast<T>(SignedBit4(src[offset / 8], pos).get_value());
    bump_idx(idx, shape);
  }
}

template <typename T>
void unsigned4bit_2_dt(uint8_t* src, T* dst, int num,
                       std::vector<int32_t> shape,
                       std::vector<int32_t> stride) {
  auto idx = std::vector<int32_t>(shape.size(), 0);
  for (auto i = 0; i < num; i++) {
    auto offset = 0U;
    for (auto k = 0U; k < idx.size(); k++) {
      offset += idx[k] * stride[k] * 4;
    }
    auto pos = offset % 8 == 0 ? Bit4::Pos::LOW : Bit4::Pos::HIGH;
    dst[i] = static_cast<T>(UnsignedBit4(src[offset / 8], pos).get_value());
    bump_idx(idx, shape);
  }
}

}  // namespace vart
