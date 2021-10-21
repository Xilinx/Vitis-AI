// Copyright 2018 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_FIELDS_H_
#define PIK_FIELDS_H_

// Forward/backward-compatible 'bundles' with auto-serialized 'fields'.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "pik/bit_reader.h"
#include "pik/bits.h"
#include "pik/brotli.h"
#include "pik/common.h"
#include "pik/compiler_specific.h"
#include "pik/field_encodings.h"
#include "pik/status.h"
#include "pik/write_bits.h"

#ifndef PIK_FIELDS_TRACE
#define PIK_FIELDS_TRACE 0
#endif

namespace pik {

// Chooses one of four encodings based on an a-priori "distribution":
// - raw: if IsRaw(distribution), send RawBits(distribution) = 1..32 raw bits;
//   This are values larger than ~32u, use kU32RawBits + #bits.
// - non-raw: send a 2-bit selector to choose byte b from "distribution",
//   least significant byte first. Then the value is encoded according to b:
//   -- direct: if b & 0x80, the value is b & 0x7F
//   -- offset: else if b & 0x40, the value is derived from (b & 7) + 1
//              extra bits plus an offset ((b >> 3) & 7) + 1.
//   -- extra: otherwise, the value is derived from b extra bits
//             (must be 1-32 extra bits)
// This is faster to decode and denser than Exp-Golomb or Gamma codes when both
// small and large values occur.
//
// Examples:
// Raw:    distribution 0xFFFFFFEF, value 32768 => 1000000000000000
// Direct: distribution 0x06A09088, value 32 => 10 (selector 2, b=0xA0).
// Extra:  distribution 0x08060402, value 7 => 01 0111 (selector 1, b=4).
// Offset: distribution 0x68584801, value 7 => 11 1 (selector 3, offset 5 + 1).
//
// Bit for bit example:
// An encoding mapping the following prefix code:
// 00 -> 0
// 01x -> 1..2
// 10xx -> 3..7
// 11xxxxxxxx -> 8..263
// Can be made with distribution 0x7F514080. Dissecting this from hex digits
// left to right:
// 7: 0x40 flag for this byte and 2 bits of offset 8 for 8..263
// F: final bit of offset 8 and 3 bits setting extra to 7+1 for 8..263.
// 5: 0x40 flag for this byte and 2 bits of offset 3 for 3..7
// 1: One bit indicating window size 2 set for 3..7
// 4: 0x40 flag for this byte, no offset bits set, offset 0+1 for 1..2
// 0: no bits set in this flag, offset and extra bits set to 0 indicating an
//    offset 1 and extra 1 for 1..2
// 8: 0x80 flag set to indicate direct value for 0
// 0: bits of the direct value 0
class U32Coder {
 public:
  // Byte flag indicating direct value.
  static const uint32_t kDirect = 0x80;
  // Byte flag indicating extra bits with offset rather than pure extra bits.
  static const uint32_t kOffset = 0x40;

  static size_t MaxEncodedBits(const uint32_t distribution) {
    ValidateDistribution(distribution);
    if (IsRaw(distribution)) return RawBits(distribution);
    size_t extra_bits = 0;
    for (int selector = 0; selector < 4; ++selector) {
      const size_t b = Lookup(distribution, selector);
      if (b & kDirect) {
        continue;
      } else {
        extra_bits = std::max<size_t>(extra_bits, GetExtraBits(b));
      }
    }
    return 2 + extra_bits;
  }

  static Status CanEncode(const uint32_t distribution, const uint32_t value,
                          size_t* PIK_RESTRICT encoded_bits) {
    ValidateDistribution(distribution);
    int selector;
    size_t total_bits;
    const Status ok =
        ChooseEncoding(distribution, value, &selector, &total_bits);
    *encoded_bits = ok ? total_bits : 0;
    return ok;
  }

  static uint32_t Read(const uint32_t distribution,
                       BitReader* PIK_RESTRICT reader) {
    ValidateDistribution(distribution);
    if (IsRaw(distribution)) {
      return reader->ReadBits(RawBits(distribution));
    }
    const int selector = reader->ReadFixedBits<2>();
    const size_t b = Lookup(distribution, selector);
    if (b & kDirect) {
      return b & 0x7F;
    } else {
      uint32_t offset = GetOffset(b);
      uint32_t extra_bits = GetExtraBits(b);
      return reader->ReadBits(extra_bits) + offset;
    }
  }

  // Returns false if the value is too large to encode.
  static Status Write(const uint32_t distribution, const uint32_t value,
                      size_t* pos, uint8_t* storage) {
    int selector;
    size_t total_bits;
    PIK_RETURN_IF_ERROR(
        ChooseEncoding(distribution, value, &selector, &total_bits));

    if (IsRaw(distribution)) {
      WriteBits(RawBits(distribution), value, pos, storage);
      return true;
    }
    WriteBits(2, selector, pos, storage);

    const size_t b = Lookup(distribution, selector);
    if ((b & kDirect) == 0) {  // Nothing more to write for direct encoding
      uint32_t offset = GetOffset(b);
      PIK_ASSERT(value >= offset);
      WriteBits(total_bits - 2, value - offset, pos, storage);
    }

    return true;
  }

 private:
  static PIK_INLINE bool IsRaw(const uint32_t distribution) {
    return distribution > kU32RawBits;
  }

  static PIK_INLINE size_t RawBits(const uint32_t distribution) {
    PIK_ASSERT(IsRaw(distribution));
    return distribution - kU32RawBits;
  }

  // Returns one byte from "distribution" at index "selector".
  static PIK_INLINE size_t Lookup(const uint32_t distribution,
                                  const int selector) {
    PIK_ASSERT(!IsRaw(distribution));
    return (distribution >> (selector * 8)) & 0xFF;
  }

  static PIK_INLINE uint32_t GetOffset(const uint8_t b) {
    PIK_ASSERT(!(b & kDirect));
    if (b & kOffset) return ((b >> 3) & 7) + 1;
    return 0;
  }

  static PIK_INLINE uint32_t GetExtraBits(const uint8_t b) {
    PIK_ASSERT(!(b & kDirect));
    if (b & kOffset) return (b & 7) + 1;
    PIK_ASSERT(b != 0 && b <= 32);
    return b;
  }

  static void ValidateDistribution(const uint32_t distribution) {
#if PIK_ENABLE_ASSERT
    if (IsRaw(distribution)) return;  // raw 1..32: OK
    for (int selector = 0; selector < 4; ++selector) {
      const size_t b = Lookup(distribution, selector);
      if (b & kDirect) {
        continue;  // direct: OK
      } else if (b & kOffset) {
        continue;  // extra with offset: OK
      } else {
        // Forbid b = 0 because it requires an extra call to read/write 0 bits;
        // to encode a zero value, use b = kDirect instead.
        if (b == 0 || b > 32) {
          fprintf(stderr, "Invalid distribution %8x[%d] == %zu\n", distribution,
                  selector, b);
          PIK_ASSERT(false);
        }
      }
    }
#endif
  }

  static Status ChooseEncoding(const uint32_t distribution,
                               const uint32_t value, int* PIK_RESTRICT selector,
                               size_t* PIK_RESTRICT total_bits) {
    const size_t bits_required = 32 - NumZeroBitsAboveMSB(value);
    PIK_ASSERT(bits_required <= 32);

    *selector = 0;
    *total_bits = 0;

    if (IsRaw(distribution)) {
      const size_t raw_bits = RawBits(distribution);
      if (bits_required > raw_bits) {
        return PIK_FAILURE("Insufficient raw bits");
      }
      *total_bits = raw_bits;
      return true;
    }

    // It is difficult to verify whether "distribution" is sorted, so check all
    // selectors and keep the one with the fewest total_bits.
    *total_bits = 64;  // more than any valid encoding
    for (int s = 0; s < 4; ++s) {
      const size_t b = Lookup(distribution, s);
      if (b & kDirect) {
        if ((b & 0x7F) == value) {
          *selector = s;
          *total_bits = 2;
          return true;  // Done, can't improve upon a direct encoding.
        }
        continue;
      }

      uint32_t extra_bits = GetExtraBits(b);
      if (b & kOffset) {
        uint32_t offset = GetOffset(b);
        if (value < offset || value >= offset + (1u << extra_bits)) continue;
      } else {
        if (bits_required > extra_bits) continue;
      }

      // Better than prior encoding, remember it:
      if (2 + extra_bits < *total_bits) {
        *selector = s;
        *total_bits = 2 + extra_bits;
      }
    }

    if (*total_bits == 64) return PIK_FAILURE("No feasible selector found");

    return true;
  }
};

// Encodes 64-bit unsigned integers with a fixed distribution, taking 2 bits
// to encode 0, 6 bits to encode 1 to 16, 10 bits to encode 17 to 272, 15 bits
// to encode up to 4095, and in the order of log2(value) * 1.125 bits for higher
// values.
class U64Coder {
 public:
  static uint64_t Read(BitReader* PIK_RESTRICT reader) {
    uint64_t selector = reader->ReadFixedBits<2>();
    if (selector == 0) {
      return 0;
    }
    if (selector == 1) {
      return 1 + reader->ReadFixedBits<4>();
    }
    if (selector == 2) {
      return 17 + reader->ReadFixedBits<8>();
    }

    // selector 3, varint, first 12 bits, later groups are 8 bits
    uint64_t result = reader->ReadFixedBits<12>();

    uint64_t shift = 12;
    while (reader->ReadFixedBits<1>()) {
      if (shift == 60) {
        result |= static_cast<uint64_t>(reader->ReadFixedBits<4>()) << shift;
        break;
      }
      result |= static_cast<uint64_t>(reader->ReadFixedBits<8>()) << shift;
      shift += 8;
    }

    return result;
  }

  // Returns false if the value is too large to encode.
  static Status Write(uint64_t value, size_t* pos, uint8_t* storage) {
    if (value == 0) {
      // Selector: use 0 bits, value 0
      WriteBits(2, 0, pos, storage);
    } else if (value <= 16) {
      // Selector: use 4 bits, value 1..16
      WriteBits(2, 1, pos, storage);
      WriteBits(4, value - 1, pos, storage);
    } else if (value <= 272) {
      // Selector: use 8 bits, value 17..272
      WriteBits(2, 2, pos, storage);
      WriteBits(8, value - 17, pos, storage);
    } else {
      // Selector: varint, first a 12-bit group, after that per 8-bit group.
      WriteBits(2, 3, pos, storage);
      WriteBits(12, value & 4095, pos, storage);
      value >>= 12;
      int shift = 12;
      while (value > 0 && shift < 60) {
        // Indicate varint not done
        WriteBits(1, 1, pos, storage);
        WriteBits(8, value & 255, pos, storage);
        value >>= 8;
        shift += 8;
      }
      if (value > 0) {
        // This only could happen if shift == 60.
        WriteBits(1, 1, pos, storage);
        WriteBits(4, value & 15, pos, storage);
        // Implicitly closed sequence, no extra stop bit is required.
      } else {
        // Indicate end of varint
        WriteBits(1, 0, pos, storage);
      }
    }

    return true;
  }

  // Can always encode, but useful because it also returns bit size.
  static Status CanEncode(uint64_t value, size_t* PIK_RESTRICT encoded_bits) {
    if (value == 0) {
      *encoded_bits = 2;  // 2 selector bits
    } else if (value <= 16) {
      *encoded_bits = 2 + 4;  // 2 selector bits + 4 payload bits
    } else if (value <= 272) {
      *encoded_bits = 2 + 8;  // 2 selector bits + 8 payload bits
    } else {
      *encoded_bits = 2 + 12;  // 2 selector bits + 12 payload bits
      value >>= 12;
      int shift = 12;
      while (value > 0 && shift < 60) {
        *encoded_bits += 1 + 8;  // 1 continuation bit + 8 payload bits
        value >>= 8;
        shift += 8;
      }
      if (value > 0) {
        // This only could happen if shift == 60.
        *encoded_bits += 1 + 4;  // 1 continuation bit + 4 payload bits
      } else {
        *encoded_bits += 1;  // 1 stop bit
      }
    }

    return true;
  }
};

// 3-bit code for exif orientation, encoding values 1-8.
class OrientationCoder {
 public:
  static uint32_t Read(BitReader* PIK_RESTRICT reader) {
    return 1u + reader->ReadFixedBits<3>();
  }

  static Status Write(uint32_t value, size_t* pos, uint8_t* storage) {
    WriteBits(3, value - 1, pos, storage);
    return true;
  }

  static Status CanEncode(uint32_t value, size_t* PIK_RESTRICT encoded_bits) {
    *encoded_bits = 3;
    return value >= 1 && value <= 8;
  }
};

// Coder for byte arrays: stores encoding and #bytes via U32Coder, then raw or
// Brotli-compressed bytes.
class BytesCoder {
  static const int kBrotliQuality = 6;

 public:
  static Status CanEncode(BytesEncoding encoding, const PaddedBytes& value,
                          size_t* PIK_RESTRICT encoded_bits) {
    PIK_ASSERT(encoding == BytesEncoding::kRaw ||
               encoding == BytesEncoding::kBrotli);
    if (value.empty()) {
      return U32Coder::CanEncode(kU32Direct3Plus8,
                                 static_cast<uint32_t>(BytesEncoding::kNone),
                                 encoded_bits);
    }

    PaddedBytes compressed;
    const PaddedBytes* store_what = &value;

    // Note: we will compress a second time when Write is called.
    if (encoding == BytesEncoding::kBrotli) {
      PIK_RETURN_IF_ERROR(BrotliCompress(kBrotliQuality, value, &compressed));
      if (compressed.size() < value.size()) {
        store_what = &compressed;
      } else {
        encoding = BytesEncoding::kRaw;
      }
    }

    size_t bits_encoding, bits_size;
    PIK_RETURN_IF_ERROR(U32Coder::CanEncode(kU32Direct3Plus8,
                                            static_cast<uint32_t>(encoding),
                                            &bits_encoding) &&
                        U64Coder::CanEncode(store_what->size(), &bits_size));
    *encoded_bits =
        bits_encoding + bits_size + store_what->size() * kBitsPerByte;
    return true;
  }

  static Status Read(BitReader* PIK_RESTRICT reader,
                     PaddedBytes* PIK_RESTRICT value) {
    const BytesEncoding encoding =
        static_cast<BytesEncoding>(U32Coder::Read(kU32Direct3Plus8, reader));
    if (encoding == BytesEncoding::kNone) {
      value->clear();
      return true;
    }
    if (encoding != BytesEncoding::kRaw && encoding != BytesEncoding::kBrotli) {
      return PIK_FAILURE("Unrecognized BytesEncoding encoding");
    }

    const uint64_t num_bytes = U64Coder::Read(reader);
    // Prevent fuzzer from running out of memory.
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    if (num_bytes > 16 * 1024 * 1024) {
      return PIK_FAILURE("BytesCoder size too large for fuzzer");
    }
#endif
    value->resize(num_bytes);
    if (num_bytes != 0 && value->size() == 0) {
      return PIK_FAILURE("Failed to allocate memory for BytesCoder");
    }

    // Read groups of bytes without calling FillBitBuffer every time.
    constexpr size_t kBytesPerGroup = 4;  // guaranteed by FillBitBuffer
    uint32_t i;
    for (i = 0; i + kBytesPerGroup <= value->size(); i += kBytesPerGroup) {
      reader->FillBitBuffer();
#if PIK_BYTE_ORDER_LITTLE
      const uint32_t buf = reader->PeekFixedBits<32>();
      reader->Advance(32);
      memcpy(value->data() + i, &buf, 4);
#else
      for (int idx_byte = 0; idx_byte < kBytesPerGroup; ++idx_byte) {
        value->data()[i + idx_byte] = reader->PeekFixedBits<8>();
        reader->Advance(8);
      }
#endif
    }

    reader->FillBitBuffer();
    for (; i < value->size(); ++i) {
      value->data()[i] = reader->PeekFixedBits<8>();
      reader->Advance(8);
    }

    if (encoding == BytesEncoding::kBrotli) {
      const size_t kMaxOutput = 1ULL << 32;
      size_t bytes_read = 0;
      PaddedBytes decompressed;
      if (PIK_UNLIKELY(!BrotliDecompress(*value, kMaxOutput, &bytes_read,
                                         &decompressed))) {
        return false;
      }
      if (bytes_read != value->size()) {
        PIK_NOTIFY_ERROR("Read too few");
      }
      value->swap(decompressed);
    }
    return true;
  }

  static Status Write(BytesEncoding encoding, const PaddedBytes& value,
                      size_t* PIK_RESTRICT pos, uint8_t* storage) {
    PIK_ASSERT(encoding == BytesEncoding::kRaw ||
               encoding == BytesEncoding::kBrotli);
    if (value.empty()) {
      return U32Coder::Write(kU32Direct3Plus8,
                             static_cast<uint32_t>(BytesEncoding::kNone), pos,
                             storage);
    }

    PaddedBytes compressed;
    const PaddedBytes* store_what = &value;

    if (encoding == BytesEncoding::kBrotli) {
      PIK_RETURN_IF_ERROR(BrotliCompress(kBrotliQuality, value, &compressed));
      if (compressed.size() < value.size()) {
        store_what = &compressed;
      } else {
        encoding = BytesEncoding::kRaw;
      }
    }

    PIK_RETURN_IF_ERROR(U32Coder::Write(
        kU32Direct3Plus8, static_cast<uint32_t>(encoding), pos, storage));
    PIK_RETURN_IF_ERROR(
        U64Coder::Write(store_what->size(), pos, storage));

    size_t i = 0;
#if PIK_BYTE_ORDER_LITTLE
    // Write 4 bytes at a time
    uint32_t buf;
    for (; i + 4 <= store_what->size(); i += 4) {
      memcpy(&buf, store_what->data() + i, 4);
      WriteBits(32, buf, pos, storage);
    }
#endif

    // Write remaining bytes
    for (; i < store_what->size(); ++i) {
      WriteBits(8, store_what->data()[i], pos, storage);
    }
    return true;
  }
};

// A "bundle" is a forward- and backward compatible collection of fields.
// They are used for FileHeader/FrameHeader/GroupHeader. Bundles can be extended
// by appending(!) fields. Optional fields may be omitted from the bitstream by
// conditionally visiting them. When reading new bitstreams with old code, we
// skip unknown fields at the end of the bundle. This requires storing the
// amount of extra appended bits, and that fields are visited in chronological
// order of being added to the format, because old decoders cannot skip some
// future fields and resume reading old fields. Similarly, new readers query
// bits in an "extensions" field to skip (groups of) fields not present in old
// bitstreams. Note that each bundle must include an "extensions" field prior to
// freezing the format, otherwise it cannot be extended.
//
// To ensure interoperability, there will be no opaque fields.
//
// HOWTO:
// - basic usage: define a struct with member variables ("fields") and a
//   VisitFields(v) member function that calls v->U32/Bool etc. for each field,
//   specifying their default values. The ctor must call Bundle::Init(this).
//
// - print a trace of visitors: ensure each bundle has a static Name() member
//   function, and #define PIK_FIELDS_TRACE 1.
//
// - optional fields: in VisitFields, add if (v->Conditional(your_condition))
//   { v->U32(dist, default, &field); }. This prevents reading/writing field
//   if !your_condition, which is typically computed from a prior field.
//   WARNING: do not add an else branch; to ensure all fields are initialized,
//   instead add another if (v->Conditional(!your_condition)).
//
// - repeated fields: for dynamic sizes, add a std::vector field and in
//   VisitFields, call v->SetSizeWhenReading before accessing the field. For
//   static or bounded sizes, use an array or std::array. In all cases, simply
//   visit each array element as if it were a normal field.
//
// - nested bundles: add a bundle as a normal field and in VisitFields call
//   PIK_RETURN_IF_ERROR(v->VisitNested(&nested));
//
// - allow future extensions: define a "uint64_t extensions" field and call
//   v->BeginExtensions(&extensions) after visiting all non-extension fields,
//   and `return v->EndExtensions();` after the last extension field.
//
// - encode an entire bundle in one bit if ALL its fields equal their default
//   values: add a "bool all_default" field and as the first visitor:
//   if (v->AllDefault(*this, &all_default)) return true;
//   Note: if extensions are present, AllDefault() == false.

class Bundle {
 public:
  // These are called from headers.cc.

  template <class T>
  static void Init(T* PIK_RESTRICT t) {
    Trace("Init");
    InitVisitor visitor;
    if (!visitor.Visit(t)) {
      PIK_ASSERT(false);  // Init should never fail.
    }
  }

  // Returns whether ALL fields (including `extensions`, if present) are equal
  // to their default value.
  template <class T>
  static bool AllDefault(const T& t) {
    Trace("[[AllDefault");
    AllDefaultVisitor visitor;
    if (!visitor.VisitConst(t)) {
      PIK_ASSERT(false);  // AllDefault should never fail.
    }
#if PIK_FIELDS_TRACE
    printf("  %d]]\n", visitor.AllDefault());
#endif
    return visitor.AllDefault();
  }

  // Prepares for Write(): "*total_bits" is the amount of storage required;
  // "*extension_bits" must be passed to Write().
  template <class T>
  static Status CanEncode(const T& t, size_t* PIK_RESTRICT extension_bits,
                          size_t* PIK_RESTRICT total_bits) {
    Trace("CanEncode");
    CanEncodeVisitor visitor;
    PIK_RETURN_IF_ERROR(visitor.VisitConst(t));
    return visitor.GetSizes(extension_bits, total_bits);
  }

  template <class T>
  static Status Read(BitReader* reader, T* PIK_RESTRICT t) {
    Trace("Read");
    ReadVisitor visitor(reader);
    PIK_RETURN_IF_ERROR(visitor.Visit(t));
    return visitor.OK();
  }

  template <class T>
  static Status Write(const T& t, const size_t extension_bits,
                      size_t* PIK_RESTRICT pos, uint8_t* storage) {
    Trace("Write");
    WriteVisitor visitor(extension_bits, pos, storage);
    PIK_RETURN_IF_ERROR(visitor.VisitConst(t));
    return visitor.OK();
  }

 private:
  static void Trace(const char* op) {
#if PIK_FIELDS_TRACE
    printf("---- %s\n", op);
#endif
  }

  // A bundle can be in one of three states concerning extensions: not-begun,
  // active, ended. Bundles may be nested, so we need a stack of states.
  class ExtensionStates {
   public:
    static constexpr size_t kMaxDepth = 64;

    void Push() {
      // Initial state = not-begun.
      begun_ <<= 1;
      ended_ <<= 1;
    }

    // Clears current state; caller must check IsEnded beforehand.
    void Pop() {
      begun_ >>= 1;
      ended_ >>= 1;
    }

    // Returns true if state == active || state == ended.
    Status IsBegun() const { return (begun_ & 1) != 0; }
    // Returns true if state != not-begun && state != active.
    Status IsEnded() const { return (ended_ & 1) != 0; }

    void Begin() {
      PIK_ASSERT(!IsBegun());
      PIK_ASSERT(!IsEnded());
      begun_ += 1;
    }

    void End() {
      PIK_ASSERT(IsBegun());
      PIK_ASSERT(!IsEnded());
      ended_ += 1;
    }

   private:
    // Current state := least-significant bit of begun_ and ended_.
    uint64_t begun_ = 0;
    uint64_t ended_ = 0;
  };

  // Visitors generate Init/AllDefault/Read/Write logic for all fields. Each
  // bundle's VisitFields member function calls visitor->U32/Bytes/etc. We do
  // not overload operator() because a function name is easier to search for.

  template <class Derived>
  class VisitorBase {
   public:
    ~VisitorBase() { PIK_ASSERT(depth_ == 0); }

    // This is the only call site of T::VisitFields. Adds tracing and ensures
    // EndExtensions was called.
    template <class T>
    Status Visit(T* t) {
#if PIK_FIELDS_TRACE
      char format[10];
      snprintf(format, sizeof(format), "%%%zus%%s\n", depth_ * 2);
      printf(format, "", T::Name());
#endif

      depth_ += 1;
      PIK_ASSERT(depth_ <= ExtensionStates::kMaxDepth);
      extension_states_.Push();

      Derived* self = static_cast<Derived*>(this);
      const Status ok = t->VisitFields(self);

      if (ok) {
        // If VisitFields called BeginExtensions, must also call EndExtensions.
        PIK_ASSERT(!extension_states_.IsBegun() || extension_states_.IsEnded());
      } else {
        // Failed, undefined state: don't care whether EndExtensions was called.
      }

      extension_states_.Pop();
      PIK_ASSERT(depth_ != 0);
      depth_ -= 1;

      return ok;
    }

    // For visitors accepting a const T, need to const-cast so we can call the
    // non-const T::VisitFields. NOTE: T is not modified.
    template <class T>
    Status VisitConst(const T& t) {
      return Visit(const_cast<T*>(&t));
    }

    // Returns whether VisitFields should visit some subsequent fields.
    // "condition" is typically from prior fields, e.g. flags.
    // Overridden by InitVisitor.
    Status Conditional(bool condition) { return condition; }

    // Overridden by InitVisitor, AllDefaultVisitor and CanEncodeVisitor.
    template <class Fields>
    Status AllDefault(const Fields& fields, bool* PIK_RESTRICT all_default) {
      Derived* self = static_cast<Derived*>(this);
      self->Bool(true, all_default);
      return *all_default;
    }

    // Returns the result of visiting a nested Bundle.
    // Overridden by InitVisitor.
    template <class Fields>
    Status VisitNested(Fields* fields) {
      Derived* self = static_cast<Derived*>(this);
      return self->Visit(fields);
    }

    // Overridden by ReadVisitor.
    template <typename T>
    void SetSizeWhenReading(uint32_t size, const T* container) {
      PIK_ASSERT(container->size() == size);
    }

    // Called before any conditional visit based on "extensions".
    // Overridden by ReadVisitor, CanEncodeVisitor and WriteVisitor.
    void BeginExtensions(uint64_t* PIK_RESTRICT extensions) {
      Derived* self = static_cast<Derived*>(this);
      self->U64(0, extensions);

      extension_states_.Begin();
    }

    // Called after all extension fields (if any). Although non-extension fields
    // could be visited afterward, we prefer the convention that extension
    // fields are always the last to be visited.
    // Overridden by ReadVisitor.
    Status EndExtensions() {
      extension_states_.End();
      return true;
    }

   private:
    size_t depth_ = 0;  // for indentation.
    ExtensionStates extension_states_;
  };

  struct InitVisitor : public VisitorBase<InitVisitor> {
    void U32(const uint32_t distribution, const uint32_t default_value,
             uint32_t* PIK_RESTRICT value) {
      *value = default_value;
    }

    void U64(const uint64_t default_value, uint64_t* PIK_RESTRICT value) {
      *value = default_value;
    }

    template <typename T>
    void Enum(const uint32_t distribution, const T default_value,
              T* PIK_RESTRICT value) {
      *value = default_value;
    }

    template <typename T>
    void Orientation(
        const T default_value, T* PIK_RESTRICT value) {
      *value = default_value;
    }

    void Bool(bool default_value, bool* PIK_RESTRICT value) {
      *value = default_value;
    }

    void Bytes(const BytesEncoding unused_encoding,
               PaddedBytes* PIK_RESTRICT value) {
      value->clear();
    }

    // Always visit conditional fields to ensure they are initialized.
    Status Conditional(bool condition) { return true; }

    template <class Fields>
    Status AllDefault(const Fields& fields, bool* PIK_RESTRICT all_default) {
      // Just initialize this field and don't skip initializing others.
      Bool(true, all_default);
      return false;
    }

    template <class Fields>
    Status VisitNested(Fields* fields) {
      // Avoid re-initializing nested bundles (their ctors already called
      // Bundle::Init for their fields).
      return true;
    }
  };

  class AllDefaultVisitor : public VisitorBase<AllDefaultVisitor> {
   public:
    void U32(const uint32_t distribution, const uint32_t default_value,
             const uint32_t* PIK_RESTRICT value) {
      all_default_ &= *value == default_value;
    }

    void U64(const uint64_t default_value, const uint64_t* PIK_RESTRICT value) {
      all_default_ &= *value == default_value;
    }

    template <typename T>
    void Enum(const uint32_t distribution, const T default_value,
              const T* PIK_RESTRICT value) {
      all_default_ &= *value == default_value;
    }

    template <typename T>
    void Orientation(
        const T default_value, T* PIK_RESTRICT value) {
      all_default_ &= *value == default_value;
    }

    void Bool(bool default_value, const bool* PIK_RESTRICT value) {
      all_default_ &= *value == default_value;
    }

    void Bytes(const BytesEncoding unused_encoding,
               const PaddedBytes* PIK_RESTRICT value) {
      all_default_ &= value->empty();
    }

    template <class Fields>
    Status AllDefault(const Fields& fields, bool* PIK_RESTRICT all_default) {
      // Visit all fields so we can compute the actual all_default_ value.
      return false;
    }

    bool AllDefault() const { return all_default_; }

   private:
    bool all_default_ = true;
  };

  class ReadVisitor : public VisitorBase<ReadVisitor> {
   public:
    ReadVisitor(BitReader* reader) : reader_(reader) {}

    void U32(const uint32_t distribution, const uint32_t default_value,
             uint32_t* PIK_RESTRICT value) {
      *value = U32Coder::Read(distribution, reader_);
    }

    void U64(const uint64_t default_value, uint64_t* PIK_RESTRICT value) {
      *value = U64Coder::Read(reader_);
    }

    template <typename T>
    void Enum(const uint32_t distribution, const T default_value,
              T* PIK_RESTRICT value) {
      uint32_t bits;
      U32(distribution, static_cast<uint32_t>(default_value), &bits);
      *value = static_cast<T>(bits);
    }

    template <typename T>
    void Orientation(
        const T default_value, T* PIK_RESTRICT value) {
      *value = static_cast<T>(OrientationCoder::Read(reader_));
    }

    void Bool(bool default_value, bool* PIK_RESTRICT value) {
      uint32_t bits;
      U32(kU32RawBits + 1, default_value, &bits);
      PIK_ASSERT(bits <= 1);
      *value = bits == 1;
    }

    void Bytes(const BytesEncoding unused_encoding,
               PaddedBytes* PIK_RESTRICT value) {
      ok_ &= BytesCoder::Read(reader_, value);
    }

    template <typename T>
    void SetSizeWhenReading(uint32_t size, T* container) {
      // Sets the container size to the given size in case of reading. The size
      // must have been read from a previously visited field.
      container->resize(size);
    }

    void BeginExtensions(uint64_t* PIK_RESTRICT extensions) {
      VisitorBase<ReadVisitor>::BeginExtensions(extensions);
      if (*extensions != 0) {
        // Read the additional U64 indicating the number of extension bits
        // (more compact than sending the total size).
        extension_bits_ = U64Coder::Read(reader_);  // >= 0
        // Used by EndExtensions to skip past any _remaining_ extensions.
        pos_after_ext_size_ = reader_->BitsRead();
        PIK_ASSERT(pos_after_ext_size_ != 0);
      }
    }

    Status EndExtensions() {
      PIK_RETURN_IF_ERROR(VisitorBase<ReadVisitor>::EndExtensions());
      // Happens if extensions == 0: don't read size, done.
      if (pos_after_ext_size_ == 0) return true;

      // Skip new fields this (old?) decoder didn't know about, if any.
      const size_t bits_read = reader_->BitsRead();
      const uint64_t end = pos_after_ext_size_ + extension_bits_;
      if (bits_read > end) {
        return PIK_FAILURE("Read more extension bits than budgeted");
      }
      const size_t remaining_bits = end - bits_read;
      if (remaining_bits != 0) {
        fprintf(stderr, "Skipping %zu-bit extension(s)\n", remaining_bits);
        reader_->SkipBits(remaining_bits);
      }
      return true;
    }

    Status OK() const { return ok_; }

   private:
    bool ok_ = true;
    BitReader* const reader_;
    uint64_t extension_bits_ = 0;    // May be 0 even if extensions present.
    size_t pos_after_ext_size_ = 0;  // 0 iff extensions == 0.
  };

  class CanEncodeVisitor : public VisitorBase<CanEncodeVisitor> {
   public:
    void U32(const uint32_t distribution, const uint32_t default_value,
             const uint32_t* PIK_RESTRICT value) {
      size_t encoded_bits = 0;
      ok_ &= U32Coder::CanEncode(distribution, *value, &encoded_bits);
      encoded_bits_ += encoded_bits;
    }

    void U64(const uint64_t default_value, const uint64_t* PIK_RESTRICT value) {
      size_t encoded_bits = 0;
      ok_ &= U64Coder::CanEncode(*value, &encoded_bits);
      encoded_bits_ += encoded_bits;
    }

    template <typename T>
    void Enum(const uint32_t distribution, const T default_value,
              T* PIK_RESTRICT value) {
      uint32_t bits = static_cast<uint32_t>(*value);
      U32(distribution, static_cast<uint32_t>(default_value), &bits);
    }

    template <typename T>
    void Orientation(
        const T default_value, T* PIK_RESTRICT value) {
      size_t encoded_bits = 0;
      ok_ &= OrientationCoder::CanEncode(static_cast<uint32_t>(*value),
          &encoded_bits);
      encoded_bits_ += encoded_bits;
    }

    void Bool(const bool default_value, bool* PIK_RESTRICT value) {
      uint32_t bits = static_cast<uint32_t>(*value);
      U32(kU32RawBits + 1, default_value, &bits);
    }

    void Bytes(const BytesEncoding encoding,
               const PaddedBytes* PIK_RESTRICT value) {
      size_t encoded_bits = 0;
      ok_ &= BytesCoder::CanEncode(encoding, *value, &encoded_bits);
      encoded_bits_ += encoded_bits;
    }

    template <class Fields>
    Status AllDefault(const Fields& fields, bool* PIK_RESTRICT all_default) {
      *all_default = Bundle::AllDefault(fields);
      Bool(true, all_default);
      return *all_default;
    }

    void BeginExtensions(uint64_t* PIK_RESTRICT extensions) {
      VisitorBase<CanEncodeVisitor>::BeginExtensions(extensions);
      if (*extensions != 0) {
        PIK_ASSERT(pos_after_ext_ == 0);
        pos_after_ext_ = encoded_bits_;
        PIK_ASSERT(pos_after_ext_ != 0);  // visited "extensions"
      }
    }
    // EndExtensions = default.

    Status GetSizes(size_t* PIK_RESTRICT extension_bits,
                    size_t* PIK_RESTRICT total_bits) {
      PIK_RETURN_IF_ERROR(ok_);
      *extension_bits = 0;
      *total_bits = encoded_bits_;
      // Only if extension field was nonzero will we encode the size.
      if (pos_after_ext_ != 0) {
        PIK_ASSERT(encoded_bits_ >= pos_after_ext_);
        *extension_bits = encoded_bits_ - pos_after_ext_;
        // Also need to encode *extension_bits and bill it to *total_bits.
        size_t encoded_bits = 0;
        ok_ &= U64Coder::CanEncode(*extension_bits, &encoded_bits);
        *total_bits += encoded_bits;
      }
      return true;
    }

   private:
    bool ok_ = true;
    size_t encoded_bits_ = 0;
    // Snapshot of encoded_bits_ after visiting the extension field, but NOT
    // including the hidden "extension_bits" u64.
    uint64_t pos_after_ext_ = 0;
  };

  class WriteVisitor : public VisitorBase<WriteVisitor> {
   public:
    WriteVisitor(const size_t extension_bits, size_t* pos, uint8_t* storage)
        : extension_bits_(extension_bits), pos_(pos), storage_(storage) {}

    void U32(const uint32_t distribution, const uint32_t default_value,
             const uint32_t* PIK_RESTRICT value) {
      ok_ &= U32Coder::Write(distribution, *value, pos_, storage_);
    }

    void U64(const uint64_t default_value, const uint64_t* PIK_RESTRICT value) {
      ok_ &= U64Coder::Write(*value, pos_, storage_);
    }

    template <typename T>
    void Enum(const uint32_t distribution, const T default_value,
              T* PIK_RESTRICT value) {
      const uint32_t bits = static_cast<uint32_t>(*value);
      U32(distribution, static_cast<uint32_t>(default_value), &bits);
    }

    template <typename T>
    void Orientation(
        const T default_value, T* PIK_RESTRICT value) {
      ok_ &= OrientationCoder::Write(static_cast<uint32_t>(*value),
          pos_, storage_);
    }

    void Bool(const bool default_value, bool* PIK_RESTRICT value) {
      const uint32_t bits = static_cast<uint32_t>(*value);
      U32(kU32RawBits + 1, default_value, &bits);
    }

    void Bytes(const BytesEncoding encoding,
               const PaddedBytes* PIK_RESTRICT value) {
      ok_ &= BytesCoder::Write(encoding, *value, pos_, storage_);
    }

    void BeginExtensions(uint64_t* PIK_RESTRICT extensions) {
      VisitorBase<WriteVisitor>::BeginExtensions(extensions);
      if (*extensions == 0) {
        PIK_ASSERT(extension_bits_ == 0);
      } else {
        // NOTE: extension_bits_ can be zero if the extensions do not require
        // any additional fields.
        ok_ &= U64Coder::Write(extension_bits_, pos_, storage_);
      }
    }
    // EndExtensions = default.

    Status OK() const { return ok_; }

   private:
    const size_t extension_bits_;
    size_t* PIK_RESTRICT pos_;
    uint8_t* storage_;
    bool ok_ = true;
  };
};

}  // namespace pik

#endif  // PIK_FIELDS_H_
