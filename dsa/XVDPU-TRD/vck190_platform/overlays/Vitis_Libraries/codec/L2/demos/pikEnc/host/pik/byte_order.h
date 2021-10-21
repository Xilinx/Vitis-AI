// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_BYTE_ORDER_H_
#define PIK_BYTE_ORDER_H_

#include <stdint.h>
#include <string.h>  // memcpy
#include "pik/compiler_specific.h"

#if PIK_COMPILER_MSVC
#include <intrin.h>  // _byteswap_*
#else
#include <x86intrin.h>
#endif

#if (defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__))
#define PIK_BYTE_ORDER_LITTLE 1
#else
// This means that we don't know that the byte order is little endian, in
// this case we use endian-neutral code that works for both little- and
// big-endian.
#define PIK_BYTE_ORDER_LITTLE 0
#endif

// Returns whether the system is little-endian (least-significant byte first).
#if PIK_BYTE_ORDER_LITTLE
static constexpr bool IsLittleEndian() { return true; }
#else
static inline bool IsLittleEndian() {
  const uint32_t multibyte = 1;
  uint8_t byte;
  memcpy(&byte, &multibyte, 1);
  return byte == 1;
}
#endif

#if PIK_COMPILER_MSVC
#define PIK_BSWAP32(x) _byteswap_ulong(x)
#define PIK_BSWAP64(x) _byteswap_uint64(x)
#else
#define PIK_BSWAP32(x) __builtin_bswap32(x)
#define PIK_BSWAP64(x) __builtin_bswap64(x)
#endif

static PIK_INLINE uint32_t LoadBE16(const uint8_t* p) {
  const uint32_t byte1 = p[0];
  const uint32_t byte0 = p[1];
  return (byte1 << 8) + byte0;
}

static PIK_INLINE uint32_t LoadLE16(const uint8_t* p) {
  const uint32_t byte0 = p[0];
  const uint32_t byte1 = p[1];
  return (byte1 << 8) + byte0;
}

static PIK_INLINE uint32_t LoadBE32(const uint8_t* p) {
#if PIK_BYTE_ORDER_LITTLE
  uint32_t big;
  memcpy(&big, p, 4);
  return PIK_BSWAP32(big);
#else
  // Byte-order-independent - can't assume this machine is big endian.
  const uint32_t byte3 = p[0];
  const uint32_t byte2 = p[1];
  const uint32_t byte1 = p[2];
  const uint32_t byte0 = p[3];
  return (byte3 << 24) + (byte2 << 16) + (byte1 << 8) + byte0;
#endif
}

static PIK_INLINE uint32_t LoadLE32(const uint8_t* p) {
#if PIK_BYTE_ORDER_LITTLE
  uint32_t little;
  memcpy(&little, p, 4);
  return little;
#else
  // Byte-order-independent - can't assume this machine is big endian.
  const uint32_t byte0 = p[0];
  const uint32_t byte1 = p[1];
  const uint32_t byte2 = p[2];
  const uint32_t byte3 = p[3];
  return (byte3 << 24) + (byte2 << 16) + (byte1 << 8) + byte0;
#endif
}

static PIK_INLINE void StoreBE16(const uint32_t native, uint8_t* p) {
  p[0] = (native >> 8) & 0xFF;
  p[1] = native & 0xFF;
}

static PIK_INLINE void StoreLE16(const uint32_t native, uint8_t* p) {
  p[1] = (native >> 8) & 0xFF;
  p[0] = native & 0xFF;
}

static PIK_INLINE void StoreBE32(const uint32_t native, uint8_t* p) {
#if PIK_BYTE_ORDER_LITTLE
  const uint32_t big = PIK_BSWAP32(native);
  memcpy(p, &big, 4);
#else
  // Byte-order-independent - can't assume this machine is big endian.
  p[0] = native >> 24;
  p[1] = (native >> 16) & 0xFF;
  p[2] = (native >> 8) & 0xFF;
  p[3] = native & 0xFF;
#endif
}

static PIK_INLINE void StoreLE32(const uint32_t native, uint8_t* p) {
#if PIK_BYTE_ORDER_LITTLE
  const uint32_t little = native;
  memcpy(p, &little, 4);
#else
  // Byte-order-independent - can't assume this machine is big endian.
  p[3] = native >> 24;
  p[2] = (native >> 16) & 0xFF;
  p[1] = (native >> 8) & 0xFF;
  p[0] = native & 0xFF;
#endif
}

// Big/Little Endian order.
struct OrderBE {};
struct OrderLE {};

// Wrappers for calling from generic code.
static PIK_INLINE void Store16(OrderBE, const uint32_t native, uint8_t* p) {
  return StoreBE16(native, p);
}

static PIK_INLINE void Store16(OrderLE, const uint32_t native, uint8_t* p) {
  return StoreLE16(native, p);
}

static PIK_INLINE void Store32(OrderBE, const uint32_t native, uint8_t* p) {
  return StoreBE32(native, p);
}

static PIK_INLINE void Store32(OrderLE, const uint32_t native, uint8_t* p) {
  return StoreLE32(native, p);
}

static PIK_INLINE uint32_t Load16(OrderBE, const uint8_t* p) {
  return LoadBE16(p);
}

static PIK_INLINE uint32_t Load16(OrderLE, const uint8_t* p) {
  return LoadLE16(p);
}

static PIK_INLINE uint32_t Load32(OrderBE, const uint8_t* p) {
  return LoadBE32(p);
}

static PIK_INLINE uint32_t Load32(OrderLE, const uint8_t* p) {
  return LoadLE32(p);
}

#endif  // PIK_BYTE_ORDER_H_
