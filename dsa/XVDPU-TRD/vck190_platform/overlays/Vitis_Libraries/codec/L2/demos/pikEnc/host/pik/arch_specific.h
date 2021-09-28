// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_ARCH_SPECIFIC_H_
#define PIK_ARCH_SPECIFIC_H_

#include <stddef.h>
#include <stdint.h>

#include "pik/compiler_specific.h"

namespace pik {

#if defined(__x86_64__) || defined(_M_X64)
#define PIK_ARCH_X64 1
#else
#define PIK_ARCH_X64 0
#endif

#ifdef __aarch64__
#define PIK_ARCH_AARCH64 1
#else
#define PIK_ARCH_AARCH64 0
#endif

#if defined(__powerpc64__) || defined(_M_PPC)
#define PIK_ARCH_PPC 1
#else
#define PIK_ARCH_PPC 0
#endif

// Returns the nominal (without Turbo Boost) CPU clock rate [Hertz]. Useful for
// (roughly) characterizing the CPU speed.
double NominalClockRate();

// Returns tsc_timer frequency, useful for converting ticks to seconds. This is
// unaffected by CPU throttling ("invariant"). Thread-safe. Returns timebase
// frequency on PPC, NominalClockRate on X64, otherwise 1E9.
double InvariantTicksPerSecond();

#if PIK_ARCH_X64

// This constant avoids image.h depending on simd.h.
static constexpr size_t kMaxVectorSize = 32;  // AVX2

// Calls CPUID instruction with eax=level and ecx=count and returns the result
// in abcd array where abcd = {eax, ebx, ecx, edx} (hence the name abcd).
void Cpuid(const uint32_t level, const uint32_t count,
           uint32_t* PIK_RESTRICT abcd);

// Returns the APIC ID of the CPU on which we're currently running.
uint32_t ApicId();

#else

static constexpr size_t kMaxVectorSize = 16;

#endif  // PIK_ARCH_X64

}  // namespace pik

#endif  // PIK_ARCH_SPECIFIC_H_
