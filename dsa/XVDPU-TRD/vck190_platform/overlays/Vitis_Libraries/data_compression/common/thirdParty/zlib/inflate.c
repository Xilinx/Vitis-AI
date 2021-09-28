/* inflate.c -- zlib decompression
 * Copyright (C) 1995-2016 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/*
 * Change history:
 *
 * 1.2.beta0    24 Nov 2002
 * - First version -- complete rewrite of inflate to simplify code, avoid
 *   creation of window when not needed, minimize use of window when it is
 *   needed, make inffast.c even faster, implement gzip decoding, and to
 *   improve code readability and style over the previous zlib inflate code
 *
 * 1.2.beta1    25 Nov 2002
 * - Use pointers for available input and output checking in inffast.c
 * - Remove input and output counters in inffast.c
 * - Change inffast.c entry and loop from avail_in >= 7 to >= 6
 * - Remove unnecessary second byte pull from length extra in inffast.c
 * - Unroll direct copy to three copies per loop in inffast.c
 *
 * 1.2.beta2    4 Dec 2002
 * - Change external routine names to reduce potential conflicts
 * - Correct filename to inffixed.h for fixed tables in inflate.c
 * - Make hbuf[] unsigned char to match parameter type in inflate.c
 * - Change strm->next_out[-state->offset] to *(strm->next_out - state->offset)
 *   to avoid negation problem on Alphas (64 bit) in inflate.c
 *
 * 1.2.beta3    22 Dec 2002
 * - Add comments on state->bits assertion in inffast.c
 * - Add comments on op field in inftrees.h
 * - Fix bug in reuse of allocated window after inflateReset()
 * - Remove bit fields--back to byte structure for speed
 * - Remove distance extra == 0 check in inflate_fast()--only helps for lengths
 * - Change post-increments to pre-increments in inflate_fast(), PPC biased?
 * - Add compile time option, POSTINC, to use post-increments instead (Intel?)
 * - Make MATCH copy in inflate() much faster for when inflate_fast() not used
 * - Use local copies of stream next and avail values, as well as local bit
 *   buffer and bit count in inflate()--for speed when inflate_fast() not used
 *
 * 1.2.beta4    1 Jan 2003
 * - Split ptr - 257 statements in inflate_table() to avoid compiler warnings
 * - Move a comment on output buffer sizes from inffast.c to inflate.c
 * - Add comments in inffast.c to introduce the inflate_fast() routine
 * - Rearrange window copies in inflate_fast() for speed and simplification
 * - Unroll last copy for window match in inflate_fast()
 * - Use local copies of window variables in inflate_fast() for speed
 * - Pull out common wnext == 0 case for speed in inflate_fast()
 * - Make op and len in inflate_fast() unsigned for consistency
 * - Add FAR to lcode and dcode declarations in inflate_fast()
 * - Simplified bad distance check in inflate_fast()
 * - Added inflateBackInit(), inflateBack(), and inflateBackEnd() in new
 *   source file infback.c to provide a call-back interface to inflate for
 *   programs like gzip and unzip -- uses window as output buffer to avoid
 *   window copying
 *
 * 1.2.beta5    1 Jan 2003
 * - Improved inflateBack() interface to allow the caller to provide initial
 *   input in strm.
 * - Fixed stored blocks bug in inflateBack()
 *
 * 1.2.beta6    4 Jan 2003
 * - Added comments in inffast.c on effectiveness of POSTINC
 * - Typecasting all around to reduce compiler warnings
 * - Changed loops from while (1) or do {} while (1) to for (;;), again to
 *   make compilers happy
 * - Changed type of window in inflateBackInit() to unsigned char *
 *
 * 1.2.beta7    27 Jan 2003
 * - Changed many types to unsigned or unsigned short to avoid warnings
 * - Added inflateCopy() function
 *
 * 1.2.0        9 Mar 2003
 * - Changed inflateBack() interface to provide separate opaque descriptors
 *   for the in() and out() functions
 * - Changed inflateBack() argument and in_func typedef to swap the length
 *   and buffer address return values for the input function
 * - Check next_in and next_out for Z_NULL on entry to inflate()
 *
 * The history for versions after 1.2.0 are in ChangeLog in zlib distribution.
 */
#include "zlib.hpp"
#include "zutil.h"
#include "inftrees.h"
#include "inflate.h"
#include "inffast.h"
// Minimum size set as 1MB
#define MIN_INPUT_SIZE MEGA_BYTE
using namespace xf::compression;
#ifdef MAKEFIXED
#ifndef BUILDFIXED
#define BUILDFIXED
#endif
#endif

/* function prototypes */
local int inflateStateCheck OF((z_streamp strm));
local void fixedtables OF((struct inflate_state FAR * state));
local int updatewindow OF((z_streamp strm, const unsigned char FAR* end, unsigned copy));
#ifdef BUILDFIXED
void makefixed OF((void));
#endif
local unsigned syncsearch OF((unsigned FAR* have, const unsigned char FAR* buf, unsigned len));
#if 0
local int inflateStateCheck(strm)
z_streamp strm;
#else
local int inflateStateCheck(z_streamp strm)
#endif
{
    struct inflate_state FAR* state;
    if (strm == Z_NULL || strm->zalloc == (alloc_func)0 || strm->zfree == (free_func)0) return 1;
    state = (struct inflate_state FAR*)strm->state;
    if (state == Z_NULL || state->strm != strm || state->mode < HEAD || state->mode > SYNC) return 1;
    return 0;
}
#if 0
int ZEXPORT inflateResetKeep(strm)
z_streamp strm;
#else
int ZEXPORT inflateResetKeep(z_streamp strm)
#endif
{
    struct inflate_state FAR* state;

    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
    strm->total_in = strm->total_out = state->total = 0;
    strm->msg = Z_NULL;
    if (state->wrap) /* to support ill-conceived Java test suite */
        strm->adler = state->wrap & 1;
    state->mode = HEAD;
    state->last = 0;
    state->havedict = 0;
    state->dmax = 32768U;
    state->head = Z_NULL;
    state->hold = 0;
    state->bits = 0;
    state->lencode = state->distcode = state->next = state->codes;
    state->sane = 1;
    state->back = -1;
    Tracev((stderr, "inflate: reset\n"));
    return Z_OK;
}

#if 0
int ZEXPORT inflateReset(strm)
z_streamp strm;
#else
int ZEXPORT inflateReset(z_streamp strm)
#endif
{
    struct inflate_state FAR* state;

    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
    state->wsize = 0;
    state->whave = 0;
    state->wnext = 0;
    return inflateResetKeep(strm);
}

#if 0
int ZEXPORT inflateReset2(strm, windowBits)
z_streamp strm;
int windowBits;
#else
int ZEXPORT inflateReset2(z_streamp strm, int windowBits)
#endif
{
    int wrap;
    struct inflate_state FAR* state;

    /* get the state */
    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;

    /* extract wrap request from windowBits parameter */
    if (windowBits < 0) {
        wrap = 0;
        windowBits = -windowBits;
    } else {
        wrap = (windowBits >> 4) + 5;
    }

    /* set number of window bits, free window if different */
    if (windowBits && (windowBits < 8 || windowBits > 15)) return Z_STREAM_ERROR;
    if (state->window != Z_NULL && state->wbits != (unsigned)windowBits) {
        ZFREE(strm, state->window);
        state->window = Z_NULL;
    }

    /* update state and reset the rest of it */
    state->wrap = wrap;
    state->wbits = (unsigned)windowBits;
    return inflateReset(strm);
}
#if 0
int ZEXPORT inflateInit2_(strm, windowBits, version, stream_size)
z_streamp strm;
int windowBits;
const char *version;
int stream_size;
#else
int ZEXPORT inflateInit2_(z_streamp strm, int windowBits, const char* version, int stream_size)
#endif
{
    int ret;
    struct inflate_state FAR* state;

    if (version == Z_NULL || version[0] != ZLIB_VERSION[0] || stream_size != (int)(sizeof(z_stream)))
        return Z_VERSION_ERROR;
    if (strm == Z_NULL) return Z_STREAM_ERROR;
    strm->msg = Z_NULL; /* in case we return an error */
    if (strm->zalloc == (alloc_func)0) {
#ifdef Z_SOLO
        return Z_STREAM_ERROR;
#else
        strm->zalloc = zcalloc;
        strm->opaque = (voidpf)0;
#endif
    }
    if (strm->zfree == (free_func)0)
#ifdef Z_SOLO
        return Z_STREAM_ERROR;
#else
        strm->zfree = zcfree;
#endif
    state = (struct inflate_state FAR*)ZALLOC(strm, 1, sizeof(struct inflate_state));
    if (state == Z_NULL) return Z_MEM_ERROR;
    Tracev((stderr, "inflate: allocated\n"));
    strm->state = (struct internal_state FAR*)state;
    state->strm = strm;
    state->window = Z_NULL;
    state->mode = HEAD; /* to pass state test in inflateReset2() */
    ret = inflateReset2(strm, windowBits);
    if (ret != Z_OK) {
        ZFREE(strm, state);
        strm->state = Z_NULL;
    }
    return ret;
}
#if 0
int ZEXPORT inflateInit_(strm, version, stream_size)
z_streamp strm;
const char *version;
int stream_size;
#else
int ZEXPORT inflateInit_(z_streamp strm, const char* version, int stream_size)
#endif
{ return inflateInit2_(strm, DEF_WBITS, version, stream_size); }

#if 0
int ZEXPORT inflatePrime(strm, bits, value)
z_streamp strm;
int bits;
int value;
#else
int ZEXPORT inflatePrime(z_streamp strm, int bits, int value)
#endif
{
    struct inflate_state FAR* state;

    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
    if (bits < 0) {
        state->hold = 0;
        state->bits = 0;
        return Z_OK;
    }
    if (bits > 16 || state->bits + (uInt)bits > 32) return Z_STREAM_ERROR;
    value &= (1L << bits) - 1;
    state->hold += (unsigned)value << state->bits;
    state->bits += (uInt)bits;
    return Z_OK;
}

/*
   Return state with length and distance decoding tables and index sizes set to
   fixed code decoding.  Normally this returns fixed tables from inffixed.h.
   If BUILDFIXED is defined, then instead this routine builds the tables the
   first time it's called, and returns those tables the first time and
   thereafter.  This reduces the size of the code by about 2K bytes, in
   exchange for a little execution time.  However, BUILDFIXED should not be
   used for threaded applications, since the rewriting of the tables and virgin
   may not be thread-safe.
 */
#if 0
local void fixedtables(state)
struct inflate_state FAR *state;
#else
local void fixedtables(struct inflate_state FAR* state)
#endif
{
#if 0 
    static int virgin = 1;
    static code *lenfix, *distfix;
    static code fixed[544];

    /* build fixed huffman tables if first call (may not be thread safe) */
    if (virgin) {
        unsigned sym, bits;
        static code *next;

        /* literal/length table */
        sym = 0;
        while (sym < 144) state->lens[sym++] = 8;
        while (sym < 256) state->lens[sym++] = 9;
        while (sym < 280) state->lens[sym++] = 7;
        while (sym < 288) state->lens[sym++] = 8;
        next = fixed;
        lenfix = next;
        bits = 9;
        inflate_table(LENS, state->lens, 288, &(next), &(bits), state->work);

        /* distance table */
        sym = 0;
        while (sym < 32) state->lens[sym++] = 5;
        distfix = next;
        bits = 5;
        inflate_table(DISTS, state->lens, 32, &(next), &(bits), state->work);

        /* do this just once */
        virgin = 0;
    }
#else /* !BUILDFIXED */
#include "inffixed.h"
#endif /* BUILDFIXED */
    state->lencode = lenfix;
    state->lenbits = 9;
    state->distcode = distfix;
    state->distbits = 5;
}

#ifdef MAKEFIXED
#include <stdio.h>

/*
   Write out the inffixed.h that is #include'd above.  Defining MAKEFIXED also
   defines BUILDFIXED, so the tables are built on the fly.  makefixed() writes
   those tables to stdout, which would be piped to inffixed.h.  A small program
   can simply call makefixed to do this:

    void makefixed(void);

    int main(void)
    {
        makefixed();
        return 0;
    }

   Then that can be linked with zlib built with MAKEFIXED defined and run:

    a.out > inffixed.h
 */
void makefixed() {
    unsigned low, size;
    struct inflate_state state;

    fixedtables(&state);
    puts("    /* inffixed.h -- table for decoding fixed codes");
    puts("     * Generated automatically by makefixed().");
    puts("     */");
    puts("");
    puts("    /* WARNING: this file should *not* be used by applications.");
    puts("       It is part of the implementation of this library and is");
    puts("       subject to change. Applications should only use zlib.h.");
    puts("     */");
    puts("");
    size = 1U << 9;
    printf("    static const code lenfix[%u] = {", size);
    low = 0;
    for (;;) {
        if ((low % 7) == 0) printf("\n        ");
        printf("{%u,%u,%d}", (low & 127) == 99 ? 64 : state.lencode[low].op, state.lencode[low].bits,
               state.lencode[low].val);
        if (++low == size) break;
        putchar(',');
    }
    puts("\n    };");
    size = 1U << 5;
    printf("\n    static const code distfix[%u] = {", size);
    low = 0;
    for (;;) {
        if ((low % 6) == 0) printf("\n        ");
        printf("{%u,%u,%d}", state.distcode[low].op, state.distcode[low].bits, state.distcode[low].val);
        if (++low == size) break;
        putchar(',');
    }
    puts("\n    };");
}
#endif /* MAKEFIXED */

/*
   Update the window with the last wsize (normally 32K) bytes written before
   returning.  If window does not exist yet, create it.  This is only called
   when a window is already in use, or when output has been written during this
   inflate call, but the end of the deflate stream has not been reached yet.
   It is also called to create a window for dictionary data when a dictionary
   is loaded.

   Providing output buffers larger than 32K to inflate() should provide a speed
   advantage, since only the last 32K of output is copied to the sliding window
   upon return from inflate(), and since all distances after the first 32K of
   output will fall in the output data, making match copies simpler and faster.
   The advantage may be dependent on the size of the processor's data caches.
 */
#if 0
local int updatewindow(strm, end, copy)
z_streamp strm;
const Bytef *end;
unsigned copy;
#else
local int updatewindow(z_streamp strm, const Bytef* end, unsigned copy)
#endif
{
    struct inflate_state FAR* state;
    unsigned dist;

    state = (struct inflate_state FAR*)strm->state;

    /* if it hasn't been done already, allocate space for the window */
    if (state->window == Z_NULL) {
        state->window = (unsigned char FAR*)ZALLOC(strm, 1U << state->wbits, sizeof(unsigned char));
        if (state->window == Z_NULL) return 1;
    }

    /* if window not in use yet, initialize */
    if (state->wsize == 0) {
        state->wsize = 1U << state->wbits;
        state->wnext = 0;
        state->whave = 0;
    }

    /* copy state->wsize or less output bytes into the circular window */
    if (copy >= state->wsize) {
        zmemcpy(state->window, end - state->wsize, state->wsize);
        state->wnext = 0;
        state->whave = state->wsize;
    } else {
        dist = state->wsize - state->wnext;
        if (dist > copy) dist = copy;
        zmemcpy(state->window + state->wnext, end - copy, dist);
        copy -= dist;
        if (copy) {
            zmemcpy(state->window, end - copy, copy);
            state->wnext = copy;
            state->whave = state->wsize;
        } else {
            state->wnext += dist;
            if (state->wnext == state->wsize) state->wnext = 0;
            if (state->whave < state->wsize) state->whave += dist;
        }
    }
    return 0;
}

/* Macros for inflate(): */

/* check function to use adler32() for zlib or crc32() for gzip */
#ifdef GUNZIP
#define UPDATE(check, buf, len) (state->flags ? crc32(check, buf, len) : adler32(check, buf, len))
#else
#define UPDATE(check, buf, len) adler32(check, buf, len)
#endif

/* check macros for header crc */
#ifdef GUNZIP
#define CRC2(check, word)                       \
    do {                                        \
        hbuf[0] = (unsigned char)(word);        \
        hbuf[1] = (unsigned char)((word) >> 8); \
        check = crc32(check, hbuf, 2);          \
    } while (0)

#define CRC4(check, word)                        \
    do {                                         \
        hbuf[0] = (unsigned char)(word);         \
        hbuf[1] = (unsigned char)((word) >> 8);  \
        hbuf[2] = (unsigned char)((word) >> 16); \
        hbuf[3] = (unsigned char)((word) >> 24); \
        check = crc32(check, hbuf, 4);           \
    } while (0)
#endif

/* Load registers with state in inflate() for speed */
#define LOAD()                  \
    do {                        \
        put = strm->next_out;   \
        left = strm->avail_out; \
        next = strm->next_in;   \
        have = strm->avail_in;  \
        hold = state->hold;     \
        bits = state->bits;     \
    } while (0)

/* Restore state from registers in inflate() */
#define RESTORE()               \
    do {                        \
        strm->next_out = put;   \
        strm->avail_out = left; \
        strm->next_in = next;   \
        strm->avail_in = have;  \
        state->hold = hold;     \
        state->bits = bits;     \
    } while (0)

/* Clear the input bit accumulator */
#define INITBITS() \
    do {           \
        hold = 0;  \
        bits = 0;  \
    } while (0)

/* Get a byte of input into the bit accumulator, or return from inflate()
   if there is no input available. */
#define PULLBYTE()                                \
    do {                                          \
        if (have == 0) goto inf_leave;            \
        have--;                                   \
        hold += (unsigned long)(*next++) << bits; \
        bits += 8;                                \
    } while (0)

/* Assure that there are at least n bits in the bit accumulator.  If there is
   not enough available input to do that, then return from inflate(). */
#define NEEDBITS(n)                              \
    do {                                         \
        while (bits < (unsigned)(n)) PULLBYTE(); \
    } while (0)

/* Return the low n bits of the bit accumulator (n < 16) */
#define BITS(n) ((unsigned)hold & ((1U << (n)) - 1))

/* Remove n bits from the bit accumulator */
#define DROPBITS(n)            \
    do {                       \
        hold >>= (n);          \
        bits -= (unsigned)(n); \
    } while (0)

/* Remove zero to seven bits as needed to go to a byte boundary */
#define BYTEBITS()         \
    do {                   \
        hold >>= bits & 7; \
        bits -= bits & 7;  \
    } while (0)

/*
   inflate() uses a state machine to process as much input data and generate as
   much output data as possible before returning.  The state machine is
   structured roughly as follows:

    for (;;) switch (state) {
    ...
    case STATEn:
        if (not enough input data or output space to make progress)
            return;
        ... make progress ...
        state = STATEm;
        break;
    ...
    }

   so when inflate() is called again, the same case is attempted again, and
   if the appropriate resources are provided, the machine proceeds to the
   next state.  The NEEDBITS() macro is usually the way the state evaluates
   whether it can proceed or should return.  NEEDBITS() does the return if
   the requested bits are not available.  The typical use of the BITS macros
   is:

        NEEDBITS(n);
        ... do something with BITS(n) ...
        DROPBITS(n);

   where NEEDBITS(n) either returns from inflate() if there isn't enough
   input left to load n bits into the accumulator, or it continues.  BITS(n)
   gives the low n bits in the accumulator.  When done, DROPBITS(n) drops
   the low n bits off the accumulator.  INITBITS() clears the accumulator
   and sets the number of available bits to zero.  BYTEBITS() discards just
   enough bits to put the accumulator on a byte boundary.  After BYTEBITS()
   and a NEEDBITS(8), then BITS(8) would return the next byte in the stream.

   NEEDBITS(n) uses PULLBYTE() to get an available byte of input, or to return
   if there is no input available.  The decoding of variable length codes uses
   PULLBYTE() directly in order to pull just enough bytes to decode the next
   code, and no more.

   Some states loop until they get enough input, making sure that enough
   state information is maintained to continue the loop where it left off
   if NEEDBITS() returns in the loop.  For example, want, need, and keep
   would all have to actually be part of the saved state in case NEEDBITS()
   returns:

    case STATEw:
        while (want < need) {
            NEEDBITS(n);
            keep[want++] = BITS(n);
            DROPBITS(n);
        }
        state = STATEx;
    case STATEx:

   As shown above, if the next state is also the next case, then the break
   is omitted.

   A state may also return if there is not enough output space available to
   complete that state.  Those states are copying stored data, writing a
   literal byte, and copying a matching string.

   When returning, a "goto inf_leave" is used to update the total counters,
   update the check value, and determine whether any progress has been made
   during that inflate() call in order to return the proper return code.
   Progress is defined as a change in either strm->avail_in or strm->avail_out.
   When there is a window, goto inf_leave will update the window with the last
   output written.  If a goto inf_leave occurs in the middle of decompression
   and there is no window currently, goto inf_leave will create one and copy
   output to the window for the next call of inflate().

   In this implementation, the flush parameter of inflate() only affects the
   return code (per zlib.h).  inflate() always writes as much as possible to
   strm->next_out, given the space available and the provided input--the effect
   documented in zlib.h of Z_SYNC_FLUSH.  Furthermore, inflate() always defers
   the allocation of and copying into a sliding window until necessary, which
   provides the effect documented in zlib.h for Z_FINISH when the entire input
   stream available.  So the only thing the flush parameter actually does is:
   when flush is set to Z_FINISH, inflate() cannot return Z_OK.  Instead it
   will return Z_BUF_ERROR if it has not reached the end of the stream.
 */
#include <stdio.h>

extern "C" {

#if 0
int ZEXPORT inflate(strm, flush)
z_streamp strm;
int flush;
#else
int ZEXPORT inflate(z_streamp strm, int flush)
#endif
{
    std::string u50_xclbin("/opt/xilinx/zlib/u50_gen3x16_xdma_201920_3.xclbin");
    std::string xrt_ini("/opt/xilinx/zlib/xrt.ini");
    const uint8_t c_max_cr = 0;
    bool use_cpu_sol = false;
    bool use_fpga_sol = false;
    char* xclbin = getenv("XILINX_LIBZ_XCLBIN");
    if (xclbin && *xclbin) u50_xclbin = xclbin;
    char* xrtent = getenv("XRT_INI_PATH");
    if (xrtent && *xrtent) xrt_ini = xrtent;
    char* cu_id = getenv("XILINX_CU_ID");

    uint8_t no_cu = 0;
    if (cu_id) {
        no_cu = atoi(cu_id);
        if (no_cu > D_COMPUTE_UNIT) no_cu %= D_COMPUTE_UNIT;
    }

    std::ifstream xrt_ini_file(xrt_ini.c_str(), std::ifstream::binary);
    if (xrt_ini_file.is_open()) {
        // Path to XRT ini file
        setenv("XRT_INI_PATH", xrt_ini.c_str(), 1);
        xrt_ini_file.close();
    }

    std::ifstream bin_file(u50_xclbin.c_str(), std::ifstream::binary);
    if (bin_file.is_open()) {
        use_fpga_sol = true;
        bin_file.close();
    } else {
#if (VERBOSE_LEVEL >= 1)
        std::cout << "Unable to open binary file " << std::endl;
#endif
        use_cpu_sol = true;
    }

    char* min_size = getenv("MIN_INPUT_SIZE");
    uint32_t small_size = MIN_INPUT_SIZE;
    if (min_size != NULL) {
        small_size = atoi(min_size);
    }

    // Check input size if its less than
    // MIN_INPUT_SIZE use SW flow
    uint64_t input_size = strm->avail_in;
    if (input_size < small_size) {
#if (VERBOSE_LEVEL >= 1)
        std::cout << "Input Size is less than MIN_INPUT_SIZE";
        std::cout << " Falling back to SW Solution " << std::endl;
#endif
        use_cpu_sol = true;
        use_fpga_sol = false;
    }

    if (use_fpga_sol) {
        // Arg 1: xclbin
        // xilinx ZLIB object
        xfZlib* xlz = nullptr;
        bool flag = false;
        do {
            xlz = new xfZlib(u50_xclbin.c_str(), c_max_cr, DECOMP_ONLY, 0, 0, FULL);
            int err_code = xlz->error_code();
            if (err_code && (err_code != c_clOutOfResource)) {
#if (VERBOSE_LEVEL >= 1)
                std::cout << "Failed to use FPGA for Decompression, ";
                std::cout << " switching to SW solution \n" << std::endl;
#endif
                use_cpu_sol = true;
                break;
            } else {
                uint8_t* input = strm->next_in;
                uint8_t* output = strm->next_out;
                int debytes = 0;
                if (cu_id) {
                    // Zlib decompression -- Set CU ID
                    debytes = xlz->decompress((uint8_t*)input, (uint8_t*)output, input_size, no_cu);
                } else {
                    // Zlib decompression -- Random CU generation
                    debytes = xlz->decompress_buffer((uint8_t*)input, (uint8_t*)output, input_size);
                }
                int err_code = xlz->error_code();
                delete xlz;
                if (debytes == 0) {
                    if (err_code == c_headermismatch) {
#if (VERBOSE_LEVEL >= 1)
                        std::cout << "Zlib Data Format Check Failed" << std::endl;
#endif
                        use_cpu_sol = true;
                        break;
                    }

                    if (err_code == c_clOutOfResource) {
#if (VERBOSE_LEVEL >= 1)
                        std::cout << "Failed to create device buffer, Retrying . . ." << std::endl;
#endif
                    }
                    flag = true;
                } else {
                    strm->total_out = debytes;
                    strm->next_out = output;
                    return 1;
                }
            }
        } while (flag);
    }

    if (use_cpu_sol) {
#if (VERBOSE_LEVEL >= 1)
        printf("CPU Solution \n");
#endif
        // printf("In inflate integrated \n");
        struct inflate_state FAR* state;
        z_const unsigned char FAR* next;        /* next input */
        unsigned char FAR* put;                 /* next output */
        unsigned int have, left;                /* available input and output */
        unsigned long hold;                     /* bit buffer */
        unsigned bits;                          /* bits in bit buffer */
        unsigned in, out;                       /* save starting available input and output */
        unsigned copy;                          /* number of stored or match bytes to copy */
        unsigned char FAR* from;                /* where to copy match bytes from */
        code here;                              /* current decoding table entry */
        code last;                              /* parent table entry */
        unsigned len;                           /* length to copy for repeats, bits to drop */
        int ret;                                /* return code */
        static const unsigned short order[19] = /* permutation of code lengths */
            {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};
        static int flag_kal;

        state = (struct inflate_state FAR*)strm->state;

        LOAD();

        in = have;
        out = left;
        ret = Z_OK;

        for (;;) {
            switch (state->mode) {
                case HEAD:
                    if (state->wrap == 0) {
                        state->mode = TYPEDO;
                        break;
                    }

                    NEEDBITS(16);

                    if (((BITS(8) << 8) + (hold >> 8)) % 31) {
                        strm->msg = (char*)"incorrect header check";
                        state->mode = BAD;
                        break;
                    }

                    if (BITS(4) != Z_DEFLATED) {
                        strm->msg = (char*)"unknown compression method";
                        state->mode = BAD;
                        // printf("Un known method \n");
                        break;
                    }

                    DROPBITS(4);
                    len = BITS(4) + 8;

                    if (state->wbits == 0) state->wbits = len;

                    if (len > 15 || len > state->wbits) {
                        strm->msg = (char*)"invalid window size";
                        state->mode = BAD;
                        // printf("Invalid window size \n");
                        break;
                    }

                    state->dmax = 1U << len;

                    Tracev((stderr, "inflate:   zlib header ok\n"));

                    strm->adler = state->check = adler32(0L, Z_NULL, 0);
                    state->mode = hold & 0x200 ? DICTID : TYPE;

                    INITBITS();
                    break;
                case TYPE:
                    if (flush == Z_BLOCK || flush == Z_TREES) {
                        goto inf_leave;
                    }
                case TYPEDO:
                    if (state->last) {
                        // printf("TYPEDO \n");
                        BYTEBITS();
                        state->mode = CHECK;
                        break;
                    }

                    NEEDBITS(3);
                    state->last = BITS(1);
                    DROPBITS(1);

                    switch (BITS(2)) {
                        case 0: /* stored block */
                            Tracev((stderr, "inflate:     stored block%s\n", state->last ? " (last)" : ""));
                            state->mode = STORED;
                            break;
                        case 1: /* fixed block */
                            fixedtables(state);
                            Tracev((stderr, "inflate:     fixed codes block%s\n", state->last ? " (last)" : ""));
                            state->mode = LEN_; /* decode codes */
                            if (flush == Z_TREES) {
                                DROPBITS(2);
                                goto inf_leave;
                            }
                            break;
                        case 2: /* dynamic block */
                            Tracev((stderr, "inflate:     dynamic codes block%s\n", state->last ? " (last)" : ""));
                            state->mode = TABLE;
                            break;
                        case 3:
                            // printf("Invalid block \n");
                            strm->msg = (char*)"invalid block type";
                            state->mode = BAD;
                    }

                    DROPBITS(2);
                    break;

                case STORED:
                    BYTEBITS(); /* go to byte boundary */
                    NEEDBITS(32);
                    if ((hold & 0xffff) != ((hold >> 16) ^ 0xffff)) {
                        strm->msg = (char*)"invalid stored block lengths";
                        state->mode = BAD;
                        // printf("Invalid storedblock \n");
                        break;
                    }
                    state->length = (unsigned)hold & 0xffff;
                    Tracev((stderr, "inflate:       stored length %u\n", state->length));
                    INITBITS();
                    state->mode = COPY_;
                    if (flush == Z_TREES) goto inf_leave;

                case COPY_:
                    state->mode = COPY;
                case COPY:
                    copy = state->length;
                    // printf("left %d \n", left);
                    if (copy) {
                        if (copy > have) copy = have;
                        if (copy > left) copy = left;
                        if (copy == 0) goto inf_leave;
                        zmemcpy(put, next, copy);
                        have -= copy;
                        next += copy;
                        left -= copy;
                        put += copy;
                        state->length -= copy;
                        break;
                    }
                    Tracev((stderr, "inflate:       stored end\n"));
                    state->mode = TYPE;
                    break;
                case TABLE:
                    NEEDBITS(14);
                    state->nlen = BITS(5) + 257;
                    DROPBITS(5);
                    state->ndist = BITS(5) + 1;
                    DROPBITS(5);
                    state->ncode = BITS(4) + 4;
                    DROPBITS(4);
                    Tracev((stderr, "inflate:       table sizes ok\n"));
                    state->have = 0;
                    state->mode = LENLENS;
                case LENLENS:
                    while (state->have < state->ncode) {
                        // printf("In LENLENS \n");
                        NEEDBITS(3);
                        state->lens[order[state->have++]] = (unsigned short)BITS(3);
                        DROPBITS(3);
                    }
                    while (state->have < 19) state->lens[order[state->have++]] = 0;
                    state->next = state->codes;
                    state->lencode = (const code FAR*)(state->next);
                    state->lenbits = 7;
                    ret = inflate_table(CODES, state->lens, 19, &(state->next), &(state->lenbits), state->work);
                    // printf("ret inflate_table %d\n", ret);
                    if (ret) {
                        strm->msg = (char*)"invalid code lengths set";
                        state->mode = BAD;
                        // printf("Invalid code length set \n");
                        break;
                    }
                    Tracev((stderr, "inflate:       code lengths ok\n"));
                    state->have = 0;
                    state->mode = CODELENS;
                case CODELENS:
                    while (state->have < state->nlen + state->ndist) {
                        for (;;) {
                            here = state->lencode[BITS(state->lenbits)];
                            if ((unsigned)(here.bits) <= bits) break;
                            PULLBYTE();
                        }

                        if (here.val < 16) {
                            DROPBITS(here.bits);
                            state->lens[state->have++] = here.val;
                        } else {
                            if (here.val == 16) {
                                NEEDBITS(here.bits + 2);
                                DROPBITS(here.bits);
                                if (state->have == 0) {
                                    strm->msg = (char*)"invalid bit length repeat";
                                    state->mode = BAD;
                                    // printf("Invalid bit length repeat\n");
                                    break;
                                }
                                len = state->lens[state->have - 1];
                                copy = 3 + BITS(2);
                                DROPBITS(2);
                            } else if (here.val == 17) {
                                NEEDBITS(here.bits + 3);
                                DROPBITS(here.bits);
                                len = 0;
                                copy = 3 + BITS(3);
                                DROPBITS(3);
                            } else {
                                NEEDBITS(here.bits + 7);
                                DROPBITS(here.bits);
                                len = 0;
                                copy = 11 + BITS(7);
                                DROPBITS(7);
                            }

                            if (state->have + copy > state->nlen + state->ndist) {
                                strm->msg = (char*)"invalid bit length repeat";
                                state->mode = BAD;
                                // printf("Invalid bit length repeat \n");
                                break;
                            }
                            while (copy--) state->lens[state->have++] = (unsigned short)len;
                        }
                    } // End of while

                    /* handle error breaks in while */
                    if (state->mode == BAD) break;

                    // printf("state->lens[256] %d \n", state->lens[256]);
                    /* check for end-of-block code (better have one) */
                    if (state->lens[256] == 0) {
                        strm->msg = (char*)"invalid code -- missing end-of-block";
                        // printf("missing end of block \n");
                        state->mode = BAD;
                        break;
                    }

                    /* build code tables -- note: do not change the lenbits or distbits
                       values here (9 and 6) without reading the comments in inftrees.h
                       concerning the ENOUGH constants, which depend on those values */
                    state->next = state->codes;
                    state->lencode = (const code FAR*)(state->next);
                    state->lenbits = 9;
                    ret = inflate_table(LENS, state->lens, state->nlen, &(state->next), &(state->lenbits), state->work);
                    // printf("ret %d build code tables\n", ret);
                    if (ret) {
                        strm->msg = (char*)"invalid literal/lengths set";
                        state->mode = BAD;
                        // printf("invalid literal/length set \n");
                        break;
                    }
                    state->distcode = (const code FAR*)(state->next);
                    state->distbits = 6;
                    ret = inflate_table(DISTS, state->lens + state->nlen, state->ndist, &(state->next),
                                        &(state->distbits), state->work);
                    // printf("ret %d build distance tables\n", ret);
                    if (ret) {
                        strm->msg = (char*)"invalid distances set";
                        state->mode = BAD;
                        // printf("invalid distance set \n");
                        break;
                    }
                    Tracev((stderr, "inflate:       codes ok\n"));
                    state->mode = LEN_;
                    if (flush == Z_TREES) goto inf_leave;
                case LEN_:
                    state->mode = LEN;
                case LEN:
                    // printf("In LEN - have %d left %d\n", have, left);
                    // printf("LEN before if state->mode %d \n", state->mode);
                    if (have >= 6 && left >= 258) {
                        RESTORE();
                        inflate_fast(strm, out);
                        LOAD();
                        if (state->mode == TYPE) state->back = -1;
                        break;
                    }
                case CHECK:
                    // printf("CHECK stage \n");
                    state->mode = DONE;
                case DONE:
                    // printf("Its done \n");
                    ret = Z_STREAM_END;
                    goto inf_leave;
                case BAD:
                    ret = Z_DATA_ERROR;
                    goto inf_leave;
                case MEM:
                    return Z_MEM_ERROR;
                case SYNC:
                default:
                    return Z_STREAM_ERROR;
            }
        }
    /*
       Return from inflate(), updating the total counts and the check value.
       If there was no progress during the inflate() call, return a buffer
       error.  Call updatewindow() to create and/or update the window state.
       Note: a memory error from inflate() is non-recoverable.
     */
    inf_leave:
        RESTORE();
        if (state->wsize || (out != strm->avail_out && state->mode < BAD && (state->mode < CHECK || flush != Z_FINISH)))
            if (updatewindow(strm, strm->next_out, out - strm->avail_out)) {
                state->mode = MEM;
                return Z_MEM_ERROR;
            }
        // printf("in %d strm->avail_in %d \n", in, strm->avail_in);

        in -= strm->avail_in;
        out -= strm->avail_out;
        strm->total_in += in;
        strm->total_out += out;
        state->total += out;
        if ((state->wrap & 4) && out) strm->adler = state->check = UPDATE(state->check, strm->next_out - out, out);
        strm->data_type = (int)state->bits + (state->last ? 64 : 0) + (state->mode == TYPE ? 128 : 0) +
                          (state->mode == LEN_ || state->mode == COPY_ ? 256 : 0);
        // printf("in %d out %d flush %d ret %d \n", in, out, flush, ret);
        if (((in == 0 && out == 0) || flush == Z_FINISH) && ret == Z_OK) {
            ret = Z_BUF_ERROR;
        }
        return ret;
    }
}
}
#if 0
int ZEXPORT inflateEnd(strm)
z_streamp strm;
#else
int ZEXPORT inflateEnd(z_streamp strm)
#endif
{
    struct inflate_state FAR* state;
    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
    if (state->window != Z_NULL) ZFREE(strm, state->window);
    ZFREE(strm, strm->state);
    strm->state = Z_NULL;
    Tracev((stderr, "inflate: end\n"));
    return Z_OK;
}
#if 0
int ZEXPORT inflateGetDictionary(strm, dictionary, dictLength)
z_streamp strm;
Bytef *dictionary;
uInt *dictLength;
#else
int ZEXPORT inflateGetDictionary(z_streamp strm, Bytef* dictionary, uInt* dictLength)
#endif
{
    struct inflate_state FAR* state;

    /* check state */
    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;

    /* copy dictionary */
    if (state->whave && dictionary != Z_NULL) {
        zmemcpy(dictionary, state->window + state->wnext, state->whave - state->wnext);
        zmemcpy(dictionary + state->whave - state->wnext, state->window, state->wnext);
    }
    if (dictLength != Z_NULL) *dictLength = state->whave;
    return Z_OK;
}

#if 0
int ZEXPORT inflateSetDictionary(strm, dictionary, dictLength)
z_streamp strm;
const Bytef *dictionary;
uInt dictLength;
#else
int ZEXPORT inflateSetDictionary(z_streamp strm, const Bytef* dictionary, uInt dictLength)
#endif
{
    struct inflate_state FAR* state;
    unsigned long dictid;
    int ret;

    /* check state */
    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
    if (state->wrap != 0 && state->mode != DICT) return Z_STREAM_ERROR;

    /* check for correct dictionary identifier */
    if (state->mode == DICT) {
        dictid = adler32(0L, Z_NULL, 0);
        dictid = adler32(dictid, dictionary, dictLength);
        if (dictid != state->check) return Z_DATA_ERROR;
    }

    /* copy dictionary to window using updatewindow(), which will amend the
       existing dictionary if appropriate */
    ret = updatewindow(strm, dictionary + dictLength, dictLength);
    if (ret) {
        state->mode = MEM;
        return Z_MEM_ERROR;
    }
    state->havedict = 1;
    Tracev((stderr, "inflate:   dictionary set\n"));
    return Z_OK;
}

#if 0
int ZEXPORT inflateGetHeader(strm, head)
z_streamp strm;
gz_headerp head;
#else
int ZEXPORT inflateGetHeader(z_streamp strm, gz_headerp head)
#endif
{
    struct inflate_state FAR* state;

    /* check state */
    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
    if ((state->wrap & 2) == 0) return Z_STREAM_ERROR;

    /* save header structure */
    state->head = head;
    head->done = 0;
    return Z_OK;
}

/*
   Search buf[0..len-1] for the pattern: 0, 0, 0xff, 0xff.  Return when found
   or when out of input.  When called, *have is the number of pattern bytes
   found in order so far, in 0..3.  On return *have is updated to the new
   state.  If on return *have equals four, then the pattern was found and the
   return value is how many bytes were read including the last byte of the
   pattern.  If *have is less than four, then the pattern has not been found
   yet and the return value is len.  In the latter case, syncsearch() can be
   called again with more data and the *have state.  *have is initialized to
   zero for the first call.
 */
#if 0
local unsigned syncsearch(have, buf, len)
unsigned FAR *have;
const unsigned char FAR *buf;
unsigned len;
#else
local unsigned syncsearch(unsigned FAR* have, const unsigned char FAR* buf, unsigned len)
#endif
{
    unsigned got;
    unsigned next;

    got = *have;
    next = 0;
    while (next < len && got < 4) {
        if ((int)(buf[next]) == (got < 2 ? 0 : 0xff))
            got++;
        else if (buf[next])
            got = 0;
        else
            got = 4 - got;
        next++;
    }
    *have = got;
    return next;
}
#if 0
int ZEXPORT inflateSync(strm)
z_streamp strm;
#else
int ZEXPORT inflateSync(z_streamp strm)
#endif
{
    unsigned len;          /* number of bytes to look at or looked at */
    unsigned long in, out; /* temporary to save total_in and total_out */
    unsigned char buf[4];  /* to restore bit buffer to byte string */
    struct inflate_state FAR* state;

    /* check parameters */
    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
    if (strm->avail_in == 0 && state->bits < 8) return Z_BUF_ERROR;

    /* if first time, start search in bit buffer */
    if (state->mode != SYNC) {
        state->mode = SYNC;
        state->hold <<= state->bits & 7;
        state->bits -= state->bits & 7;
        len = 0;
        while (state->bits >= 8) {
            buf[len++] = (unsigned char)(state->hold);
            state->hold >>= 8;
            state->bits -= 8;
        }
        state->have = 0;
        syncsearch(&(state->have), buf, len);
    }

    /* search available input */
    len = syncsearch(&(state->have), strm->next_in, strm->avail_in);
    strm->avail_in -= len;
    strm->next_in += len;
    strm->total_in += len;

    /* return no joy or set up to restart inflate() on a new block */
    if (state->have != 4) return Z_DATA_ERROR;
    in = strm->total_in;
    out = strm->total_out;
    inflateReset(strm);
    strm->total_in = in;
    strm->total_out = out;
    state->mode = TYPE;
    return Z_OK;
}

/*
   Returns true if inflate is currently at the end of a block generated by
   Z_SYNC_FLUSH or Z_FULL_FLUSH. This function is used by one PPP
   implementation to provide an additional safety check. PPP uses
   Z_SYNC_FLUSH but removes the length bytes of the resulting empty stored
   block. When decompressing, PPP checks that at the end of input packet,
   inflate is waiting for these length bytes.
 */
#if 0
int ZEXPORT inflateSyncPoint(strm)
z_streamp strm;
#else
int ZEXPORT inflateSyncPoint(z_streamp strm)
#endif
{
    struct inflate_state FAR* state;

    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
    return state->mode == STORED && state->bits == 0;
}
#if 0
int ZEXPORT inflateCopy(dest, source)
z_streamp dest;
z_streamp source;
#else
int ZEXPORT inflateCopy(z_streamp dest, z_streamp source)
#endif
{
    struct inflate_state FAR* state;
    struct inflate_state FAR* copy;
    unsigned char FAR* window;
    unsigned wsize;

    /* check input */
    if (inflateStateCheck(source) || dest == Z_NULL) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)source->state;

    /* allocate space */
    copy = (struct inflate_state FAR*)ZALLOC(source, 1, sizeof(struct inflate_state));
    if (copy == Z_NULL) return Z_MEM_ERROR;
    window = Z_NULL;
    if (state->window != Z_NULL) {
        window = (unsigned char FAR*)ZALLOC(source, 1U << state->wbits, sizeof(unsigned char));
        if (window == Z_NULL) {
            ZFREE(source, copy);
            return Z_MEM_ERROR;
        }
    }

    /* copy state */
    zmemcpy((voidpf)dest, (voidpf)source, sizeof(z_stream));
    zmemcpy((voidpf)copy, (voidpf)state, sizeof(struct inflate_state));
    copy->strm = dest;
    if (state->lencode >= state->codes && state->lencode <= state->codes + ENOUGH - 1) {
        copy->lencode = copy->codes + (state->lencode - state->codes);
        copy->distcode = copy->codes + (state->distcode - state->codes);
    }
    copy->next = copy->codes + (state->next - state->codes);
    if (window != Z_NULL) {
        wsize = 1U << state->wbits;
        zmemcpy(window, state->window, wsize);
    }
    copy->window = window;
    dest->state = (struct internal_state FAR*)copy;
    return Z_OK;
}

#if 0
int ZEXPORT inflateUndermine(strm, subvert)
z_streamp strm;
int subvert;
#else
int ZEXPORT inflateUndermine(z_streamp strm, int subvert)
#endif
{
    struct inflate_state FAR* state;

    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
#ifdef INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
    state->sane = !subvert;
    return Z_OK;
#else
    (void)subvert;
    state->sane = 1;
    return Z_DATA_ERROR;
#endif
}

#if 0
int ZEXPORT inflateValidate(strm, check)
z_streamp strm;
int check;
#else
int ZEXPORT inflateValidate(z_streamp strm, int check)
#endif
{
    struct inflate_state FAR* state;

    if (inflateStateCheck(strm)) return Z_STREAM_ERROR;
    state = (struct inflate_state FAR*)strm->state;
    if (check)
        state->wrap |= 4;
    else
        state->wrap &= ~4;
    return Z_OK;
}

#if 0
long ZEXPORT inflateMark(strm)
z_streamp strm;
#else
long ZEXPORT inflateMark(z_streamp strm)
#endif
{
    struct inflate_state FAR* state;

    if (inflateStateCheck(strm)) return -(1L << 16);
    state = (struct inflate_state FAR*)strm->state;
    return (long)(((unsigned long)((long)state->back)) << 16) +
           (state->mode == COPY ? state->length : (state->mode == MATCH ? state->was - state->length : 0));
}

#if 0
unsigned long ZEXPORT inflateCodesUsed(strm)
z_streamp strm;
#else
unsigned long ZEXPORT inflateCodesUsed(z_streamp strm)
#endif
{
    struct inflate_state FAR* state;
    if (inflateStateCheck(strm)) return (unsigned long)-1;
    state = (struct inflate_state FAR*)strm->state;
    return (unsigned long)(state->next - state->codes);
}
