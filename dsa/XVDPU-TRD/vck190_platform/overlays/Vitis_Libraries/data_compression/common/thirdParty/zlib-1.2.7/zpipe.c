/* zpipe.c: example of proper use of zlib's inflate() and deflate()
   Not copyrighted -- provided to the public domain
   Version 1.4  11 December 2005  Mark Adler */

/* Version history:
   1.0  30 Oct 2004  First version
   1.1   8 Nov 2004  Add void casting for unused return values
                     Use switch statement for inflate() return values
   1.2   9 Nov 2004  Add assertions to document zlib guarantees
   1.3   6 Apr 2005  Remove incorrect assertion in inf()
   1.4  11 Dec 2005  Add hack to avoid MSDOS end-of-line conversions
                     Avoid some compiler warnings for input and output buffers
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iomanip>
#include <chrono>
#include <vector>
#include "zlib.h"

#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#include <fcntl.h>
#include <io.h>
#define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#define SET_BINARY_MODE(file)
#endif

#define CHUNK (256 * 1024)

/* Compress from file source to file dest until EOF on source.
   def() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_STREAM_ERROR if an invalid compression
   level is supplied, Z_VERSION_ERROR if the version of zlib.h and the
   version of the library linked do not match, or Z_ERRNO if there is
   an error reading or writing the files. */
int def(uint8_t* inVec, uint8_t* outVec, size_t input_size, uint64_t* csize, uint16_t num_iter, int level) {
    int ret, flush;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    uint64_t inIdx = 0;
    uint64_t outIdx = 0;

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    ret = deflateInit(&strm, level);
    if (ret != Z_OK) return ret;

    for (int i = 0; i < num_iter; i++) {
        inIdx = 0;
        outIdx = 0;
        /* compress until end of file */
        do {
            uint64_t chunk = ((inIdx + CHUNK) >= input_size) ? (input_size - inIdx) : CHUNK;
            memcpy(&in, &inVec[inIdx], chunk);
            inIdx += chunk;
            strm.avail_in = chunk;

            flush = (inIdx >= input_size) ? Z_FINISH : Z_NO_FLUSH;
            strm.next_in = in;

            /* run deflate() on input until output buffer not full, finish
               compression if all of source has been read in */
            do {
                strm.avail_out = CHUNK;
                strm.next_out = out;
                ret = deflate(&strm, flush);   /* no bad return value */
                assert(ret != Z_STREAM_ERROR); /* state not clobbered */
                have = CHUNK - strm.avail_out;
                memcpy(&outVec[outIdx], &out, have);
                outIdx += have;
            } while (strm.avail_out == 0);
            assert(strm.avail_in == 0); /* all input will be used */

            /* done when last data in file processed */
        } while (flush != Z_FINISH);
        assert(ret == Z_STREAM_END); /* stream will be complete */

        /* reset the stream pointer */
        (void)deflateReset(&strm);
    }

    /* close the stream pointer */
    (void)deflateEnd(&strm);

    // Final output size
    *csize = outIdx;

    return Z_OK;
}

/* Decompress from file source to file dest until stream ends or EOF.
   inf() returns Z_OK on success, Z_MEM_ERROR if memory could not be
   allocated for processing, Z_DATA_ERROR if the deflate data is
   invalid or incomplete, Z_VERSION_ERROR if the version of zlib.h and
   the version of the library linked do not match, or Z_ERRNO if there
   is an error reading or writing the files. */
int inf(uint8_t* inVec, uint8_t* outVec, size_t input_size, uint64_t* csize, uint16_t numiter) {
    int ret;
    unsigned have;
    z_stream strm;
    unsigned char in[CHUNK];
    unsigned char out[CHUNK];

    uint64_t inIdx = 0;
    uint64_t outIdx = 0;

    /* allocate inflate state */
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit(&strm);
    if (ret != Z_OK) return ret;

    for (int i = 0; i < numiter; i++) {
        inIdx = 0;
        outIdx = 0;
        do {
            uint64_t chunk = ((inIdx + CHUNK) >= input_size) ? (input_size - inIdx) : CHUNK;
            memcpy(&in, &inVec[inIdx], chunk);
            inIdx += chunk;
            strm.avail_in = chunk;

            if (strm.avail_in == 0) break;
            strm.next_in = in;

            do {
                strm.avail_out = CHUNK;
                strm.next_out = out;
                ret = inflate(&strm, Z_NO_FLUSH);
                assert(ret != Z_STREAM_ERROR);

                switch (ret) {
                    case Z_NEED_DICT:
                        ret = Z_DATA_ERROR;
                    case Z_DATA_ERROR:
                    case Z_MEM_ERROR:
                        (void)inflateEnd(&strm);
                        return ret;
                }
                have = CHUNK - strm.avail_out;
                memcpy(&outVec[outIdx], &out, have);
                outIdx += have;
            } while (strm.avail_out == 0);
        } while (ret != Z_STREAM_END);
    }

    *csize = outIdx;

    /* clean up and return */
    (void)inflateEnd(&strm);
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

/* report a zlib or i/o error */
void zerr(int ret) {
    fputs("zpipe: ", stderr);
    switch (ret) {
        case Z_ERRNO:
            if (ferror(stdin)) fputs("error reading stdin\n", stderr);
            if (ferror(stdout)) fputs("error writing stdout\n", stderr);
            break;
        case Z_STREAM_ERROR:
            fputs("invalid compression level\n", stderr);
            break;
        case Z_DATA_ERROR:
            fputs("invalid or incomplete deflate data\n", stderr);
            break;
        case Z_MEM_ERROR:
            fputs("out of memory\n", stderr);
            break;
        case Z_VERSION_ERROR:
            fputs("zlib version mismatch!\n", stderr);
    }
}
