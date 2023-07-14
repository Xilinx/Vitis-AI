/*
 * Copyright (c) 2019 Mujib Haider
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _JFIF_H_
#define _JFIF_H_

#ifdef __cplusplus
extern "C" {
#endif

#define SHORT_SWAP(y, x) y = (x>>8 | x<<8)
#define JFIF_MAX_FILE_SIZE 20E6

typedef enum {
    JFIF_MARKER_STUFF = 0x00ff,
    JFIF_MARKER_SOF0  = 0xc0ff,
    JFIF_MARKER_SOF1  = 0xc1ff,
    JFIF_MARKER_SOF2  = 0xc2ff,
    JFIF_MARKER_SOF3  = 0xc3ff,
    JFIF_MARKER_DHT   = 0xc4ff,
    JFIF_MARKER_SOF5  = 0xc5ff,
    JFIF_MARKER_SOF6  = 0xc6ff,
    JFIF_MARKER_SOF7  = 0xc7ff,
    JFIF_MARKER_JPG   = 0xc8ff,
    JFIF_MARKER_SOF9  = 0xc9ff,
    JFIF_MARKER_SOF10 = 0xcaff,
    JFIF_MARKER_SOF11 = 0xcbff,
    JFIF_MARKER_DAC   = 0xccff,
    JFIF_MARKER_SOF13 = 0xcdff,
    JFIF_MARKER_SOF14 = 0xceff,
    JFIF_MARKER_SOF15 = 0xcfff,
    JFIF_MARKER_RST0  = 0xd0ff,
    JFIF_MARKER_RST1  = 0xd1ff,
    JFIF_MARKER_RST2  = 0xd2ff,
    JFIF_MARKER_RST3  = 0xd3ff,
    JFIF_MARKER_RST4  = 0xd4ff,
    JFIF_MARKER_RST5  = 0xd5ff,
    JFIF_MARKER_RST6  = 0xd6ff,
    JFIF_MARKER_RST7  = 0xd7ff,
    JFIF_MARKER_SOI   = 0xd8ff,
    JFIF_MARKER_EOI   = 0xd9ff,
    JFIF_MARKER_SOS   = 0xdaff,
    JFIF_MARKER_DQT   = 0xdbff,
    JFIF_MARKER_DNL   = 0xdcff,
    JFIF_MARKER_DRI   = 0xddff,
    JFIF_MARKER_DHP   = 0xdeff,
    JFIF_MARKER_EXP   = 0xdfff,
    JFIF_MARKER_APP0  = 0xe0ff,
    JFIF_MARKER_APP1  = 0xe1ff,
    JFIF_MARKER_APP2  = 0xe2ff,
    JFIF_MARKER_APP3  = 0xe3ff,
    JFIF_MARKER_APP4  = 0xe4ff,
    JFIF_MARKER_APP5  = 0xe5ff,
    JFIF_MARKER_APP6  = 0xe6ff,
    JFIF_MARKER_APP7  = 0xe7ff,
    JFIF_MARKER_APP8  = 0xe8ff,
    JFIF_MARKER_APP9  = 0xe9ff,
    JFIF_MARKER_APP10 = 0xeaff,
    JFIF_MARKER_APP11 = 0xebff,
    JFIF_MARKER_APP12 = 0xecff,
    JFIF_MARKER_APP13 = 0xedff,
    JFIF_MARKER_APP14 = 0xeeff,
    JFIF_MARKER_APP15 = 0xefff,
    JFIF_MARKER_JPG0  = 0xf0ff,
    JFIF_MARKER_JPG13 = 0xfdff,
    JFIF_MARKER_COM   = 0xfeff,
    JFIF_MARKER_TEM   = 0x01ff,
    JFIF_MARKER_UNDEF = 0xffff
} jfif_marker_t;

typedef enum {
    JFIF_FORMAT_INVALID,
    JFIF_FORMAT_YUV400_N,
    JFIF_FORMAT_YUV420_N,
    JFIF_FORMAT_YUV420_I,
    JFIF_FORMAT_YUV422_N,
    JFIF_FORMAT_YUV422_I,
    JFIF_FORMAT_YUV444_N,
    JFIF_FORMAT_YUV444_I,
    JFIF_FORMAT_UNDEF
} jfif_format_t;

typedef struct {
    unsigned int APP:16;
    unsigned int Lp:16;
    unsigned char identifier[5];
    unsigned int version:16;
    unsigned int units:8;
    unsigned int Hdensity:16;
    unsigned int Vdensity:16;
    unsigned int HthumbnailA:8;
    unsigned int VthumbnailA:8;
    unsigned char *RGB;
} jfif_app_header_t;

typedef struct {
    unsigned int C:8;  /* Component Id */
    unsigned int H:4;  /* Horizontal Sampling, high four bits */
    unsigned int V:4;  /* Vertical Sampling, lower four bits */
    unsigned int Tq:8; /* Quantization Table Id */
} jfif_frame_component_t;

typedef struct {
    unsigned int SOF:16;
    unsigned int Lf:16;  /* Frame Header Length */
    unsigned int P:8;    /* Sample Precision,  precision in bits */
    unsigned int Y:16;   /* Maximum Number Of Lines  */
    unsigned int X:16;   /* Maximum Number Of Samples, per line */
    unsigned int Nf:8;   /* Number of Image Components in Frame */
    jfif_frame_component_t c[3];
} jfif_frame_header_t;

typedef struct {
    unsigned int Cs:8;  /* scan component selector */
    unsigned int Td:4;  /* DC entropy coding Table Selector */
    unsigned int Ta:4;  /* AC entropy codong Table Selector */
} jfif_scan_component_t;

typedef struct {
    unsigned int SOS:16;
    unsigned int Ls:16;  /* Scan Header Length */
    unsigned int Ns:8;   /* Number Of Image Components In Scan */
    jfif_scan_component_t c[3];
    unsigned int Ss:8;
    unsigned int Se:8;
    unsigned int Ah:4;
    unsigned int Al:4;
} jfif_scan_header_t;

typedef struct {
    unsigned int DQT:16;
    unsigned int Lq:16;
    unsigned int Pq:4;
    unsigned int Tq:4;
    unsigned char Qk[128];
} jfif_quantization_table_t;

typedef struct {
    unsigned int DHT:16;
    unsigned int Lh:16;
    unsigned int Tc:4;
    unsigned int Th:4;
    unsigned int L[16];
    unsigned int V[16][256];
} jfif_huffman_table_t;

/* JFIF context */
typedef struct {
    /* File pointers */
    unsigned char *file_ptr;
    unsigned int   file_size;
    unsigned char *soi_ptr;
    unsigned char *app_ptr;
    unsigned char *dqt_ptr[2];
    unsigned char  dqt_num;
    unsigned char *sof_ptr;
    unsigned char *dht_ptr[4];
    unsigned char  dht_num;
    unsigned char *sos_ptr[3];
    unsigned char  sos_num;
    unsigned char *eoi_ptr;
    unsigned char *scan_ptr[3];
    unsigned int   scan_size[3];
    unsigned int   parser_full_file;

    /* File contents */
    jfif_app_header_t app_header;
    jfif_frame_header_t frame_header;
    jfif_scan_header_t scan_header[3];
    jfif_quantization_table_t q_table[2];
    jfif_huffman_table_t h_table[4];
} jfif_t;

/* Global variables */
extern const jfif_quantization_table_t jfif_default_q_table[2];
extern const jfif_huffman_table_t jfif_default_h_table[4];

/* Initialization routines */
jfif_t        *jfif_init(unsigned char *FilePtr);
void           jfif_reset(jfif_t *ctx);
void           jfif_destroy(jfif_t *ctx);
void           jfif_copy(jfif_t *dst, jfif_t *src);

/* File I/O routines */
unsigned int   jfif_file_read(jfif_t *ctx, const char * filename);
unsigned int   jfif_file_write(jfif_t *ctx, const char *filename);

/* Parser routines */
int            jfif_parse(jfif_t *ctx);

/* Composer routines */
int            jfif_compose_defaults(jfif_t *ctx, jfif_format_t format, int x, int y, int qf);
int            jfif_compose_scan(jfif_t *ctx, int cmps, unsigned int size, unsigned char *scan);
int            jfif_compose(jfif_t *ctx);

/* Accessor routines */
int            jfif_get_frame_components(jfif_t *ctx);
int            jfif_get_frame_width(jfif_t *ctx, int id);
int            jfif_get_frame_height(jfif_t *ctx, int id);
int            jfif_get_frame_h_factor(jfif_t *ctx, int id);
int            jfif_get_frame_v_factor(jfif_t *ctx, int id);
int            jfif_get_frame_x(jfif_t *ctx);
int            jfif_get_frame_y(jfif_t *ctx);
int            jfif_get_scan_num(jfif_t *ctx);
int            jfif_get_scan_components(jfif_t *ctx, int id);
int            jfif_get_scan_start_component(jfif_t *ctx, int id);
int            jfif_get_scan_end_component(jfif_t *ctx, int id);
unsigned int   jfif_get_scan_size(jfif_t *ctx, int id);
unsigned char *jfif_get_scan_data(jfif_t *ctx, int id);
int            jfif_get_q_num(jfif_t *ctx);
unsigned char *jfif_get_q_offset(jfif_t *ctx, int id);
int            jfif_get_q_length(jfif_t *ctx, int id);
int            jfif_get_q_precision(jfif_t *ctx, int id);
unsigned char *jfif_get_q_data(jfif_t *ctx, int id);
int            jfif_get_h_num(jfif_t *ctx);
unsigned char *jfif_get_h_offset(jfif_t *ctx, int id);
int            jfif_get_h_length(jfif_t *ctx, int id);

int            jfif_set_scan_num(jfif_t *ctx, int num);
int            jfif_set_scan_components(jfif_t *ctx, int id, int num);
int            jfif_set_scan_size(jfif_t *ctx, int id, unsigned int size);
int            jfif_set_scan_data(jfif_t *ctx, int id, unsigned char *scan);

/* Debug routines */
void           jfif_dump_app_header(jfif_t *ctx);
void           jfif_dump_frame_header(jfif_t *ctx);
void           jfif_dump_scan_header(jfif_t *ctx, int id);
void           jfif_dump_scan_data(jfif_t *ctx, int id);
void           jfif_dump_q_table(jfif_t *ctx, int id);
void           jfif_dump_h_table(jfif_t *ctx, int id);
void           jfif_dump_file(jfif_t *ctx, int loglevel);

#ifdef __cplusplus
}
#endif

#endif // _JFIF_H_
