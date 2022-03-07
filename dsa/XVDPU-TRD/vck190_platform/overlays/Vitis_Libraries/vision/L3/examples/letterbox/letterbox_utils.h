/*
 * Copyright 2019 Xilinx, Inc.
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

typedef struct {
    int w;
    int h;
    int c;
    float* data;
} image;

image make_empty_image(int w, int h, int c) {
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c) {
    image out = make_empty_image(w, h, c);
    out.data = (float*)calloc(h * w * c, sizeof(float));
    return out;
}

void fill_image(image m, float s) {
    int i;
    for (i = 0; i < m.h * m.w * m.c; ++i) m.data[i] = s;
}

static float get_pixel(image m, int x, int y, int c) {
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c * m.h * m.w + y * m.w + x];
}

static void set_pixel(image m, int x, int y, int c, float val) {
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c * m.h * m.w + y * m.w + x] = val;
}

void embed_image(image source, image dest, int dx, int dy) {
    int x, y, k;
    for (k = 0; k < source.c; ++k) {
        for (y = 0; y < source.h; ++y) {
            for (x = 0; x < source.w; ++x) {
                float val = get_pixel(source, x, y, k);
                set_pixel(dest, dx + x, dy + y, k, val);
            }
        }
    }
}

image load_image_cv(const cv::Mat& img) {
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();
    image im = make_image(w, h, c);

    unsigned char* data = img.data;

    for (int i = 0; i < h; ++i) {
        for (int k = 0; k < c; ++k) {
            for (int j = 0; j < w; ++j) {
                im.data[k * w * h + i * w + j] = data[i * w * c + j * c + k] / 256.;
            }
        }
    }

    /* for(int i = 0; i < im.w*im.h; ++i){
         float swap = im.data[i];
         im.data[i] = im.data[i+im.w*im.h*2];
         im.data[i+im.w*im.h*2] = swap;
     }
 */
    return im;
}

void write_image_cv(image im, const cv::Mat& img) {
    int h = img.rows;
    int w = img.cols;
    int c = img.channels();

    unsigned char* data = img.data;

    for (int i = 0; i < h; ++i) {
        for (int k = 0; k < c; ++k) {
            for (int j = 0; j < w; ++j) {
                float a = im.data[k * w * h + i * w + j] * 256;
                data[i * w * c + j * c + k] = a;
            }
        }
    }
}
