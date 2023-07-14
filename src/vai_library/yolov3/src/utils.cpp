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

#include "utils.hpp"

using cv::Mat;
using cv::Size;

namespace vitis {
namespace ai {
namespace yolov3 {

image load_image_cv(const cv::Mat& img);
image letterbox_image(image im, int w, int h);
image make_image(int w, int h, int c);
image make_empty_image(int w, int h, int c);
void free_image(image m);
image resize_image(image im, int w, int h);
void fill_image(image m, float s);
void embed_image(image source, image dest, int dx, int dy);
float get_pixel(image m, int x, int y, int c);
void set_pixel(image m, int x, int y, int c, float val);
void add_pixel(image m, int x, int y, int c, float val);
image ipl_to_image(IplImage* src);
void ipl_into_image(IplImage* src, image im);

//# Overload method with float input data for DPUV1
void convertInputImage(const cv::Mat& frame, int width, int height, int channel,
                       float scale, float* data) {
  int size = width * height * channel;
  image img_new = load_image_cv(frame);
  image img_yolo = letterbox_image(img_new, width, height);

  vector<float> bb(size);

  for (int a = 0; a < channel; ++a) {
    for (int b = 0; b < height; ++b) {
      for (int c = 0; c < width; ++c) {
        bb[a * width * height + b * width + c] =
            img_yolo.data[a * height * width + b * width + c];
      }
    }
  }

  for (int i = 0; i < size; ++i) {
    data[i] = bb.data()[i];
    if (data[i] < 0) data[i] = (float)scale;
  }

  free_image(img_new);
  free_image(img_yolo);
}

void convertInputImage(const cv::Mat& frame, int width, int height, int channel,
                       float scale, int8_t* data) {
  int size = width * height * channel;
  image img_new = load_image_cv(frame);
  image img_yolo = letterbox_image(img_new, width, height);

  vector<float> bb(size);
  for (int b = 0; b < height; ++b) {
    for (int c = 0; c < width; ++c) {
      for (int a = 0; a < channel; ++a) {
        bb[b * width * channel + c * channel + a] =
            img_yolo.data[a * height * width + b * width + c];
      }
    }
  }
  for (int i = 0; i < size; ++i) {
    data[i] = int(bb.data()[i] * scale);
    if (data[i] < 0) data[i] = 127;
  }
  free_image(img_new);
  free_image(img_yolo);
}

image load_image_cv(const cv::Mat& img) {
  int h = img.rows;
  int w = img.cols;
  int c = img.channels();
  image im = make_image(w, h, c);

  unsigned char* data = img.data;

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      im.data[0 * w * h + i * w + j] = data[i * w * 3 + j * 3 + 2] * 0.00390625;
      im.data[1 * w * h + i * w + j] = data[i * w * 3 + j * 3 + 1] * 0.00390625;
      im.data[2 * w * h + i * w + j] = data[i * w * 3 + j * 3 + 0] * 0.00390625;
    }
  }
  return im;
}

image letterbox_image(image im, int w, int h) {
  int new_w = im.w;
  int new_h = im.h;
  if (((float)w / im.w) < ((float)h / im.h)) {
    new_w = w;
    new_h = (im.h * w) / im.w;
  } else {
    new_h = h;
    new_w = (im.w * h) / im.h;
  }
  image resized = resize_image(im, new_w, new_h);
  image boxed = make_image(w, h, im.c);
  fill_image(boxed, .5);

  embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
  free_image(resized);

  return boxed;
}

image make_image(int w, int h, int c) {
  image out = make_empty_image(w, h, c);
  out.data = (float*)calloc(h * w * c, sizeof(float));
  return out;
}

image make_empty_image(int w, int h, int c) {
  image out;
  out.data = 0;
  out.h = h;
  out.w = w;
  out.c = c;
  return out;
}

void free_image(image m) {
  if (m.data) {
    free(m.data);
  }
}

image resize_image(image im, int w, int h) {
  image resized = make_image(w, h, im.c);
  image part = make_image(w, im.h, im.c);
  int r, c, k;
  float w_scale = (float)(im.w - 1) / (w - 1);
  float h_scale = (float)(im.h - 1) / (h - 1);
  for (k = 0; k < im.c; ++k) {
    for (r = 0; r < im.h; ++r) {
      for (c = 0; c < w; ++c) {
        float val = 0;
        if (c == w - 1 || im.w == 1) {
          val = get_pixel(im, im.w - 1, r, k);
        } else {
          float sx = c * w_scale;
          int ix = (int)sx;
          float dx = sx - ix;
          val = (1 - dx) * get_pixel(im, ix, r, k) +
                dx * get_pixel(im, ix + 1, r, k);
        }
        set_pixel(part, c, r, k, val);
      }
    }
  }
  for (k = 0; k < im.c; ++k) {
    for (r = 0; r < h; ++r) {
      float sy = r * h_scale;
      int iy = (int)sy;
      float dy = sy - iy;
      for (c = 0; c < w; ++c) {
        float val = (1 - dy) * get_pixel(part, c, iy, k);
        set_pixel(resized, c, r, k, val);
      }
      if (r == h - 1 || im.h == 1) continue;
      for (c = 0; c < w; ++c) {
        float val = dy * get_pixel(part, c, iy + 1, k);
        add_pixel(resized, c, r, k, val);
      }
    }
  }

  free_image(part);
  return resized;
}

void fill_image(image m, float s) {
  int i;
  for (i = 0; i < m.h * m.w * m.c; ++i) m.data[i] = s;
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

image ipl_to_image(IplImage* src) {
  int h = src->height;
  int w = src->width;
  int c = src->nChannels;
  image out = make_image(w, h, c);
  ipl_into_image(src, out);
  return out;
}

void ipl_into_image(IplImage* src, image im) {
  unsigned char* data = (unsigned char*)src->imageData;
  int h = src->height;
  int w = src->width;
  int c = src->nChannels;
  int step = src->widthStep;
  int i, j, k;

  for (i = 0; i < h; ++i) {
    for (k = 0; k < c; ++k) {
      for (j = 0; j < w; ++j) {
        im.data[k * w * h + i * w + j] = data[i * step + j * c + k] / 256.;
      }
    }
  }
}

float get_pixel(image m, int x, int y, int c) {
  assert(x < m.w && y < m.h && c < m.c);
  return m.data[c * m.h * m.w + y * m.w + x];
}

void set_pixel(image m, int x, int y, int c, float val) {
  if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
  assert(x < m.w && y < m.h && c < m.c);
  m.data[c * m.h * m.w + y * m.w + x] = val;
}

void add_pixel(image m, int x, int y, int c, float val) {
  assert(x < m.w && y < m.h && c < m.c);
  m.data[c * m.h * m.w + y * m.w + x] += val;
}

void rgbgr_image(image im) {
  int i;
  for (i = 0; i < im.w * im.h; ++i) {
    float swap = im.data[i];
    im.data[i] = im.data[i + im.w * im.h * 2];
    im.data[i + im.w * im.h * 2] = swap;
  }
}

void convert_RGB(cv::Mat img) {
  char temp;
  for (int i = 0; i < img.cols * img.rows; ++i) {
    temp = img.data[i * 3];
    img.data[i * 3] = img.data[i * 3 + 2];
    img.data[i * 3 + 2] = temp;
  }
}

cv::Mat letterbox(const cv::Mat& im, int w, int h) {
  float scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);
  int new_w = im.cols * scale;
  int new_h = im.rows * scale;
  cv::Mat img_res;
  cv::resize(im, img_res, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  cv::Mat new_img(cv::Size(w, h), CV_8UC3, cv::Scalar(114, 114, 114));
  int x = (w - new_w) / 2;
  int y = (h - new_h) / 2;
  auto rect = cv::Rect{x, y, new_w, new_h};
  img_res.copyTo(new_img(rect));
  return new_img;
}

cv::Mat letterbox_tf(const cv::Mat& im, int w, int h) {
  float scale = min((float)w / (float)im.cols, (float)h / (float)im.rows);
  int new_w = im.cols * scale;
  int new_h = im.rows * scale;
  Mat img_res;
  cv::resize(im, img_res, Size(new_w, new_h));
  // resize(im, img_res, Size(new_w, new_h), 0, 0, INTER_CUBIC);

  Mat new_img(Size(w, h), CV_8UC3, cv::Scalar(128, 128, 128));
  Mat t_mat = Mat::zeros(2, 3, CV_32FC1);
  t_mat.at<float>(0, 0) = 1;
  t_mat.at<float>(0, 2) = (w - new_w) / 2;
  t_mat.at<float>(1, 1) = 1;
  t_mat.at<float>(1, 2) = (h - new_h) / 2;
  cv::warpAffine(img_res, new_img, t_mat, new_img.size());
  return new_img;
}

}  // namespace yolov3
}  // namespace ai
}  // namespace vitis
