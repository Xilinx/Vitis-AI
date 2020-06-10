#ifndef CAFFE_YOLO_TRANSFORMER_HPP_
#define CAFFE_YOLO_TRANSFORMER_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/blob.hpp"

namespace caffe {

//#ifdef USE_OPENCV
typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;

inline float get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

inline void set_pixel(image m, int x, int y, int c, float val)
{
    if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] = val;
}

inline void add_pixel(image m, int x, int y, int c, float val)
{
    assert(x < m.w && y < m.h && c < m.c);
    m.data[c*m.h*m.w + y*m.w + x] += val;
}

image make_empty_image(int w, int h, int c);
image make_image(int w, int h, int c);

void free_image(image m);
void fill_image(image m, float s);
void embed_image(image source, image dest, int dx, int dy);
void rgbgr_image(image im);
void ipl_into_image(IplImage* src, image im);
image ipl_to_image(IplImage* src);
image resize_image(image im, int w, int h);
image letterbox_image(image im, int w, int h);
image load_image_cv(const char *filename, int channels);
image load_image_yolo(const char* file, int w, int h, int c);

template<typename Dtype>
void yolo_transform(cv::Mat cv_img, Blob<Dtype>* blob);
//#endif  // USE_OPENCV

} // namespace

#endif  // CAFFE_YOLO_TRANSFORMER_
