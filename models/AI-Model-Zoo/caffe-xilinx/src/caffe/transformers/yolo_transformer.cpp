#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <iostream>
#include "caffe/transformers/yolo_transformer.hpp"

typedef void* mat_cv;
using std::cerr;

namespace caffe {

image mat_to_image(cv::Mat mat)
{
    int w = mat.cols;
    int h = mat.rows;
    int c = mat.channels();
    image im = make_image(w, h, c);
    unsigned char *data = (unsigned char *)mat.data;
    int step = mat.step;
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < c; ++k) {
            for (int x = 0; x < w; ++x) {
                //uint8_t val = mat.ptr<uint8_t>(y)[c * x + k];
                //uint8_t val = mat.at<Vec3b>(y, x).val[k];
                //im.data[k*w*h + y*w + x] = val / 255.0f;

                im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
            }
        }
    }
    return im;
}

mat_cv *load_image_mat_cv(const char *filename, int flag)
{
    cv::Mat *mat_ptr = NULL;
    try {
        cv::Mat mat = cv::imread(filename, flag);
        if (mat.empty())
        {
            std::string shrinked_filename = filename;
            if (shrinked_filename.length() > 1024) {
                shrinked_filename.resize(1024);
                shrinked_filename = std::string("name is too long: ") + shrinked_filename;
            }
            cerr << "Cannot load image " << shrinked_filename << std::endl;
            std::ofstream bad_list("bad.list", std::ios::out | std::ios::app);
            bad_list << shrinked_filename << std::endl;
            //if (check_mistakes) getchar();
            return NULL;
        }
        cv::Mat dst;
        if (mat.channels() == 3) cv::cvtColor(mat, dst, cv::COLOR_BGR2RGB);
        else if (mat.channels() == 4) cv::cvtColor(mat, dst, cv::COLOR_RGBA2BGRA);
        else dst = mat;

        mat_ptr = new cv::Mat(dst);

        return (mat_cv *)mat_ptr;
    }
    catch (...) {
        cerr << "OpenCV exception: load_image_mat_cv \n";
    }
    if (mat_ptr) delete mat_ptr;
    return NULL;
}
// ----------------------------------------

cv::Mat load_image_mat(const char *filename, int channels)
{
    int flag = cv::IMREAD_UNCHANGED;
    if (channels == 0) flag = cv::IMREAD_COLOR;
    else if (channels == 1) flag = cv::IMREAD_GRAYSCALE;
    else if (channels == 3) flag = cv::IMREAD_COLOR;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }
    //flag |= IMREAD_IGNORE_ORIENTATION;    // un-comment it if you want

    cv::Mat *mat_ptr = (cv::Mat *)load_image_mat_cv(filename, flag);

    if (mat_ptr == NULL) {
        return cv::Mat();
    }
    cv::Mat mat = *mat_ptr;
    delete mat_ptr;

    return mat;
}
// ----------------------------------------


image load_image_cv2(const char *filename, int channels)
{
    cv::Mat mat = load_image_mat(filename, channels);

    if (mat.empty()) {
        return make_image(10, 10, channels);
    }
    return mat_to_image(mat);
}

//#ifdef USE_OPENCV
image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

void free_image(image m)
{
    if(m.data){
        free(m.data);
    }
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w,h,c);
    out.data = (float*) calloc(h*w*c, sizeof(float));
    return out;
}

void fill_image(image m, float s)
{
    int i;
    for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void embed_image(image source, image dest, int dx, int dy)
{
    int x,y,k;
    for(k = 0; k < source.c; ++k){
        for(y = 0; y < source.h; ++y){
            for(x = 0; x < source.w; ++x){
                float val = get_pixel(source, x,y,k);
                set_pixel(dest, dx+x, dy+y, k, val);
            }
        }
    }
}

void ipl_into_image(IplImage* src, image im)
{
    unsigned char *data = (unsigned char *)src->imageData;
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    int step = src->widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/256.;
            }
        }
    }
}

image ipl_to_image(IplImage* src)
{
    int h = src->height;
    int w = src->width;
    int c = src->nChannels;
    image out = make_image(w, h, c);
    ipl_into_image(src, out);
    return out;
}

void rgbgr_image(image im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i){
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

image resize_image(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);   
    image part = make_image(w, im.h, im.c);
    int r, c, k;
    float w_scale = (float)(im.w - 1) / (w - 1);
    float h_scale = (float)(im.h - 1) / (h - 1);
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < im.h; ++r){
            for(c = 0; c < w; ++c){
                float val = 0;
                if(c == w-1 || im.w == 1){
                    val = get_pixel(im, im.w-1, r, k);
                } else {
                    float sx = c*w_scale;
                    int ix = (int) sx;
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
                }
                set_pixel(part, c, r, k, val);
            }
        }
    }
    for(k = 0; k < im.c; ++k){
        for(r = 0; r < h; ++r){
            float sy = r*h_scale;
            int iy = (int) sy;
            float dy = sy - iy;
            for(c = 0; c < w; ++c){
                float val = (1-dy) * get_pixel(part, c, iy, k);
                set_pixel(resized, c, r, k, val);
            }
            if(r == h-1 || im.h == 1) continue;
            for(c = 0; c < w; ++c){
                float val = dy * get_pixel(part, c, iy+1, k);
                add_pixel(resized, c, r, k, val);
            }
        }
    }

    free_image(part);
    return resized;
}

image load_image_cv(const char *filename, int channels)
{
    IplImage* src = 0;
    int flag = -1;
    if (channels == 0) flag = -1;
    else if (channels == 1) flag = 0;
    else if (channels == 3) flag = 1;
    else {
        fprintf(stderr, "OpenCV can't force load with %d channels\n", channels);
    }

    if( (src = cvLoadImage(filename, flag)) == 0 )
    {
        fprintf(stderr, "Cannot load image \"%s\"\n", filename);
        return make_image(10,10,3);
        //exit(0);
    }
    image out = ipl_to_image(src);
    cvReleaseImage(&src);
    rgbgr_image(out);
    return out;
}

image letterbox_image(image im, int w, int h)
{
    int new_w = im.w;
    int new_h = im.h;
    if (((float)w/im.w) < ((float)h/im.h)) {
        new_w = w;
        new_h = (im.h * w)/im.w;
    } else {
        new_h = h;
        new_w = (im.w * h)/im.h;
    }
    image resized = resize_image(im, new_w, new_h);
    image boxed = make_image(w, h, im.c);
    fill_image(boxed, .5);
    //int i;
    //for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
    embed_image(resized, boxed, (w-new_w)/2, (h-new_h)/2); 
    free_image(resized);
    return boxed;
}

image load_image_yolo(const char* file, int w, int h, int c) {
  image img = load_image_cv(file, c);
  image img_yolo = letterbox_image(img, w, h);
  free_image(img);
  return img_yolo;
}

image load_image_yolov4(const char* file, int w, int h, int c, int letterbox) {
  image img = load_image_cv2(file, c);
  image img_yolo;
  if (letterbox) {
    img_yolo = letterbox_image(img, w, h);
  } else {
    img_yolo = resize_image(img, w, h);
  } 
  free_image(img);
  return img_yolo;
}

image yolo_transform_image(cv::Mat cv_img, int w ,int h){

  IplImage img = IplImage(cv_img);

  // IplImage to image
  image img_out = ipl_to_image(&img);

//  cvReleaseImage(img);
  // rgbgr
  rgbgr_image(img_out);

  // some processing
  image img_yolo = letterbox_image(img_out, w, h);
  //image img_yolo = resize_image(img_out, w, h);
  free_image(img_out);
  return img_yolo;

}

image yolov4_transform_image(cv::Mat cv_img, int w ,int h, int letterbox){

  cv::Mat img_dst;
  if (cv_img.channels() == 3) cv::cvtColor(cv_img, img_dst, cv::COLOR_BGR2RGB);
  else if (cv_img.channels() == 4) cv::cvtColor(cv_img, img_dst, cv::COLOR_BGRA2RGBA);
  else img_dst = cv_img;

  // IplImage to image
  image img = mat_to_image(img_dst);

  image img_yolo;
  if (letterbox) {
    img_yolo = letterbox_image(img, w, h);
  } else {
    img_yolo = resize_image(img, w, h);
  } 
  // some processing
  // image img_yolo = letterbox_image(img_yolo, w, h);
  // free_image(img_out);
  free_image(img);
  return img_yolo;

}

template<typename Dtype>
void yolo_transform(cv::Mat cv_img, Blob<Dtype>* blob) {
  yolo_transform(cv_img, blob, 1);   
}

template<typename Dtype>
void yolo_transform(cv::Mat cv_img, Blob<Dtype>* blob, int letterbox) {
  vector<int> shape = blob->shape();
  int n = shape[0];
  int c = shape[1];
  int h = shape[2];
  int w = shape[3];

  CHECK(n==1 && c==cv_img.channels());

  // convert to IplImage
  // IplImage img = IplImage(cv_img);

  // IplImage to image
  // image img_out = ipl_to_image(&img);

  // rgbgr
   // rgbgr_image(img_out);

  // some processing
  // image img_yolo = letterbox_image(img_out, w, h);

  image img_yolo;
  if (!letterbox) {
    img_yolo = yolov4_transform_image(cv_img, w, h, letterbox);
  } else {
    img_yolo = yolo_transform_image(cv_img, w, h);
  }
  Dtype* ptr = blob->mutable_cpu_data();
  for(int i=0; i<c*h*w; ++i) {
    ptr[i] = (Dtype)(img_yolo.data[i]);
  }
  // free_image(img_out);
  free_image(img_yolo);
}
//#endif  // USE_OPENCV

template void yolo_transform(cv::Mat cv_img, Blob<float>* blob);
template void yolo_transform(cv::Mat cv_img, Blob<double>* blob);

template void yolo_transform(cv::Mat cv_img, Blob<float>* blob, int letterbox);
template void yolo_transform(cv::Mat cv_img, Blob<double>* blob, int letterbox);

} // namespace
