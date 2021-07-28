#ifndef _YOLO_H_
#define _YOLO_H_

extern "C" {

struct Array {
    int height;
    int width;
    int channels;
    float* data;
};

int yolov2_postproc(
    Array* output_tensors, int nArrays, const float* biases,
    int net_h, int net_w, int classes, int anchor_cnt,
    int img_height, int img_width, float conf_thresh,
    float iou_thresh, int batch_idx, float** retboxes);

int yolov3_postproc(
    Array* output_tensors, int nArrays, const float* biases,
    int net_h, int net_w, int classes, int anchorCnt,
    int img_height, int img_width, float conf_thresh,
    float iou_thresh, int batch_idx, float** retboxes);

void clearBuffer(float* buf);

} // extern "C"
#endif
