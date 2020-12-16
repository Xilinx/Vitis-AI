
#ifndef CAFFE_SEGMENT_PIXEL_IOU_LAYER_H
#define CAFFE_SEGMENT_PIXEL_IOU_LAYER_H

#include <vector>
#include <unordered_map>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe {

    template <typename Dtype>
    class SegmentPixelIOULayer : public Layer<Dtype> {
    public:
        explicit SegmentPixelIOULayer(const LayerParameter& param) : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "IOU"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }

        // If there are two top blobs, then the second blob will contain
        // accuracies per class.
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 1; }

    protected:
        /**
         * @param bottom input Blob vector (length 2)
         *   -# @f$ (N \times C \times H' \times W') @f$
         *      the predictions @f$ x @f$, a Blob with values in
         *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
         *      the @f$ K = CHW @f$ classes. Each @f$ x_n @f$ is mapped to a predicted
         *      label @f$ \hat{l}_n @f$ given by its maximal index:
         *      @f$ \hat{l}_n = \arg\max\limits_k x_{nk} @f$
         *   -# @f$ (N \times 1 \times H \times W) @f$
         *      the labels @f$ l @f$, an integer-valued Blob with values
         *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
         *      indicating the correct class label among the @f$ K @f$ classes
         * @param top output Blob vector (length 1)
         *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
         *      the computed IOU: @f$
         *        \frac{1}{N} \sum\limits_{n=1}^N \delta\{ \hat{l}_n = l_n \}
         *      @f$, where @f$
         *      \delta\{\mathrm{condition}\} = \left\{
         *         \begin{array}{lr}
         *            1 & \mbox{if condition} \\
         *            0 & \mbox{otherwise}
         *         \end{array} \right.
         *      @f$
         */
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);
//        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//                                 const vector<Blob<Dtype>*>& top);


        /// @brief Not implemented -- SegmentPixelIOULayer cannot be used as a loss.
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
            for (int i = 0; i < propagate_down.size(); ++i) {
                if (propagate_down[i]) { NOT_IMPLEMENTED; }
            }
        }
//        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//                                  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

        std::vector<u_int64_t> tp_;
        std::vector<u_int64_t> fp_;
        std::vector<u_int64_t> fn_;
        u_int64_t images_;
        const u_int64_t total_images_ = 500;

        const char* labels_ [19] = {
                "road",
                "sidewalk",
                "building",
                "wall",
                "fence",
                "pole",
                "traffic light",
                "traffic sign",
                "vegetation",
                "terrain",
                "sky",
                "person",
                "rider",
                "car",
                "truck",
                "bus",
                "train",
                "motorcycle",
                "bicycle"
        };

        std::unordered_map<int, int> labels_map_ = {
            {7, 0},
            {8, 1},
            {11, 2},
            {12, 3},
            {13, 4},
            {17, 5},
            {19, 6},
            {20, 7},
            {21, 8},
            {22, 9},
            {23, 10},
            {24, 11},
            {25, 12},
            {26, 13},
            {27, 14},
            {28, 15},
            {31, 16},
            {32, 17},
            {33, 18}
        };
        std::unordered_map<int, int> labels_to_image_map_ = {
                {0, 7},
                {1, 8},
                {2, 11},
                {3, 12},
                {4, 13},
                {5, 17},
                {6, 19},
                {7, 20},
                {8, 21},
                {9, 22},
                {10, 23},
                {11, 24},
                {12, 25},
                {13, 26},
                {14, 27},
                {15, 28},
                {16, 31},
                {17, 32},
                {18, 33}
        };
    };
}

#endif //CAFFE_SEGMENT_PIXEL_IOU_LAYER_H
