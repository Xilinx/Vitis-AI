#ifndef CAFFE_PE_DATA_TRANSFORMER_HPP
#define CAFFE_PE_DATA_TRANSFORMER_HPP

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
using namespace cv;
#endif  //USE OPENCV

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class PEDataTransformer {
    public:
    explicit PEDataTransformer(const PETransformationParameter& param, Phase phase);
    virtual ~PEDataTransformer() {}

    /**
    * @brief Initialize the Random number generations if needed by the
    *    transformation.
    */
    void InitRand();

    /**
    * @brief Applies the transformation defined in the data layer's
    * transform_param block to the data.
    *
    * @param im
    *    The sample image
    * @param buf
    *    the paramters of key points
    * @param transformed_data
    *    output data
    * @param transformed_label
    *    output label
    */
    void Transform(Mat& im, const float* buf, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label);

    struct AugmentSelection {
        bool flip;
        float degree;
        Size crop;
        float scale;
    };

    struct Joints {
        vector<Point2f> joints;
        vector<float> isVisible;
    };

    struct MetaData {
        string dataset;
        Size img_size;
        bool isValidation;
        int numOtherPeople;
        int people_index;
        int annolist_index;
        int write_number;
        int total_write_number;
        int epoch;
        Point2f objpos; //objpos_x(float), objpos_y (float)
        float scale_self;
        Joints joint_self; //(3*16)

        vector<Point2f> objpos_other; //length is numOtherPeople
        vector<float> scale_other; //length is numOtherPeople
        vector<Joints> joint_others; //length is numOtherPeople
    };

    void generateLabelMap(Dtype*, Mat&, MetaData meta);

    bool augmentation_flip(Mat& img, Mat& img_aug, Mat& mask_miss, MetaData& meta);
    float augmentation_rotate(Mat& img_src, Mat& img_aug, Mat& mask_miss, MetaData& meta);
    float augmentation_scale(Mat& img, Mat& img_temp, Mat& mask_miss, MetaData& meta);
    Size augmentation_croppad(Mat& img_temp, Mat& img_aug, Mat& mask_miss, Mat& mask_miss_aug, MetaData& meta);

    void RotatePoint(Point2f& p, Mat R);
    bool onPlane(Point p, Size img_size);
    void swapLeftRight(Joints& j);

    int np_in_lmdb_;
    int np_;

    protected:
    /**
    * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
    *
    * @param n
    *    The upperbound (exclusive) value of the random number.
    * @return
    *    A uniformly random integer value from ({0, 1, ..., n-1}).
    */
    virtual int Rand(int n);

    void ReadMetaData(MetaData& meta, const float* buf, int type);
    void TransformMetaJoints(MetaData& meta);
    void TransformJoints(Joints& joints);
    void putGaussianMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma);
    void putVecMaps(Dtype* entryX, Dtype* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);

    void Transform(Mat& img, const float* buf, Dtype* transformed_data, Dtype* transformed_label);

    // Tranformation parameters
    PETransformationParameter param_;

    shared_ptr<Caffe::RNG> rng_;
    Phase phase_;
    Blob<Dtype> data_mean_;
    vector<Dtype> mean_values_;
    bool single_;
};

}  // namespace caffe

#endif  // CAFFE_CPM_DATA_TRANSFORMER_HPP_
