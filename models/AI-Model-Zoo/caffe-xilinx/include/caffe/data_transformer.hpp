#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <string>
#include <vector>

#include <google/protobuf/repeated_field.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

using google::protobuf::RepeatedPtrField;

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param, Phase phase);
  virtual ~DataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

#ifndef CPU_ONLY
  void TransformGPU(int N, int C, int H, int W,
              const Dtype *in, Dtype *out, int *);
#endif
  void Copy(const Datum& datum, Dtype *data);
  void Copy(const cv::Mat& datum, Dtype *data);
  void CopyPtrEntry(string* str, Dtype* transformed_ptr,
                    bool output_labels, Dtype *label,
                    BlockingQueue<string*>* free);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob);
  void Transform(const Datum& datum, const Datum& seg_datum, 
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob);
  void Transform(DrivingData& data, Blob<Dtype>* transformed_blob);
  void Transform(DriveData& data, Blob<Dtype>* transformed_blob);
  void Transform(DirectRegressionData& data, Blob<Dtype>* transformed_blob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param rand1
   *    Random value (0,RAND_MAX+1]
   * @param rand2
   *    Random value (0,RAND_MAX+1]
   * @param rand3
   *    Random value (0,RAND_MAX+1]
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
    void TransformPtrEntry(string* str, Dtype* transformed_ptr,
                           int rand1, int rand2, int rand3,
                           bool output_labels, Dtype *label,
                           BlockingQueue<string*>* free);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Datum.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<Datum> & datum_vector,
                Blob<Dtype>* transformed_blob);
  void Transform(const vector<Datum> & datum_vector, const vector<Datum> & seg_datum_vector,
                Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the annotated data.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See annotated_data_layer.cpp for an example.
   * @param transformed_anno_vec
   *    This is destination annotation.
   */
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                 vector<AnnotationGroup>* transformed_anno_vec);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the annotated data. (with lane annotated data)
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See annotated_data_layer.cpp for an example.
   * @param transformed_anno_vec
   *    This is destination annotation.
   */
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
		 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec_2);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec_2);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec_2,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec,
                 RepeatedPtrField<AnnotationGroup>* transformed_anno_vec_2,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 vector<AnnotationGroup>* transformed_anno_vec_2,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 vector<AnnotationGroup>* transformed_anno_vec_2,
                 bool* do_mirror);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 vector<AnnotationGroup>* transformed_anno_vec_2);
  void Transform(const AnnotatedDatum& anno_datum,
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                 vector<AnnotationGroup>* transformed_anno_vec,
                 vector<AnnotationGroup>* transformed_anno_vec_2);

  /**
   * @brief Transform the annotation according to the transformation applied
   * to the datum.
   *
   * @param anno_datum
   *    AnnotatedDatum containing the data and annotation to be transformed.
   * @param do_resize
   *    If true, resize the annotation accordingly before crop.
   * @param crop_bbox
   *    The cropped region applied to anno_datum.datum()
   * @param do_mirror
   *    If true, meaning the datum has mirrored.
   * @param transformed_anno_group_all
   *    Stores all transformed AnnotationGroup.
   */
  void TransformAnnotation(
      const AnnotatedDatum& anno_datum, const bool do_resize,
      const NormalizedBBox& crop_bbox, const bool do_mirror,
      RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all);
  void TransformAnnotation(
      const AnnotatedDatum& anno_datum, const bool do_resize,
      const NormalizedBBox& crop_bbox, const bool do_mirror,
      RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all,
      RepeatedPtrField<AnnotationGroup>* transformed_anno_group_all_2);

  /**
   * @brief Crops the datum according to bbox.
   */
  void CropImage(const Datum& datum, const NormalizedBBox& bbox,
                 Datum* crop_datum);

  /**
   * @brief Crops the datum and AnnotationGroup according to bbox.
   */
  void CropImage(const AnnotatedDatum& anno_datum, const NormalizedBBox& bbox,
                 AnnotatedDatum* cropped_anno_datum);
  void CropImage(const AnnotatedDatum& anno_datum, const NormalizedBBox& bbox,
                 AnnotatedDatum* cropped_anno_datum, bool has_seg_datum);

  /**
   * @brief Expand the datum.
   */
  void ExpandImage(const Datum& datum, const float expand_ratio,
                   NormalizedBBox* expand_bbox, Datum* expanded_datum);

  /**
   * @brief Expand the datum and adjust AnnotationGroup.
   */
  void ExpandImage(const AnnotatedDatum& anno_datum,
                   AnnotatedDatum* expanded_anno_datum);

  /**
   * @brief Apply distortion to the datum.
   */
  void DistortImage(const Datum& datum, Datum* distort_datum);

#ifdef USE_OPENCV
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a vector of Mat.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See memory_layer.cpp for an example.
   */
  void Transform(const vector<cv::Mat> & mat_vector,
                Blob<Dtype>* transformed_blob);
  void Transform(const vector<cv::Mat> & mat_vector, const vector<cv::Mat> & seg_mat_vector,
                Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   */
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror);
  void Transform(const cv::Mat& cv_img, const cv::Mat& cv_segmap, Blob<Dtype>* transformed_blob,
                 Blob<Dtype>* transformed_segblob, NormalizedBBox* crop_bbox, bool* do_mirror);
  void Transform(const cv::Mat& cv_img, Blob<Dtype>* transformed_blob);
  void Transform(const cv::Mat& cv_img, const cv::Mat& cv_segmap, 
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob);

  void TransformImgAndSeg(const std::vector<cv::Mat>& cv_img_seg,
    Blob<Dtype>* transformed_data_blob, Blob<Dtype>* transformed_label_blob,
    const int ignore_label);
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to a cv::Mat
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See image_data_layer.cpp for an example.
   * @param rand1
   *    Random value (0,RAND_MAX+1]
   * @param rand2
   *    Random value (0,RAND_MAX+1]
   * @param rand3
   *    Random value (0,RAND_MAX+1]
   */
  void TransformPtr(const cv::Mat& cv_img, Dtype* transformed_ptr,
                    int rand1, int rand2, int rand3);

  /**
   * @brief Crops img according to bbox.
   */
  void CropImage(const cv::Mat& img, const NormalizedBBox& bbox,
                 cv::Mat* crop_img);

  /**
   * @brief Expand img to include mean value as background.
   */
  void ExpandImage(const cv::Mat& img, const float expand_ratio,
                   NormalizedBBox* expand_bbox, cv::Mat* expand_img);

  void TransformInv(const Blob<Dtype>* blob, vector<cv::Mat>* cv_imgs);
  void TransformInv(const Dtype* data, cv::Mat* cv_img, const int height,
                    const int width, const int channels);

  void Transform(cv::Mat& cv_img, Blob<Dtype>* transformed_blob,
      DrivingData& data);
      
  void Transform(cv::Mat& cv_img, Blob<Dtype>* transformed_blob, DriveData& data);

  void Transform(cv::Mat& cv_img, Blob<Dtype>* transformed_blob,
      DirectRegressionData& data);
 
#endif  // USE_OPENCV

  /**
   * @brief Applies the same transformation defined in the data layer's
   * transform_param block to all the num images in a input_blob.
   *
   * @param input_blob
   *    A Blob containing the data to be transformed. It applies the same
   *    transformation to all the num images in the blob.
   * @param transformed_blob
   *    This is destination blob, it will contain as many images as the
   *    input blob. It can be part of top blob's data.
   */
  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const Datum& datum, bool use_gpu = false);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const vector<Datum> & datum_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param mat_vector
   *    A vector of Mat containing the data to be transformed.
   */
#ifdef USE_OPENCV
  vector<int> InferBlobShape(const vector<cv::Mat> & mat_vector);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param cv_img
   *    cv::Mat containing the data to be transformed.
   */
  vector<int> InferBlobShape(const cv::Mat& cv_img, bool use_gpu = false);
#endif  // USE_OPENCV

  Blob<Dtype> data_mean_;
 protected:
  void TransformGPU(const Datum& datum, Dtype* transformed_data);
  void Transform(const Datum& datum, Dtype* transformed_data);
  void Transform(const Datum& datum, const Datum& seg_datum, 
                 Dtype* transformed_data, Dtype* transformed_segmap);
  // Transform and return the transformation information.
  void Transform(const Datum& datum, Dtype* transformed_data,
                 NormalizedBBox* crop_bbox, bool* do_mirror);
  void Transform(const Datum& datum, const Datum& seg_datum, 
                 Dtype* transformed_data, Dtype* transformed_segmap,
                 NormalizedBBox* crop_bbox, bool* do_mirror);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data and return transform information.
   */
  void Transform(const Datum& datum, Blob<Dtype>* transformed_blob,
                 NormalizedBBox* crop_bbox, bool* do_mirror);
  void Transform(const Datum& datum, const Datum& seg_datum,
                 Blob<Dtype>* transformed_blob, Blob<Dtype>* transformed_segblob,
                 NormalizedBBox* crop_bbox, bool* do_mirror);

  // Tranformation parameters
  TransformationParameter param_;
  void TransformPtrInt(Datum* datum, Dtype* transformed_data,
                       int rand1, int rand2, int rand3);

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  vector<Dtype> mean_values_;
  Dtype *mean_values_gpu_ptr_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
