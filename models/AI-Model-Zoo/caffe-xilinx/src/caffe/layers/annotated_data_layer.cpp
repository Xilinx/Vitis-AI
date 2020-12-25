#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"

namespace caffe {
const int kNumRegressionMasks = 8;

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }

  // Read a data point, and use it to initialize the top blob.
  AnnotatedDatum& anno_datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        LOG(INFO) << "annotation group size: " << anno_datum.annotation_group_size();
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        if (anno_datum.label_ori()) label_shape[3] = 10;
        else label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
    has_seg_type_ = anno_data_param.has_seg_label();
    //vector<int> seg_label_shape(4,1);
    if (has_seg_type_) {
      vector<int> segmap_shape =
          this->data_transformer_->InferBlobShape(anno_datum.seg_datum());
      this->transformed_segmap_.Reshape(segmap_shape);
      segmap_shape[0] = batch_size;
      //seg_label_shape[1] = 1;
      //seg_label_shape[2] = top_shape[2];
      //seg_label_shape[3] = top_shape[3];
      top[2]->Reshape(segmap_shape);
      for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].segmap_.Reshape(segmap_shape);
      }
      //this->transformed_segmap_.Reshape(seg_label_shape);
      LOG(INFO) << "output segmap size: " << top[2]->num() << ","
      << top[2]->channels() << "," << top[2]->height() << ","
      << top[2]->width();
    }
    has_lane_type_ = anno_data_param.has_lane_label();
    if (has_lane_type_) {
      vector<int> lane_label_shape(4,0);
      vector<int> lane_type_shape(4,0);
      int _label_shape[4] = {
	batch_size, kNumRegressionMasks,
	anno_data_param.tiling_height() * anno_data_param.label_resolution(),
	anno_data_param.tiling_width() * anno_data_param.label_resolution()
      };
      int _type_shape[4] = {
	batch_size, 1,
	anno_data_param.tiling_height() * anno_data_param.catalog_resolution(),
	anno_data_param.tiling_width() * anno_data_param.catalog_resolution()
      };
      memcpy(&lane_label_shape[0], _label_shape, sizeof(_label_shape));
      top[3]->Reshape(lane_label_shape);

      memcpy(&lane_type_shape[0], _type_shape, sizeof(_type_shape));
      top[4]->Reshape(lane_type_shape);
      for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	this->prefetch_[i].lanelabel_.Reshape(lane_label_shape);
	this->prefetch_[i].lanemap_.Reshape(lane_type_shape);
      }
      LOG(INFO) << "output lane label map size: " << top[3]->num() << ","
		<< top[3]->channels() << "," << top[3]->height() << ","
		<< top[3]->width();
      LOG(INFO) << "output lane type map size: " << top[4]->num() << ","
		<< top[4]->channels() << "," << top[4]->height() << ","
		<< top[4]->width();
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  //LOG(INFO) << has_seg_type_;
  if (has_seg_type_) {
    CHECK(this->transformed_segmap_.count());
  }

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  AnnotatedDatum& anno_datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  //vector<int> top_seg_shape;
  if (has_seg_type_) {
    vector<int> top_seg_shape = this->data_transformer_->InferBlobShape(anno_datum.seg_datum());
    this->transformed_segmap_.Reshape(top_seg_shape);
    top_seg_shape[0] = batch_size;
    batch->segmap_.Reshape(top_seg_shape);
  }

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_segmap = NULL;
  if (has_seg_type_) {
    top_segmap = batch->segmap_.mutable_cpu_data();
  }
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  //Dtype* top_segmap = NULL;
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
    //if (!has_seg_type_) {
    //  top_segmap = batch->segmap_.mutable_cpu_data();
    //}
  }
  Dtype* top_lane_label = NULL;
  Dtype* top_lane_map = NULL;
  if (this->output_labels_ && has_lane_type_) {
    top_lane_label = batch->lanelabel_.mutable_cpu_data();
    top_lane_map = batch->lanemap_.mutable_cpu_data();
  }


  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  map<int, vector<AnnotationGroup> > lane_anno;
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    AnnotatedDatum distort_datum;
    AnnotatedDatum* expand_datum = NULL;
    if (transform_param.has_distort_param()) {
      distort_datum.CopyFrom(anno_datum);
      this->data_transformer_->DistortImage(anno_datum.datum(),
                                            distort_datum.mutable_datum());
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(distort_datum, expand_datum);
      } else {
        expand_datum = &distort_datum;
      }
    } else {
      if (transform_param.has_expand_param()) {
        expand_datum = new AnnotatedDatum();
        this->data_transformer_->ExpandImage(anno_datum, expand_datum);
      } else {
        expand_datum = &anno_datum;
      }
    }
    AnnotatedDatum* sampled_datum = NULL;
    bool has_sampled = false;
    if (batch_samplers_.size() > 0) {
      // Generate sampled bboxes from expand_datum.
      vector<NormalizedBBox> sampled_bboxes;
      GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
      if (sampled_bboxes.size() > 0) {
        // Randomly pick a sampled bbox and crop the expand_datum.
        int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
        sampled_datum = new AnnotatedDatum();
        if (!has_seg_type_ and !has_lane_type_) {
          this->data_transformer_->CropImage(*expand_datum,
                                             sampled_bboxes[rand_idx],
                                             sampled_datum);
        } else {
          this->data_transformer_->CropImage(*expand_datum, 
                                             sampled_bboxes[rand_idx],
                                             sampled_datum,
                                             has_seg_type_);
        }
        has_sampled = true;
      } else {
        sampled_datum = expand_datum;
      }
    } else {
      sampled_datum = expand_datum;
    }
    CHECK(sampled_datum != NULL);
    timer.Start();
    vector<int> shape =
        this->data_transformer_->InferBlobShape(sampled_datum->datum());
    vector<int> seg_shape;
    if (has_seg_type_) {
      seg_shape = this->data_transformer_->InferBlobShape(sampled_datum->seg_datum());
    }
    if (transform_param.has_resize_param()) {
      if (transform_param.resize_param().resize_mode() ==
          ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
        this->transformed_data_.Reshape(shape);
        batch->data_.Reshape(shape);
        top_data = batch->data_.mutable_cpu_data();
        if (has_seg_type_) {
          this->transformed_segmap_.Reshape(seg_shape);
          batch->segmap_.Reshape(seg_shape);
          top_segmap = batch->segmap_.mutable_cpu_data();
        }
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
        /*if (has_seg_type_) {
          CHECK(std::equal(top_seg_shape.begin() + 1, top_seg_shape.begin() + 4,
                seg_shape.begin() + 1));
        }*/
      }
    } else {
      CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
            shape.begin() + 1));
    }
    // Apply data transformations (mirror, scale, crop...)
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    if (has_seg_type_) {
      int seg_offset = batch->segmap_.offset(item_id);
      this->transformed_segmap_.set_cpu_data(top_segmap + seg_offset);
    }
    vector<AnnotationGroup> transformed_anno_vec;
    vector<AnnotationGroup> transformed_anno_vec_2; // for lane boxes
    if (this->output_labels_) {
      if (has_anno_type_) {
        // Make sure all data have same annotation type.
        CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
        if (anno_data_param.has_anno_type()) {
          sampled_datum->set_type(anno_type_);
        } else {
          CHECK_EQ(anno_type_, sampled_datum->type()) <<
              "Different AnnotationType.";
        }
        // Transform datum and annotation_group at the same time
        transformed_anno_vec.clear();
        transformed_anno_vec_2.clear();
        if (!has_seg_type_) {
	  if (has_lane_type_) {
	    this->data_transformer_->Transform(*sampled_datum,
					       &(this->transformed_data_),
					       &transformed_anno_vec,
					       &transformed_anno_vec_2);
	  } else {
	    this->data_transformer_->Transform(*sampled_datum,
					       &(this->transformed_data_),
					       &transformed_anno_vec);
	  }
        } else {
	  if (has_lane_type_) {
	    this->data_transformer_->Transform(*sampled_datum,
					       &(this->transformed_data_),
					       &(this->transformed_segmap_),
					       &transformed_anno_vec,
					       &transformed_anno_vec_2);
	  } else {
	    this->data_transformer_->Transform(*sampled_datum,
					       &(this->transformed_data_),
					       &(this->transformed_segmap_),
					       &transformed_anno_vec);
	  }
        }
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
          // Count the number of bboxes.
          for (int g = 0; g < transformed_anno_vec.size(); ++g) {
            num_bboxes += transformed_anno_vec[g].annotation_size();
          }
        } else {
          LOG(FATAL) << "Unknown annotation type.";
        }
        all_anno[item_id] = transformed_anno_vec;
	if (has_lane_type_) {
	  lane_anno[item_id] = transformed_anno_vec_2;
	}
      } else {
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
        // Otherwise, store the label from datum.
        CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
        top_label[item_id] = sampled_datum->datum().label();
      }
    } else {
      this->data_transformer_->Transform(sampled_datum->datum(),
                                         &(this->transformed_data_));
    }
    // clear memory
    if (has_sampled) {
      delete sampled_datum;
    }
    if (transform_param.has_expand_param()) {
      delete expand_datum;
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
  }
  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      if(anno_datum.label_ori()) label_shape[3] = 10;
      else label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        if(anno_datum.label_ori()) caffe_set<Dtype>(10, -1, batch->label_.mutable_cpu_data());
        else caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              if(anno_datum.label_ori()){
              top_label[idx++] = bbox.angle_cos();
              top_label[idx++] = bbox.angle_sin();
              }
              top_label[idx++] = bbox.difficult();
              //LOG(INFO) << "label:" << bbox.xmin()<< " " << bbox.ymin() << " " << bbox.xmax() << " " << bbox.ymax();
              //LOG(INFO) << "label:" << bbox.angle_cos()<< " " << bbox.angle_sin();
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  if (this->output_labels_ && has_lane_type_) {
    // Generate lane label/type map
    const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
    const Dtype half_shrink_factor = (1-anno_data_param.shrink_prob_factor()) / 2;
    const int full_label_width = anno_data_param.tiling_width() * anno_data_param.label_resolution();
    const int full_label_height = anno_data_param.tiling_height() * anno_data_param.label_resolution();
    const int type_label_width = anno_data_param.tiling_width() * anno_data_param.catalog_resolution();
    const int type_label_height = anno_data_param.tiling_height() * anno_data_param.catalog_resolution();
    const int type_stride = full_label_width / type_label_width;

    int img_h = transform_param.resize_param().height();
    int img_w = transform_param.resize_param().width();
    const Dtype scaling = static_cast<Dtype>(full_label_width) / img_w;

    Dtype *label_ptr = top_lane_label;
    Dtype *type_ptr = top_lane_map;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      const int num_total_labels = kNumRegressionMasks;
      cv::Mat box_mask(full_label_height, full_label_width, CV_32F, cv::Scalar(-1.0));
      const vector<AnnotationGroup>& anno_vec = lane_anno[item_id];
      int num_lane_bboxes = 0;
      for (int g = 0; g < anno_vec.size(); ++g) {
	num_lane_bboxes += anno_vec[g].annotation_size();
      }
      vector<int> itypes(num_lane_bboxes);
      vector<cv::Mat *> labels;
      for (int i = 0; i < num_total_labels; ++i) {
	labels.push_back( new cv::Mat(full_label_height, full_label_width, CV_32F, cv::Scalar(0.0)) );
      }

      int box_id = 0;
      for (int g = 0; g < anno_vec.size(); ++g) {
	const AnnotationGroup& anno_group = anno_vec[g];
	for (int a = 0; a < anno_group.annotation_size(); ++a) {
	  const Annotation& anno = anno_group.annotation(a);
	  const NormalizedBBox& bbox = anno.bbox();
	  int ttype = anno_group.group_label();
	  itypes[box_id] = ttype;
	  float xmin = bbox.xmin() * img_w;
	  float xmax = bbox.xmax() * img_w;
	  float ymin = bbox.ymin() * img_h;
	  float ymax = bbox.ymax() * img_h;
	  float w = xmax - xmin;
	  float h = ymax - ymin;
	  int gxmin = cvFloor((xmin + w * half_shrink_factor) * scaling);
	  int gxmax = cvCeil((xmax + w * half_shrink_factor) * scaling);
	  int gymin = cvFloor((ymin + h * half_shrink_factor) * scaling);
	  int gymax = cvCeil((ymax + h * half_shrink_factor) * scaling);
	  CHECK_LE(gxmin, gxmax);
	  CHECK_LE(gymin, gymax);
	  if (gxmin >= full_label_width) {
	    gxmin = full_label_width - 1;
	  }
	  if (gymin >= full_label_height) {
	    gymin = full_label_height - 1;
	  }
	  CHECK_LE(0, gxmin);
	  CHECK_LE(0, gymin);
	  CHECK_LE(gxmax, full_label_width);
	  CHECK_LE(gymax, full_label_height);
	  if (gxmin == gxmax) {
	    if (gxmax < full_label_width - 1) {
	      gxmax++;
	    } else if (gxmin > 0) {
	      gxmin--;
	    }
	  }
	  if (gymin == gymax) {
	    if (gymax < full_label_height - 1) {
	      gymax++;
	    } else if (gymin > 0) {
	      gymin--;
	    }
	  }
	  CHECK_LT(gxmin, gxmax);
	  CHECK_LT(gymin, gymax);
	  if (gxmax == full_label_width) {
	    gxmax--;
	  }
	  if (gymax == full_label_height) {
	    gymax--;
	  }
	  cv::Rect r(gxmin, gymin, gxmax - gxmin + 1, gymax - gymin + 1);

	  Dtype flabels[num_total_labels] = {(Dtype)1.0, (Dtype)xmin, (Dtype)ymin, (Dtype)xmax, (Dtype)ymax, 
					     (Dtype)1.0 / w, (Dtype)1.0 / h, (Dtype)1.0};
	  for (int j = 0; j < num_total_labels; ++j) {
	    cv::Mat roi(*labels[j], r);
	    roi = cv::Scalar(flabels[j]);
	  }
	  cv::Scalar idcolor(box_id);
	  cv::rectangle(box_mask, r, idcolor, -1); // thickness=-1 filled
	  box_id ++;
	}
      }

      for (int m = 0; m < num_total_labels; ++m) {
	for (int y = 0; y < full_label_height; ++y) {
	  for (int x = 0; x < full_label_width; ++x) {
	    Dtype adjustment = 0;
	    Dtype val = labels[m]->at<Dtype>(y, x);
	    if (m == 0 || m > 4) {
	      // if (m == 0 && param.use_mask() && val == 1.0f) {
	      if (m == 0 && val == 1.0f) {
                val = (box_mask.at<float>(y,x)==-1)?0.0f:1.0f;
	      }
	    } else if (labels[0]->at<Dtype>(y, x) == 0.0) {
	    } else if (m % 2 == 1) {
	      // x coordinate
	      adjustment = x / scaling;
	    } else {
	      // y coordinate
	      adjustment = y / scaling;
	    }
	    *label_ptr = val - adjustment;
	    label_ptr++;
	    // int label_ind = (((item_id * num_total_labels) + m) * full_label_height + y) * full_label_width + x
	    // label_ptr[label_ind] = val - adjustment;
	  }
	}
      }
      for (int y = 0; y < type_label_height; ++y) {
	for (int x = 0; x < type_label_width; ++x) {
          int id = (int)(box_mask.at<float>(y*type_stride, x*type_stride));
          if (id >= 0 && id < itypes.size()) {
	    *type_ptr = itypes[id];
          }
          else if (id == -1) {
	    *type_ptr = 0;
          }
          else {
	    LOG(ERROR) << "invalid id " << id << " " << y << " " << x << " " << (box_mask.at<float>(y*type_stride, x*type_stride));
          }
          type_ptr++;
	}
      }
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
