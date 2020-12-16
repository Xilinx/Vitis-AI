#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV


#include <stdint.h>


#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
//#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/drive_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h" // Ming

namespace caffe {
// Ming
const int kNumData = 1;
const int kNumLabels = 1;
const int kNumBBRegressionCoords = 4;
const int kNumRegressionMasks = 8;

template <typename Dtype>
DriveDataLayer<Dtype>::DriveDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param)
     ,reader_(param) {
    
}

template <typename Dtype>
DriveDataLayer<Dtype>::~DriveDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void DriveDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Ming
  srand(time(0));
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Ming
  DriveDataParameter param = this->layer_param().drive_data_param();
  
  vector<int> top_shape(4, 0);
  int shape[4] = {batch_size, 3, 
                    (int)param.cropped_height(), (int)param.cropped_width()};
  memcpy(&top_shape[0], shape, sizeof(shape));
  
  top[0]->Reshape(top_shape);
  /////////////the next line drivedata does not have, dirty copy
  this->transformed_data_.Reshape(top_shape);
  /////////////
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  
  if (this->output_labels_) { 
      vector<int> label_shape(4,0);
      vector<int> type_shape(4,0);
      int shape[4] = {
          batch_size, kNumRegressionMasks,
          param.tiling_height() * param.label_resolution(),
          param.tiling_width() * param.label_resolution()
      };
      int shape_type[4] = {
          batch_size, 1,
          param.tiling_height() * param.catalog_resolution(),
          param.tiling_width() * param.catalog_resolution()
      };
   memcpy(&label_shape[0], shape, sizeof(shape));
   top[1]->Reshape(label_shape);

   memcpy(&type_shape[0], shape_type, sizeof(shape_type));
   CHECK_GE(top.size(), 3);
   top[2]->Reshape(type_shape);
   for (int i = 0; i < this->PREFETCH_COUNT; ++i) { 
       this->prefetch_[i].label_.Reshape(label_shape);
       this->prefetch_[i].label_1_.Reshape(type_shape);///////////////////////
   }
  }
  //------
  /*
  // Read a data point, and use it to initialize the top blob.
  DrivingData& data = *(reader_.full().peek());
  //use reader_ to initialize blob

  const int grid_dim = data.car_label_resolution();
  const int width = data.car_label_width();
  const int height = data.car_label_height();
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;

  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(3);
  top_shape.push_back(data.car_cropped_height());
  top_shape.push_back(data.car_cropped_width());
  this->transformed_data_.Reshape(top_shape);//this drivedata does not have
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
    top_shape[1] = 8;   // mask(1), bb(4), norm(2), weight(1)
    top_shape[2] = full_label_width;
    top_shape[3] = full_label_height;
    top[1]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(top_shape);
    }
  }
  */
}

//dirty copy from Ming

int Rand() {return rand();}
float rand_float() {return rand()*1.0f / RAND_MAX;}
bool check_cross(float l1, float r1, float l2, float r2) {
    if (r1 < l2 || l1 > r2) { return false; }
    return true;
}
// ------

template <typename Dtype>
bool ReadBoundingBoxLabelToDatum( const DriveData& data, Datum* datum, const int h_off, const int w_off, const float resize, const DriveDataParameter& param, Dtype* label_type, bool can_pass, int bid )
//bool DrivingDataLayer<Dtype>::ReadBoundingBoxLabel(const DrivingData& data, int car_cropped_width, int car_cropped_height, Dtype* label) 
{
  // Ming modifed, this part just read paramter from proto
  bool have_obj = false;
  const int grid_dim = param.label_resolution();//data.car_label_resolution();
  const int width = param.tiling_width();//data.car_label_width();
  const int height = param.tiling_height();//data.car_label_height();
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;
  const Dtype half_shrink_factor = (1-param.shrink_prob_factor()) / 2;//data.car_shrink_factor() / 2;
//  const Dtype whole_factor = data.car_whole_factor();
  const float unrecog_factor = param.unrecognize_factor();
  const Dtype scaling = static_cast<Dtype>(full_label_width) / param.cropped_width();//car_cropped_width;
  
  const int type_label_width = width * param.catalog_resolution();
  const int type_label_height = height * param.catalog_resolution();
  const int type_stride = full_label_width / type_label_width;
  //--------
  /*
  // 1 pixel label, 4 bounding box coordinates, 3 normalization labels.
  const int num_total_labels = 8;
  vector<cv::Mat *> labels;
  for (int i = 0; i < num_total_labels; ++i) {
    labels.push_back(
        new cv::Mat(full_label_height,
                    full_label_width, CV_32F,
                    cv::Scalar(0.0)));
  }
  */
  for (int i = 0; i < data.car_boxes_size(); ++i) {
    // Ming
    if(i != bid) continue; 
    int xmin = data.car_boxes(i).xmin() * resize;
    int ymin = data.car_boxes(i).ymin() * resize;
    int xmax = data.car_boxes(i).xmax() * resize;
    int ymax = data.car_boxes(i).ymax() * resize;
    float ow = xmax - xmin;
    float oh = ymax - ymin;
    //Dtype area1 = (Dtype)(ymax - ymin) * (xmax - xmin);
    xmin = std::min<float>(std::max<float>(0, xmin - w_off), param.cropped_width());
    xmax = std::min<float>(std::max<float>(0, xmax - w_off), param.cropped_width());
    ymin = std::min<float>(std::max<float>(0, ymin - h_off), param.cropped_height());
    ymax = std::min<float>(std::max<float>(0, ymax - h_off), param.cropped_height());
    float w = xmax - xmin;
    float h = ymax - ymin;
    //Dtype area2 = w * h;
    if(w*h < ow*oh*unrecog_factor) continue;
    if (w < 4 || h < 4) {
      // drop boxes that are too small
      continue;
    }
    /*
    if ((area2 / area1) < whole_factor) {
      // drop boxes that out-of-bound
      continue;
    }
    */
    if (std::max(w,h) < param.train_min() || std::max(w,h) > param.train_max()) continue;
    have_obj = true;
  }
  if (can_pass && !have_obj) return false;

  const int num_total_labels = kNumRegressionMasks;

  cv::Mat box_mask(full_label_height, full_label_width, CV_32F, cv::Scalar(-1.0));
  vector<int> itypes(data.car_boxes_size()+1);
  vector<cv::Mat *> labels;
  for (int i = 0; i < num_total_labels; ++i) {
      labels.push_back( new cv::Mat(full_label_height, full_label_width, CV_32F, cv::Scalar(0.0)) );
  }

  for (int i = 0; i < data.car_boxes_size(); ++i) {
      float xmin = data.car_boxes(i).xmin()*resize;
      float ymin = data.car_boxes(i).ymin()*resize;
      float xmax = data.car_boxes(i).xmax()*resize;
      float ymax = data.car_boxes(i).ymax()*resize;
      int ttype = data.car_boxes(i).type();
      itypes[i] = ttype;
      CHECK_LT(ttype+1, param.catalog_number());
      float ow = xmax - xmin;
      float oh = ymax - ymin;
      xmin = std::min<float>(std::max<float>(0, xmin - w_off), param.cropped_width());
      xmax = std::min<float>(std::max<float>(0, xmax - w_off), param.cropped_width());
      ymin = std::min<float>(std::max<float>(0, ymin - h_off), param.cropped_height());
      ymax = std::min<float>(std::max<float>(0, ymax - h_off), param.cropped_height());
      float w = xmax - xmin;
      float h = ymax - ymin;
      if (w*h < ow*oh*unrecog_factor) continue;
      if (w < 4 || h < 4) continue;
      if (std::max(w,h) < param.reco_min() || std::max(w,h) > param.reco_max()) continue;
  //---------------
    // shrink bboxes
    int gxmin = cvFloor((xmin + w * half_shrink_factor) * scaling);
    int gxmax = cvCeil((xmax - w * half_shrink_factor) * scaling);
    int gymin = cvFloor((ymin + h * half_shrink_factor) * scaling);
    int gymax = cvCeil((ymax - h * half_shrink_factor) * scaling);

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
    //Ming
    cv::Scalar idcolor(i);
    cv::rectangle(box_mask, r, idcolor, -1);
    if (param.use_mask()) {
        caffe::CarBoundingBox_Ming box = data.car_boxes(i);
    }
    // -------------
  }
  // Ming, dirty copy
  datum->set_channels(num_total_labels);
  datum->set_height(full_label_height);
  datum->set_width(full_label_width);
  datum->set_label(0);
  datum->clear_data();
  datum->clear_float_data();
  // ------------

  int total_num_pixels = 0;
  for (int y = 0; y < full_label_height; ++y) {
    for (int x = 0; x < full_label_width; ++x) {
        // Ming
        float &val = box_mask.at<float>(y,x);
        if (val != -1) {
            float valo = labels[0]->at<float>(y, x);
            if (valo == 0) val = -1;
            if (val != -1) total_num_pixels++;
        // -----------
      //if (labels[num_total_labels - 1]->at<Dtype>(y, x) == 1.0) {
      //  total_num_pixels++;
      //}
        }
    }
  }

  if (total_num_pixels != 0) {
    Dtype reweight_value = 1.0 / total_num_pixels;
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        // Ming
        if(box_mask.at<float>(y,x) == -1){
        //if (labels[num_total_labels - 1]->at<Dtype>(y, x) == 1.0) {
          labels[num_total_labels - 1]->at<Dtype>(y, x) = 0.0f;//reweight_value;
        }
        else
          labels[num_total_labels - 1]->at<Dtype>(y, x) = reweight_value;
        // --------
      }
    }
  }

  for (int m = 0; m < num_total_labels; ++m) {
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        Dtype adjustment = 0;
        Dtype val = labels[m]->at<Dtype>(y, x);
        if (m == 0 || m > 4) {
        // Ming
            if (m == 0 && param.use_mask() && val == 1.0f) {
                val = (box_mask.at<float>(y,x)==-1)?0.0f:1.0f;
            }
          // do nothing
        } else if (labels[0]->at<Dtype>(y, x) == 0.0) {
          // do nothing
        } else if (m % 2 == 1) {
          // x coordinate
          adjustment = x / scaling;
        } else {
          // y coordinate
          adjustment = y / scaling;
        }
        //*label = val - adjustment;
        //label++;
        datum->add_float_data(val - adjustment);
        // ----------------
      }
    }
  }
  // Ming, dirty copy
  //float *ptr = label_type;
  Dtype *ptr = label_type;
  for (int y = 0; y < type_label_height; ++y) {
      for (int x = 0; x < type_label_width; ++x) {
          int id = (int)(box_mask.at<float>(y*type_stride, x*type_stride));
          if (id>=0 && id<itypes.size()) {
              *ptr = itypes[id];
          }
          else if (id == -1) {
              *ptr = 0;
          }
          else {
              LOG(ERROR) << "invalid id " << id << " " << y << ' ' << x << ' ' << (box_mask.at<float>(y*type_stride, x*type_stride)) << ' ' << data.car_img_source();
          }
          ptr++;
      }
  }

  CHECK_EQ(datum->float_data_size(), num_total_labels * full_label_height * full_label_width);
  // --------------

  for (int i = 0; i < num_total_labels; ++i) {
    delete labels[i];
  }

  return have_obj;//true;
}

// Ming dirty copy
float get_box_size(const caffe::CarBoundingBox_Ming &box) {
    float xmin = box.xmin();
    float ymin = box.ymin();
    float xmax = box.xmax();
    float ymax = box.ymax();
    return std::max(xmax-xmin, ymax-ymin);
}
// --------------

// This function is called on prefetch thread
template<typename Dtype>
void DriveDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());// Ming keep from original

  // Ming, VPGnet use param to initialize, here use data1
  DriveDataParameter param = this->layer_param_.drive_data_param();
  // --------

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Ming
  vector<int> top_shape(4,0);
  int shape[4] = {batch_size, 3, (int)param.cropped_height(), (int)param.cropped_width()};
  memcpy(&top_shape[0], shape, sizeof(shape));
  // ----------
  /*
  DrivingData& data1 = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(3);
  top_shape.push_back(data1.car_cropped_height());
  top_shape.push_back(data1.car_cropped_width());
  */
  // this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  // Ming
  const Dtype* data_mean_c = this->data_transformer_->data_mean_.cpu_data();
  vector<Dtype> v_mean(data_mean_c, data_mean_c+this->data_transformer_->data_mean_.count());
  Dtype *data_mean = &v_mean[0];
  vector<Dtype*> top_labels;
  Dtype *label_type = NULL;
  //Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    //top_label = batch->label_.mutable_cpu_data();
      Dtype *top_label = batch->label_.mutable_cpu_data();
      top_labels.push_back(top_label); 
      label_type = batch->label_1_.mutable_cpu_data(); ////////////////
  }
  
  const int crop_num = this->layer_param().drive_data_param().crop_num();
  int type_label_strip = param.tiling_height()*param.tiling_width() * param.catalog_resolution()*param.catalog_resolution();
  bool need_new_data = true;
  int bid = 0;
  // ------------

  DriveData data;
  //string *raw_data = NULL;
  DriveData *raw_data = NULL;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a data
    //DrivingData& data = *(reader_.full().pop("Waiting for data"));

    // Ming
    if (need_new_data) {
        data = *(raw_data = reader_.full().pop("Waiting for data"));
        //data.ParseFromString(*(raw_data = this->reader_.full().pop("Waiting for data")));
        //LOG(INFO) << "--------------- " << data.car_image_datum().encoded() ;
        bid = 0;
        need_new_data = false;
    }else {
        bid ++;
    }
    if (item_id+1 == batch_size) need_new_data = true;
    // -----------
    read_time += timer.MicroSeconds();
    timer.Start();

    // Apply data transformations (mirror, scale, crop...)
    Dtype *t_lable_type = NULL;
    if (label_type != NULL) {
        t_lable_type = label_type + type_label_strip * item_id;
        caffe_set(type_label_strip, (Dtype)0, t_lable_type);
    }
    vector<Datum> label_datums(kNumLabels);
    const Datum& img_datum = data.car_image_datum();
    const string& img_datum_data = img_datum.data();
    bool can_pass = rand_float() > this->layer_param().drive_data_param().random_crop_ratio();
    int cheight = param.cropped_height();
    int cwidth = param.cropped_width();
    int channal = img_datum.channels();
    int hid = bid/crop_num;

    float bsize = get_box_size(data.car_boxes(hid));
    float rmax = std::min(param.train_max() / bsize, param.resize_max());
    float rmin = std::max(param.train_min() / bsize, param.resize_min());
    try_again:
        float resize = this->layer_param().drive_data_param().resize();
        int h_off, w_off;
        if (resize < 0) {
             if (rmax <= rmin || !can_pass) {
                 can_pass = false;
                 resize = rand_float() * (param.resize_max()-param.resize_min()) + param.resize_min();
                 int rheight = (int)(img_datum.height() * resize);
                 int rwidth = (int)(img_datum.height() * resize);
                 h_off = rheight <= cheight ? 0 : Rand() % (rheight - cheight);
                 w_off = rwidth <= cwidth ? 0 : Rand() % (rwidth - cwidth); 
             } else {
                int h_max, h_min, w_max, w_min;
                resize = rand_float() * (rmax-rmin) + rmin;
                LOG(INFO) << "resize " << resize << ' ' << item_id;
                h_min = data.car_boxes(hid).ymin()*resize - param.cropped_height();
                h_max = data.car_boxes(hid).ymax()*resize;
                w_min = data.car_boxes(hid).xmin()*resize - param.cropped_width();
                w_max = data.car_boxes(hid).xmax()*resize;
                w_off = (Rand() % (w_max-w_min) + w_min);
                h_off = (Rand() % (h_max-h_min) + h_min);
                LOG(INFO) << "h_min : " << h_min << " h_max : " << h_max << " w_min : " << w_min << "w_max: " << w_max << " w_off : " << w_off << "h_off" << h_off; 
             }
        }else{
            int rheight = (int)(img_datum.height() * resize);
            int rwidth = (int)(img_datum.height() * resize);
            h_off = rheight <= cheight ? 0 : Rand() % (rheight - cheight);
            w_off = rwidth <= cwidth ? 0 : Rand() % (rwidth - cwidth);
        }
    //---------------------

    // Ming think this part need to be kept
    //int offset = batch->data_.offset(item_id);
    //this->transformed_data_.set_cpu_data(top_data + offset);
    //this->data_transformer_->Transform(data, &(this->transformed_data_));

    // Generate label.
    //int offset1 = batch->label_.offset(item_id);
    // ----------
    if (this->output_labels_) {
    // Ming
        //ReadBoundingBoxLabel(data, top_shape[3], top_shape[2], top_label + offset1);
        //if(!ReadBoundingBoxLabelToDatum(data, &label_datums[0],h_off, w_off, resize, param,(float*)t_lable_type, can_pass, hid)){
        if(!ReadBoundingBoxLabelToDatum(data, &label_datums[0],h_off, w_off, resize, param, t_lable_type, can_pass, hid)){
            if (can_pass){ goto try_again; }
        }
    }
    if (bid+1 >= crop_num*data.car_boxes_size()){
        need_new_data = true;
    }
    cv::Mat_<Dtype> mean_img(cheight, cwidth);
    Dtype* itop_data = top_data + item_id*channal*cheight*cwidth;
    float mat[] = { 1.f*resize, 0.f, (float)-w_off, 0.f, 1.f*resize, (float)-h_off};
    cv::Mat_<float> M(2,3, mat); 
    for (int c=0; c<img_datum.channels(); c++) {
        cv::Mat_<Dtype> crop_img(cheight, cwidth, itop_data + c*cheight*cwidth);
        cv::Mat_<Dtype> pmean_img(img_datum.height(), img_datum.width(), data_mean + c*img_datum.height()*img_datum.width());
        cv::Mat p_img(img_datum.height(), img_datum.width(), CV_8U, ((uint8_t*)&(img_datum_data[0])) + c*img_datum.height()*img_datum.width());
        p_img.convertTo(p_img, crop_img.type()); 
        cv::warpAffine(pmean_img, mean_img, M, mean_img.size(), cv::INTER_CUBIC);
        cv::warpAffine(p_img, crop_img, M, crop_img.size(), cv::INTER_CUBIC);
        crop_img -= mean_img;
        crop_img *= this->layer_param().drive_data_param().scale();
    }
    if (this->output_labels_) { 
        for (int i = 0; i < kNumLabels; ++i) { 
            for (int c = 0; c < label_datums[i].channels(); ++c) {
                for (int h = 0; h < label_datums[i].height(); ++h) {
                    for (int w = 0; w < label_datums[i].width(); ++w) {
                        const int top_index = ((item_id * label_datums[i].channels() + c)* label_datums[i].height() + h) * label_datums[i].width() + w;
                        const int data_index = (c * label_datums[i].height() + h) * label_datums[i].width() + w;
                        float label_datum_elem = label_datums[i].float_data(data_index);
                        top_labels[i][top_index] = static_cast<Dtype>(label_datum_elem);
                    }
                }
            }
        }
    }

    // save image and label for debug
    /*static int count = 0;
    if (count++ < 100) {
	char fileName[100];
	sprintf(fileName, "temp/image%03d.temp", count);
        FILE* fid = fopen(fileName, "wb");
        fwrite(top_data + offset, sizeof(float), top_shape[1] * top_shape[2] * top_shape[3], fid);
        fclose(fid);

	sprintf(fileName, "temp/label%03d.temp", count);
        fid = fopen(fileName, "wb");
        int size = top_shape[2] / 4;
        fwrite(top_label + offset1, sizeof(float), size * size * 8, fid);
        fclose(fid);
    }*/

    trans_time += timer.MicroSeconds();
    if (need_new_data) {
        reader_.free().push(const_cast<DriveData*>(raw_data));
        //this->reader_.free().push(const_cast<string*>(raw_data));
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}
//template bool ReadBoundingBoxLabelToDatum<float>( const DriveData& data, Datum* datum, const int h_off, const int w_off, const float resize, const DriveDataParameter& param, float* label_type, bool can_pass, int bid );
//template bool ReadBoundingBoxLabelToDatum<double>( const DriveData& data, Datum* datum, const int h_off, const int w_off, const float resize, const DriveDataParameter& param, double* label_type, bool can_pass, int bid );

INSTANTIATE_CLASS(DriveDataLayer);
REGISTER_LAYER_CLASS(DriveData);

}  // namespace caffe
