#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
#endif  //USE OPENCV

#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "caffe/pe_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"


using namespace std;


namespace caffe {

    template<typename Dtype>
    PEDataTransformer<Dtype>::PEDataTransformer(const PETransformationParameter& param, Phase phase) : param_(param), phase_(phase) {
        // check if we want to use mean_file
        if (param_.has_mean_file()) {
            CHECK_EQ(param_.mean_value_size(), 0) <<
            "Cannot specify mean_file and mean_value at the same time";
            const string& mean_file = param.mean_file();
            if (Caffe::root_solver()) {
                LOG(INFO) << "Loading mean file from: " << mean_file;
            }
            BlobProto blob_proto;
            ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
            data_mean_.FromProto(blob_proto);
        }
        // check if we want to use mean_value
        if (param_.mean_value_size() > 0) {
            CHECK(param_.has_mean_file() == false) <<
            "Cannot specify mean_file and mean_value at the same time";
            for (int c = 0; c < param_.mean_value_size(); ++c) {
                mean_values_.push_back(param_.mean_value(c));
            }
        }
        LOG(INFO) << "PEDataTransformer constructor done.";
        np_in_lmdb_ = param_.np_in_lmdb();
        np_ = param_.num_parts();
        single_ = param_.single();
    }

    template <typename Dtype>
    void PEDataTransformer<Dtype>::InitRand() {
        const bool needs_rand = param_.mirror() ||
            (param_.crop_size());
        if (needs_rand) {
            const unsigned int rng_seed = caffe_rng_rand();
            rng_.reset(new Caffe::RNG(rng_seed));
        } else {
            rng_.reset();
        }
    }

    template <typename Dtype>
    int PEDataTransformer<Dtype>::Rand(int n) {
        CHECK(rng_);
        CHECK_GT(n, 0);
        caffe::rng_t* rng =
        static_cast<caffe::rng_t*>(rng_->generator());
        return ((*rng)() % n);
    }
    
    template<typename Dtype>
    void PEDataTransformer<Dtype>::ReadMetaData(MetaData& meta, const float* buf, int type) { 

        if(single_)
        {
            meta.dataset = "single_person";
            meta.img_size = Size((int)buf[0],(int)buf[1]);
            meta.isValidation = false;
            meta.numOtherPeople = 0;
            meta.people_index = 0;
            meta.annolist_index = 1;
            meta.write_number = (int)buf[5];
            meta.total_write_number = (int)buf[6];
            static int cur_epoch = -1;
            if(meta.write_number == 0){
                cur_epoch++;
            }
            meta.epoch = cur_epoch;
            meta.objpos.x = 0;
            meta.objpos.y = 0;
            meta.scale_self = 1;
            meta.joint_self.joints.resize(np_in_lmdb_);
            meta.joint_self.isVisible.resize(np_in_lmdb_);
            for(int i = 0; i < np_in_lmdb_; i++){
                meta.joint_self.joints[i].x = buf[10+i*3];
                meta.joint_self.joints[i].y = buf[10+i*3+1];
                int isVisible = (int)buf[10+i*3+2];
                if (isVisible == 3){
                    meta.joint_self.isVisible[i] = 3;  //
                }
                else{
                    meta.joint_self.isVisible[i] = (isVisible == 1) ? 0 : 1;
                    if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
                        meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height){
                        meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
                    }
                }
            }
        }else if (type == 1) // ai challenger
        {   
            meta.dataset = "ai_challenger";
            meta.img_size = Size((int)buf[0],(int)buf[1]);
            meta.isValidation = false;
            meta.numOtherPeople = (int)buf[2];
            meta.people_index = (int)buf[3];
            meta.annolist_index = (int)buf[4];
            meta.write_number = (int)buf[5];
            meta.total_write_number = (int)buf[6];
            static int cur_epoch = -1;
            if(meta.write_number == 0){
                cur_epoch++;
            }
            meta.epoch = cur_epoch;
            meta.objpos.x = buf[7];
            meta.objpos.y = buf[8];
            meta.objpos -= Point2f(1,1);
            meta.scale_self = buf[9] * 368. / param_.crop_size_y();
            meta.joint_self.joints.resize(np_in_lmdb_);
            meta.joint_self.isVisible.resize(np_in_lmdb_);
            for(int i = 0; i < np_in_lmdb_; i++){
                meta.joint_self.joints[i].x = buf[10+i*3];
                meta.joint_self.joints[i].y = buf[10+i*3+1];
                int isVisible = (int)buf[10+i*3+2];
                if (isVisible == 3){
                    meta.joint_self.isVisible[i] = 3;  //
                }
                else{
                    meta.joint_self.isVisible[i] = (isVisible == 1) ? 0 : 1;
                    if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
                        meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height){
                        meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
                    }
                }
            }

            meta.objpos_other.resize(meta.numOtherPeople);
            meta.scale_other.resize(meta.numOtherPeople);
            meta.joint_others.resize(meta.numOtherPeople);
            for(int p = 0; p < meta.numOtherPeople; p++){
                meta.objpos_other[p].x = buf[52+p*45];
                meta.objpos_other[p].y = buf[52+p*45+1];
                meta.objpos_other[p] -= Point2f(1,1);
                meta.scale_other[p] = buf[52+p*45+2] * 368. / param_.crop_size_y();

                meta.joint_others[p].joints.resize(np_in_lmdb_);
                meta.joint_others[p].isVisible.resize(np_in_lmdb_);
                for(int i = 0; i < np_in_lmdb_; i++){
                    meta.joint_others[p].joints[i].x = buf[55+p*45+i*3];
                    meta.joint_others[p].joints[i].y = buf[55+p*45+i*3+1];
                    int isVisible = (int)buf[55+p*45+i*3+2];
                    if (isVisible == 3){
                        meta.joint_others[p].isVisible[i] = 3;  //
                    }
                    else{
                        meta.joint_others[p].isVisible[i] = (isVisible == 1) ? 0 : 1;
                        if(meta.joint_others[p].joints[i].x < 0 || meta.joint_others[p].joints[i].y < 0 || meta.joint_others[p].joints[i].x >= meta.img_size.width || meta.joint_others[p].joints[i].y >= meta.img_size.height){
                            meta.joint_others[p].isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
                        }
                    }
                }
            }
        }else if (type == 2) // ai challenger
        {   
            meta.dataset = "MPII";
            meta.img_size = Size((int)buf[0],(int)buf[1]);
            meta.isValidation = false;
            meta.numOtherPeople = (int)buf[2];
            meta.people_index = (int)buf[3];
            meta.annolist_index = (int)buf[4];
            meta.write_number = (int)buf[5];
            meta.total_write_number = (int)buf[6];
            static int cur_epoch = -1;
            if(meta.write_number == 0){
                cur_epoch++;
            }
            meta.epoch = cur_epoch;
            meta.objpos.x = buf[7];
            meta.objpos.y = buf[8];
            meta.objpos -= Point2f(1,1);
            meta.scale_self = buf[9] / param_.crop_size_y();
            meta.joint_self.joints.resize(np_in_lmdb_);
            meta.joint_self.isVisible.resize(np_in_lmdb_);
            for(int i = 0; i < np_in_lmdb_; i++){
                meta.joint_self.joints[i].x = buf[10+i*3];
                meta.joint_self.joints[i].y = buf[10+i*3+1];
                int isVisible = (int)buf[10+i*3+2];
                if (isVisible == 3){
                    meta.joint_self.isVisible[i] = 3;  //
                }
                else{
                    meta.joint_self.isVisible[i] = (isVisible == 1) ? 0 : 1;
                    if(meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
                        meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height){
                        meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
                    }
                }
            }

            meta.objpos_other.resize(meta.numOtherPeople);
            meta.scale_other.resize(meta.numOtherPeople);
            meta.joint_others.resize(meta.numOtherPeople);
            for(int p = 0; p < meta.numOtherPeople; p++){
                meta.objpos_other[p].x = buf[58+p*51];
                meta.objpos_other[p].y = buf[58+p*51+1];
                meta.objpos_other[p] -= Point2f(1,1);
                meta.scale_other[p] = buf[58+p*51+2] / param_.crop_size_y();

                meta.joint_others[p].joints.resize(np_in_lmdb_);
                meta.joint_others[p].isVisible.resize(np_in_lmdb_);
                for(int i = 0; i < np_in_lmdb_; i++){
                    meta.joint_others[p].joints[i].x = buf[61+p*51+i*3];
                    meta.joint_others[p].joints[i].y = buf[61+p*51+i*3+1];
                    int isVisible = (int)buf[61+p*51+i*3+2];
                    if (isVisible == 3){
                        meta.joint_others[p].isVisible[i] = 3;  //
                    }
                    else{
                        meta.joint_others[p].isVisible[i] = (isVisible == 1) ? 0 : 1;
                        if(meta.joint_others[p].joints[i].x < 0 || meta.joint_others[p].joints[i].y < 0 || meta.joint_others[p].joints[i].x >= meta.img_size.width || meta.joint_others[p].joints[i].y >= meta.img_size.height){
                            meta.joint_others[p].isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
                        }
                    }
                }
            }
        }
    }

    template<typename Dtype> void PEDataTransformer<Dtype>::Transform(Mat& im, const float* buf, Blob<Dtype>* transformed_data, Blob<Dtype>* transformed_label) {

      const int im_channels = transformed_data->channels();
      const int im_num = transformed_data->num();

      const int lb_num = transformed_label->num();
      CHECK_EQ(im_channels, 3);
      CHECK_EQ(im_num, lb_num);
      CHECK_GE(im_num, 1);

      Dtype* transformed_data_pointer = transformed_data->mutable_cpu_data();
      Dtype* transformed_label_pointer = transformed_label->mutable_cpu_data();
      CPUTimer timer;
      timer.Start();
      Transform(im, buf, transformed_data_pointer, transformed_label_pointer);
      VLOG(2) << "Transform: " << timer.MicroSeconds() / 1000.0  << " ms";
    }

    template<typename Dtype> void PEDataTransformer<Dtype>::Transform(Mat& img, const float* buf, Dtype* transformed_data, Dtype* transformed_label) {

        AugmentSelection as = {
            false,
            0.0,
            Size(),
            0,
        };
        MetaData meta;

        const int datum_channels = img.channels();
        const int datum_height = img.rows;
        const int datum_width = img.cols;

        int crop_x = param_.crop_size_x();
        int crop_y = param_.crop_size_y();

        CHECK_GT(datum_channels, 0);
        Mat mask_miss;
        mask_miss = Mat::ones(datum_height, datum_width, CV_8UC1) * 255;


        int offset = img.rows * img.cols;


        //color, contract
        if(param_.gray() == 1){
            cv::cvtColor(img, img, CV_BGR2GRAY);
            cv::cvtColor(img, img, CV_GRAY2BGR);
        }

        int stride = param_.stride();
        ReadMetaData(meta, buf, param_.type());
        if(param_.transform_body_joint())
            TransformMetaJoints(meta);


        //Start transforming
        Mat img_aug;
        Mat mask_miss_aug;
        Mat img_temp, img_temp2, img_temp3; //size determined by scale
        VLOG(2) << "   input size (" << img.cols << ", " << img.rows << ")"; 
        if (phase_ == TRAIN) {
            if(!single_){
                as.scale = augmentation_scale(img, img_temp, mask_miss, meta);
                as.degree = augmentation_rotate(img_temp, img_temp2, mask_miss, meta);
                as.crop = augmentation_croppad(img_temp2, img_temp3, mask_miss, mask_miss_aug, meta);
                as.flip = augmentation_flip(img_temp3, img_aug, mask_miss_aug, meta);
                resize(mask_miss_aug, mask_miss_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
            }else{
                as.scale = 1;
                as.degree = 1;
                as.degree = augmentation_rotate(img, img_temp, mask_miss, meta);
                as.flip = augmentation_flip(img_temp, img_aug, mask_miss, meta);
                resize(img_aug, img_aug, Size(crop_x, crop_y), INTER_CUBIC);
                resize(mask_miss, mask_miss, Size(crop_x, crop_y), INTER_CUBIC);
                resize(mask_miss, mask_miss_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
            }
        }
        else {
            //img_aug = img.clone();
            as.scale = 1;
            as.crop = Size();
            as.flip = 0;
            as.degree = 0;
            int top,bottom,left, right;
            top = 0;
            left = 0;
            //bottom = stride - datum_height % stride;
            //right = stride - datum_width % stride;
            //if (bottom == stride)
            //    bottom = 0;
            //if (right == stride)
            //    right = 0;
            bottom = crop_y - datum_height;
            right = crop_x - datum_width;
            copyMakeBorder(img, img_aug, top, bottom, left, right, cv::BORDER_CONSTANT,128);
            copyMakeBorder(mask_miss, img_temp, top, bottom, left, right, cv::BORDER_CONSTANT,0);
            resize(img_temp, mask_miss_aug, Size(), 1.0/stride, 1.0/stride, INTER_CUBIC);
        }

        //copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
        offset = img_aug.rows * img_aug.cols;
        int rezX = img_aug.cols;
        int rezY = img_aug.rows;
        int grid_x = rezX / stride;
        int grid_y = rezY / stride;
        int channelOffset = grid_y * grid_x;

        for (int i = 0; i < img_aug.rows; ++i) {
            for (int j = 0; j < img_aug.cols; ++j) {
                Vec3b& rgb = img_aug.at<Vec3b>(i, j);
                transformed_data[0*offset + i*img_aug.cols + j] = (rgb[0] - 128)/128.0;
                transformed_data[1*offset + i*img_aug.cols + j] = (rgb[1] - 128)/128.0;
                transformed_data[2*offset + i*img_aug.cols + j] = (rgb[2] - 128)/128.0;
            }
        }
      
        for (int g_y = 0; g_y < grid_y; g_y++){
            for (int g_x = 0; g_x < grid_x; g_x++){
                for (int i = 0; i < np_; i++){
                    float weight = float(mask_miss_aug.at<uchar>(g_y, g_x)) /255.0; //mask_miss_aug.at<uchar>(i, j); 
                    if (meta.joint_self.isVisible[i] != 3){
                        transformed_label[i*channelOffset + g_y*grid_x + g_x] = weight;
                    }
                }  
                // background channel
                transformed_label[np_*channelOffset + g_y*grid_x + g_x] = float(mask_miss_aug.at<uchar>(g_y, g_x)) /255.0;
                
            }
        }
      
        generateLabelMap(transformed_label, img_aug, meta);


    }

    template<typename Dtype>
    void PEDataTransformer<Dtype>::TransformMetaJoints(MetaData& meta) {
    //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
        TransformJoints(meta.joint_self);
        for(int i=0;i<meta.joint_others.size();i++){
            TransformJoints(meta.joint_others[i]);
        }
    }

    template<typename Dtype>
    void PEDataTransformer<Dtype>::TransformJoints(Joints& j) {
        Joints jo = j;
        //aichallenger
        //transform to [head, neck, right hand(3), left hand(3), right leg(3), left leg(3)]
        if(param_.type() ==1){
            int MPI_to_ours_1[14] = {13, 14,4,5,6,1,2,3, 10, 11, 12, 7, 8, 9};
            int MPI_to_ours_2[14] = {13, 14,4,5,6,1,2,3, 10, 11, 12, 7, 8, 9};
            jo.joints.resize(np_);
            jo.isVisible.resize(np_);
            for(int i=0;i<14;i++){
                jo.joints[i] = (j.joints[MPI_to_ours_1[i]-1] + j.joints[MPI_to_ours_2[i]-1]) * 0.5;
                jo.isVisible[i] = j.isVisible[MPI_to_ours_1[i]-1];
            }
        }else if(param_.type() ==2){
            int MPI_to_ours[16] = {9, 8, 12, 11, 10, 13, 14, 15, 7, 6, 2, 1, 0, 3, 4, 5};
            jo.joints.resize(np_);
            jo.isVisible.resize(np_);
            for(int i=0;i<16;i++){
                jo.joints[i] = j.joints[MPI_to_ours[i]];
                jo.isVisible[i] = j.isVisible[MPI_to_ours[i]];
            }
        }
        j = jo;
    }

    // include mask_miss
    template<typename Dtype>
    float PEDataTransformer<Dtype>::augmentation_scale(Mat& img_src, Mat& img_temp, Mat& mask_miss, MetaData& meta) {
        float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        float scale_multiplier;
        if(dice > param_.scale_prob()) {
            img_temp = img_src.clone();
            scale_multiplier = 1;
        }
        else {
            float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
            scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); //linear shear into [scale_min, scale_max]
        }
        float scale_abs = param_.target_dist()/meta.scale_self;
        float scale = scale_abs * scale_multiplier;
        resize(img_src, img_temp, Size(), scale, scale, INTER_CUBIC);

        resize(mask_miss, mask_miss, Size(), scale, scale, INTER_CUBIC);

        //modify meta data
        meta.objpos *= scale;
        for(int i=0; i<np_; i++){
            meta.joint_self.joints[i] *= scale;
        }
        for(int p=0; p<meta.numOtherPeople; p++){
            meta.objpos_other[p] *= scale;
            for(int i=0; i<np_; i++){
                meta.joint_others[p].joints[i] *= scale;
            }
        }
        return scale_multiplier;
    }

    template<typename Dtype>
    Size PEDataTransformer<Dtype>::augmentation_croppad(Mat& img_src, Mat& img_dst, Mat& mask_miss, Mat& mask_miss_aug, MetaData& meta) {
        float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
        int crop_x = param_.crop_size_x();
        int crop_y = param_.crop_size_y();

        float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
        float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

        Point2i center = meta.objpos + Point2f(x_offset, y_offset);
        int offset_left = -(center.x - (crop_x/2));
        int offset_up = -(center.y - (crop_y/2));

        img_dst = Mat::zeros(crop_y, crop_x, CV_8UC3) + Scalar(128,128,128);

        mask_miss_aug = Mat::zeros(crop_y, crop_x, CV_8UC1) + Scalar(0); //for MPI, COCO with Scalar(255);
        for(int i=0;i<crop_y;i++){
            for(int j=0;j<crop_x;j++){ //i,j on cropped
                int coord_x_on_img = center.x - crop_x/2 + j;
                int coord_y_on_img = center.y - crop_y/2 + i;
                if(onPlane(Point(coord_x_on_img, coord_y_on_img), Size(img_src.cols, img_src.rows))){
                    img_dst.at<Vec3b>(i,j) = img_src.at<Vec3b>(coord_y_on_img, coord_x_on_img);

                    mask_miss_aug.at<uchar>(i,j) = mask_miss.at<uchar>(coord_y_on_img, coord_x_on_img);


                }
            }
        }

        //modify meta data
        Point2f offset(offset_left, offset_up);
        meta.objpos += offset;
        for(int i=0; i<np_; i++){
            meta.joint_self.joints[i] += offset;
        }
        for(int p=0; p<meta.numOtherPeople; p++){
            meta.objpos_other[p] += offset;
            for(int i=0; i<np_; i++){
                meta.joint_others[p].joints[i] += offset;
            }
        }

        return Size(x_offset, y_offset);
    }

    template<typename Dtype>
    bool PEDataTransformer<Dtype>::onPlane(Point p, Size img_size) {
        if(p.x < 0 || p.y < 0) return false;
        if(p.x >= img_size.width || p.y >= img_size.height) return false;
        return true;
    }

    template<typename Dtype>
    bool PEDataTransformer<Dtype>::augmentation_flip(Mat& img_src, Mat& img_aug, Mat& mask_miss, MetaData& meta) {
        bool doflip;
        if(param_.aug_way() == "rand"){
            float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            doflip = (dice <= param_.flip_prob());
        }
        //else if(param_.aug_way() == "table"){
        //   doflip = (aug_flips[meta.write_number][meta.epoch % param_.num_total_augs()] == 1);
        //}
        else {
            doflip = 0;
            LOG(INFO) << "Unhandled exception!!!!!!";
        }

        if(doflip){
            cv::flip(img_src, img_aug, 1);
            int w = img_src.cols;
            
            cv::flip(mask_miss, mask_miss, 1);
            
            meta.objpos.x = w - 1 - meta.objpos.x;
            for(int i=0; i<np_; i++){
                meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
            }
            if(param_.transform_body_joint())
                swapLeftRight(meta.joint_self);

            for(int p=0; p<meta.numOtherPeople; p++){
                meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
                for(int i=0; i<np_; i++){
                    meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
                }
                if(param_.transform_body_joint())
                    swapLeftRight(meta.joint_others[p]);
            }
        }
        else {
            img_aug = img_src.clone();
        }
        return doflip;
    }

    //modified by zane added ai_challenger(np=40)
    template<typename Dtype>
    void PEDataTransformer<Dtype>::swapLeftRight(Joints& j) {

        if(param_.type() == 1){
            //id is from 1 to 14
            int right[6] = {3,4,5,9,10,11};
            int left[6] = {6,7,8,12,13,14};
            for(int i=0; i<6; i++){   
                int ri = right[i] - 1;
                int li = left[i] - 1;
                Point2f temp = j.joints[ri];
                j.joints[ri] = j.joints[li];
                j.joints[li] = temp;
                int temp_v = j.isVisible[ri];
                j.isVisible[ri] = j.isVisible[li];
                j.isVisible[li] = temp_v;
            }
        }
    }

    template<typename Dtype>
    float PEDataTransformer<Dtype>::augmentation_rotate(Mat& img_src, Mat& img_dst, Mat& mask_miss, MetaData& meta) {

        float degree;
        if(param_.aug_way() == "rand"){
            float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
        }
        //else if(param_.aug_way() == "table"){
        //    degree = aug_degs[meta.write_number][meta.epoch % param_.num_total_augs()];
        //}
        else {
            degree = 0;
            LOG(INFO) << "Unhandled exception!!!!!!";
        }
        if(degree){
            Point2f center(img_src.cols/2.0, img_src.rows/2.0);
            Mat R = getRotationMatrix2D(center, degree, 1.0);
            Rect bbox = RotatedRect(center, img_src.size(), degree).boundingRect();
            // adjust transformation matrix
            R.at<double>(0,2) += bbox.width/2.0 - center.x;
            R.at<double>(1,2) += bbox.height/2.0 - center.y;

            cv::warpAffine(img_src, img_dst, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128,128,128));

            cv::warpAffine(mask_miss, mask_miss, R, bbox.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(0)); //Scalar(0) for MPI, COCO with Scalar(255);


            //adjust meta data
            RotatePoint(meta.objpos, R);
            for(int i=0; i<np_; i++){
                RotatePoint(meta.joint_self.joints[i], R);
            }
            for(int p=0; p<meta.numOtherPeople; p++){
                RotatePoint(meta.objpos_other[p], R);
                for(int i=0; i<np_; i++){
                    RotatePoint(meta.joint_others[p].joints[i], R);
                }
            }
        }
        return degree;
    }

    template<typename Dtype>
    void PEDataTransformer<Dtype>::RotatePoint(Point2f& p, Mat R){
        Mat point(3,1,CV_64FC1);
        point.at<double>(0,0) = p.x;
        point.at<double>(1,0) = p.y;
        point.at<double>(2,0) = 1;
        Mat new_point = R * point;
        p.x = new_point.at<double>(0,0);
        p.y = new_point.at<double>(1,0);
    }

    template<typename Dtype>
    void PEDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma)
    {
        float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
        for (int g_y = 0; g_y < grid_y; g_y++){
            for (int g_x = 0; g_x < grid_x; g_x++){
                float x = start + g_x * stride;
                float y = start + g_y * stride;
                float d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
                float exponent = d2 / 2.0 / sigma / sigma;
                if(exponent > 4.6052){ //ln(100) = -ln(1%)
                    continue;
            }
            entry[g_y*grid_x + g_x] += exp(-exponent);
            if(entry[g_y*grid_x + g_x] > 1) 
                entry[g_y*grid_x + g_x] = 1;
            }
        }
    }
/*
    template<typename Dtype>
    void CPMDataTransformer<Dtype>::putVecPeaks(Dtype* entryX, Dtype* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre){
        //int thre = 4;
        centerB = centerB*0.125;
        centerA = centerA*0.125;
        Point2f bc = centerB - centerA;
        float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
        bc.x = bc.x /norm_bc;
        bc.y = bc.y /norm_bc;

        for(int j=0;j<3;j++){
        //Point2f center = centerB*0.5 + centerA*0.5;
            Point2f center = centerB*0.5*j + centerA*0.5*(2-j);

            int min_x = std::max( int(floor(center.x-thre)), 0);
            int max_x = std::min( int(ceil(center.x+thre)), grid_x);

            int min_y = std::max( int(floor(center.y-thre)), 0);
            int max_y = std::min( int(ceil(center.y+thre)), grid_y);

            for (int g_y = min_y; g_y < max_y; g_y++){
                for (int g_x = min_x; g_x < max_x; g_x++){
                    float dist = (g_x-center.x)*(g_x-center.x) + (g_y-center.y)*(g_y-center.y);
                    if(dist <= thre){
                        int cnt = count.at<uchar>(g_y, g_x);
                        if (cnt == 0){
                            entryX[g_y*grid_x + g_x] = bc.x;
                            entryY[g_y*grid_x + g_x] = bc.y;
                        }
                        else{
                            entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc.x) / (cnt + 1);
                            entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc.y) / (cnt + 1);
                            count.at<uchar>(g_y, g_x) = cnt + 1;
                        }
                    }
                }
            }
        }
    }
*/

    template<typename Dtype>
    void PEDataTransformer<Dtype>::putVecMaps(Dtype* entryX, Dtype* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre){
        //int thre = 4;
        centerB = centerB*0.125;
        centerA = centerA*0.125;
        Point2f bc = centerB - centerA;
        int min_x = std::max( int(round(std::min(centerA.x, centerB.x)-thre)), 0);
        int max_x = std::min( int(round(std::max(centerA.x, centerB.x)+thre)), grid_x);

        int min_y = std::max( int(round(std::min(centerA.y, centerB.y)-thre)), 0);
        int max_y = std::min( int(round(std::max(centerA.y, centerB.y)+thre)), grid_y);

        float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
        bc.x = bc.x /norm_bc;
        bc.y = bc.y /norm_bc;

        for (int g_y = min_y; g_y < max_y; g_y++){
            for (int g_x = min_x; g_x < max_x; g_x++){
                Point2f ba;
                ba.x = g_x - centerA.x;
                ba.y = g_y - centerA.y;
                float dist = std::abs(ba.x*bc.y -ba.y*bc.x);

                if(dist <= thre){
                    int cnt = count.at<uchar>(g_y, g_x);
                    if (cnt == 0){
                        entryX[g_y*grid_x + g_x] = bc.x;
                        entryY[g_y*grid_x + g_x] = bc.y;
                    }
                    else{
                        entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc.x) / (cnt + 1);
                        entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc.y) / (cnt + 1);
                        count.at<uchar>(g_y, g_x) = cnt + 1;
                    }
                }
            }
        }
    }

    template<typename Dtype>
    void PEDataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, Mat& img_aug, MetaData meta) {
        int rezX = img_aug.cols;
        int rezY = img_aug.rows;
        int stride = param_.stride();
        int grid_x = rezX / stride;
        int grid_y = rezY / stride;
        int channelOffset = grid_y * grid_x;

        for (int g_y = 0; g_y < grid_y; g_y++){
            for (int g_x = 0; g_x < grid_x; g_x++){
                for (int i = np_+1; i < 2*(np_+1); i++){
                    if (i == (2*np_ + 1))
                        continue;
                    transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
                }
            }
        }




        if (param_.type() == 1){
            for (int i = 0; i < 14; i++){
                Point2f center = meta.joint_self.joints[i];
                if(meta.joint_self.isVisible[i] <= 1){
                    putGaussianMaps(transformed_label + (i+np_+27)*channelOffset, center, param_.stride(), grid_x, grid_y, param_.sigma()); //self
                }
                for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
                    Point2f center = meta.joint_others[j].joints[i];
                    if(meta.joint_others[j].isVisible[i] <= 1){
                        putGaussianMaps(transformed_label + (i+np_+27)*channelOffset, center, param_.stride(), grid_x, grid_y, param_.sigma());
                    }
                }
            }

            int mid_1[13] = {0, 1, 2, 3, 1, 5, 6, 1, 8, 9,   1, 11, 12};
            int mid_2[13] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
            int thre = 1;

            for(int i=0;i<13;i++){
                Mat count = Mat::zeros(grid_y, grid_x, CV_8UC1);
                Joints jo = meta.joint_self;
                if(jo.isVisible[mid_1[i]]<=1 && jo.isVisible[mid_2[i]]<=1){
                    //putVecMaps
                    putVecMaps(transformed_label + (np_+ 1+ 2*i)*channelOffset, transformed_label + (np_+ 2+ 2*i)*channelOffset,  count, jo.joints[mid_1[i]], jo.joints[mid_2[i]], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
                }

                for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
                    Joints jo2 = meta.joint_others[j];
                    if(jo2.isVisible[mid_1[i]]<=1 && jo2.isVisible[mid_2[i]]<=1){
                        //putVecMaps
                        putVecMaps(transformed_label + (np_+ 1+ 2*i)*channelOffset, transformed_label + (np_+ 2+ 2*i)*channelOffset, count, jo2.joints[mid_1[i]], jo2.joints[mid_2[i]], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
                    }
                }
            }

            //put background channel
            for (int g_y = 0; g_y < grid_y; g_y++){
                for (int g_x = 0; g_x < grid_x; g_x++){
                    float maximum = 0;
                    //second background channel
                    for (int i = np_+27; i < np_+41; i++){
                        maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
                    }
                    transformed_label[(2*np_+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
                }
            }
        }else if (param_.type() == 2){
            for (int i = 0; i < 16; i++){
                Point2f center = meta.joint_self.joints[i];
                if(meta.joint_self.isVisible[i] <= 1){
                    putGaussianMaps(transformed_label + (i+np_+31)*channelOffset, center, param_.stride(), grid_x, grid_y, param_.sigma()); //self
                }
                for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
                    Point2f center = meta.joint_others[j].joints[i];
                    if(meta.joint_others[j].isVisible[i] <= 1){
                        putGaussianMaps(transformed_label + (i+np_+31)*channelOffset, center, param_.stride(), grid_x, grid_y, param_.sigma());
                    }
                }
            }

            int mid_1[15] = {0, 1, 2, 3, 1, 5, 6, 1, 8, 9,  10, 11,  9, 13, 14};
            int mid_2[15] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
            int thre = 1;

            for(int i=0;i<15;i++){
                Mat count = Mat::zeros(grid_y, grid_x, CV_8UC1);
                Joints jo = meta.joint_self;
                if(jo.isVisible[mid_1[i]]<=1 && jo.isVisible[mid_2[i]]<=1){
                    //putVecMaps
                    putVecMaps(transformed_label + (np_+ 1+ 2*i)*channelOffset, transformed_label + (np_+ 2+ 2*i)*channelOffset,  count, jo.joints[mid_1[i]], jo.joints[mid_2[i]], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
                }

                for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
                    Joints jo2 = meta.joint_others[j];
                    if(jo2.isVisible[mid_1[i]]<=1 && jo2.isVisible[mid_2[i]]<=1){
                        //putVecMaps
                        putVecMaps(transformed_label + (np_+ 1+ 2*i)*channelOffset, transformed_label + (np_+ 2+ 2*i)*channelOffset, count, jo2.joints[mid_1[i]], jo2.joints[mid_2[i]], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
                    }
                }
            }

            //put background channel
            for (int g_y = 0; g_y < grid_y; g_y++){
                for (int g_x = 0; g_x < grid_x; g_x++){
                    float maximum = 0;
                    //second background channel
                    for (int i = np_+31; i < np_+47; i++){
                        maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
                    }
                    transformed_label[(2*np_+1)*channelOffset + g_y*grid_x + g_x] = max(1.0-maximum, 0.0);
                }
            }
        }

    }

    INSTANTIATE_CLASS(PEDataTransformer);
} //namespace caffe
