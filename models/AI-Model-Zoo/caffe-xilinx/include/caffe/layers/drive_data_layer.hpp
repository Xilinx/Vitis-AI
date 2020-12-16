#ifndef CAFFE_DRIVE_DATA_LAYER_HPP_
#define CAFFE_DRIVE_DATA_LAYER_HPP_

#include <string>
#include <vector>
#include <utility>

#include "hdf5.h" // Ming

#include "caffe/blob.hpp"
#include "caffe/common.hpp" // Ming
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/data_layer.hpp"
//#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/filler.hpp"

namespace caffe {
    template <typename Dtype>
    class DriveDataLayer : public BasePrefetchingDataLayer<Dtype> {//inherent parent is different
    //class DriveDataLayer : public DataLayer<Dtype>{
    
        //vector<cv::Mat> genpic;
        //vector<int> genpic_type;
    public:
        explicit DriveDataLayer(const LayerParameter& param);
        virtual ~DriveDataLayer();
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                const vector<Blob<Dtype>*>& top);
        virtual inline bool ShareInParallel() const { return false; }
        virtual inline const char* type() const { return "DriveData"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int MaxTopBlobs() const { return 5; }
        virtual inline int MinTopBlobs() const { return 1; }

    protected:
        virtual void load_batch(Batch<Dtype>* batch);
        DataReader<DriveData> reader_;
        //DataReader reader_;
        Blob<Dtype> data_mean_;
        //bool ReadBoundingBoxLabelToDatum( const DriveData& data, Datum* datum, const int h_off, const int w_off, const float resize, const DriveDataParameter& param, Dtype* label_type, bool can_pass, int bid );
    };
}  // namespace caffe

#endif  // CAFFE_DRIVE_DATA_LAYER_HPP_
