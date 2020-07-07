#ifdef USE_CUDNN
#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_batch_norm_layer.hpp"
#include "caffe/layers/cudnn_conv_bn_fixed_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
// #include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
class CuDNNConvolutionBNFixedLayerTest : public GPUDeviceTest<Dtype> {
 protected:
  CuDNNConvolutionBNFixedLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 1, 3, 3)),
        blob_conv_bn_top_(new Blob<Dtype>()),
        blob_conv_top_(new Blob<Dtype>()),
        blob_bn_top_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    for (auto i = 0; i < blob_bottom_->count(); ++i)
      blob_bottom_->mutable_cpu_data()[i] = Dtype(0.1)*(i+1);
  }

  virtual void SetUpConvolutionBNFixedLayer(Phase phase, bool enable_fix=true);
  virtual void SetUpConvolutionLayer();
  virtual void SetUpBatchNormLayer(Phase phase);

  virtual ~CuDNNConvolutionBNFixedLayerTest() {
    delete blob_bottom_;
    delete blob_conv_bn_top_;
    delete blob_conv_top_;
    delete blob_bn_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_conv_bn_top_;
  Blob<Dtype>* const blob_conv_top_;
  Blob<Dtype>* const blob_bn_top_;

  shared_ptr<CuDNNConvolutionBNFixedLayer<Dtype> > conv_bn_fixed_layer_;
  shared_ptr<CuDNNConvolutionLayer<Dtype> > conv_layer_;
  shared_ptr<CuDNNBatchNormLayer<Dtype> > bn_layer_;
};

template <typename Dtype>
void CuDNNConvolutionBNFixedLayerTest<Dtype>::SetUpConvolutionBNFixedLayer(
    Phase phase, bool enable_fix) {

  LayerParameter layer_param;
  layer_param.set_phase(phase);
  // convolution param
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->add_pad(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1);
  convolution_param->set_bias_term(false);

  // batchnorm param
  BatchNormParameter* batch_norm_param = layer_param.mutable_batch_norm_param();
  batch_norm_param->mutable_scale_filler()->set_type("constant");
  batch_norm_param->mutable_scale_filler()->set_value(1);
  batch_norm_param->mutable_bias_filler()->set_type("constant");
  batch_norm_param->mutable_bias_filler()->set_value(0);

  FixedParameter* fixed_param = layer_param.mutable_fixed_param();
  fixed_param->set_enable(enable_fix);

  conv_bn_fixed_layer_ =
      make_shared<CuDNNConvolutionBNFixedLayer<Dtype> >(layer_param);
  this->conv_bn_fixed_layer_->SetUp(
      {this->blob_bottom_}, {this->blob_conv_bn_top_});
  this->conv_bn_fixed_layer_->Reshape(
      {this->blob_bottom_}, {this->blob_conv_bn_top_});
  return;
} 

template <typename Dtype>
void CuDNNConvolutionBNFixedLayerTest<Dtype>::SetUpConvolutionLayer() {

  LayerParameter layer_param;
  // convolution param
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(1);
  convolution_param->add_pad(1);
  convolution_param->set_num_output(1);
  convolution_param->mutable_weight_filler()->set_type("constant");
  convolution_param->mutable_weight_filler()->set_value(1);
  convolution_param->set_bias_term(false);

  conv_layer_ =
      make_shared<CuDNNConvolutionLayer<Dtype> >(layer_param);
  this->conv_layer_->SetUp(
      {this->blob_bottom_}, {this->blob_conv_top_});
  this->conv_layer_->Reshape(
      {this->blob_bottom_}, {this->blob_conv_top_});
  return;
} 

template <typename Dtype>
void CuDNNConvolutionBNFixedLayerTest<Dtype>::SetUpBatchNormLayer(
    Phase phase) {
  LayerParameter layer_param;
  layer_param.set_phase(phase);
  // batchnorm param
  BatchNormParameter* bn_param = layer_param.mutable_batch_norm_param();
  bn_param->mutable_scale_filler()->set_type("constant");
  bn_param->mutable_scale_filler()->set_value(1);
  bn_param->mutable_bias_filler()->set_type("constant");
  bn_param->mutable_bias_filler()->set_value(0);

  bn_layer_ =
      make_shared<CuDNNBatchNormLayer<Dtype> >(layer_param);
  this->bn_layer_->SetUp(
      {this->blob_conv_top_}, {this->blob_bn_top_});
  this->bn_layer_->Reshape(
      {this->blob_conv_top_}, {this->blob_bn_top_});
  return;
} 

TYPED_TEST_CASE(CuDNNConvolutionBNFixedLayerTest, TestDtypes);

TYPED_TEST(CuDNNConvolutionBNFixedLayerTest, TestTrainFirstForward) {

  this->SetUpConvolutionBNFixedLayer(TRAIN, false);
  this->SetUpConvolutionLayer();
  this->SetUpBatchNormLayer(TRAIN);

  this->conv_bn_fixed_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_bn_top_});

  this->conv_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_top_});
  this->bn_layer_->Forward(
      {this->blob_conv_top_}, {this->blob_bn_top_});

  auto conv_bn_data = this->blob_conv_bn_top_->cpu_data();
  // auto conv_data = this->blob_conv_top_->cpu_data();
  auto bn_data = this->blob_bn_top_->cpu_data();

  for (auto i = 0; i < this->blob_conv_bn_top_->count(); ++i) {
    EXPECT_NEAR(conv_bn_data[i], bn_data[i], 1e-3);
  }
  
}

/*
TYPED_TEST(CuDNNConvolutionBNFixedLayerTest, TestTrainFirstFixedForward) {

  this->SetUpConvolutionBNFixedLayer(TRAIN, true);
  this->SetUpConvolutionLayer();
  this->SetUpBatchNormLayer(TRAIN);

  this->conv_bn_fixed_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_bn_top_});

  this->conv_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_top_});
  this->bn_layer_->Forward(
      {this->blob_conv_top_}, {this->blob_bn_top_});

  auto conv_bn_data = this->blob_conv_bn_top_->cpu_data();
  auto bn_data = this->blob_bn_top_->cpu_data();

  for (auto i = 0; i < this->blob_conv_bn_top_->count(); ++i) {
    EXPECT_NEAR(conv_bn_data[i], bn_data[i], fabs(bn_data[i])*1e-2);
  }
  
}
*/

TYPED_TEST(CuDNNConvolutionBNFixedLayerTest, TestTrainSecondForward) {

  this->SetUpConvolutionBNFixedLayer(TRAIN, false);
  this->SetUpConvolutionLayer();
  this->SetUpBatchNormLayer(TRAIN);

  // 1st iter
  this->conv_bn_fixed_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_bn_top_});

  this->conv_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_top_});
  this->bn_layer_->Forward(
      {this->blob_conv_top_}, {this->blob_bn_top_});

  // 2nd iter
  for (auto i = 0; i < this->blob_bottom_->count(); ++i)
    this->blob_bottom_->mutable_cpu_data()[i] = 0.2*(i+1);

  this->conv_bn_fixed_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_bn_top_});

  this->conv_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_top_});
  this->bn_layer_->Forward(
      {this->blob_conv_top_}, {this->blob_bn_top_});

  auto conv_bn_data = this->blob_conv_bn_top_->cpu_data();
  auto bn_data = this->blob_bn_top_->cpu_data();

  for (auto i = 0; i < this->blob_conv_bn_top_->count(); ++i) {
    EXPECT_NEAR(conv_bn_data[i], bn_data[i], 1e-3);
  }
}

TYPED_TEST(CuDNNConvolutionBNFixedLayerTest, TestInferenceFirstForward) {

  this->SetUpConvolutionBNFixedLayer(TEST, false);
  this->SetUpConvolutionLayer();
  this->SetUpBatchNormLayer(TEST);

  this->conv_bn_fixed_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_bn_top_});

  this->conv_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_top_});
  this->bn_layer_->Forward(
      {this->blob_conv_top_}, {this->blob_bn_top_});

  auto conv_bn_data = this->blob_conv_bn_top_->cpu_data();
  auto bn_data = this->blob_bn_top_->cpu_data();

  for (auto i = 0; i < this->blob_conv_bn_top_->count(); ++i) {
    EXPECT_NEAR(conv_bn_data[i], bn_data[i], 1e-3);
  }
}

TYPED_TEST(CuDNNConvolutionBNFixedLayerTest, TestInferenceFirstFixedForward) {

  this->SetUpConvolutionBNFixedLayer(TEST);
  this->SetUpConvolutionLayer();
  this->SetUpBatchNormLayer(TEST);

  this->conv_bn_fixed_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_bn_top_});

  this->conv_layer_->Forward(
      {this->blob_bottom_}, {this->blob_conv_top_});

  this->bn_layer_->Forward(
      {this->blob_conv_top_}, {this->blob_bn_top_});

  auto conv_bn_data = this->blob_conv_bn_top_->cpu_data();
  auto bn_data = this->blob_bn_top_->cpu_data();

  for (auto i = 0; i < this->blob_conv_bn_top_->count(); ++i) {
    EXPECT_NEAR(conv_bn_data[i], bn_data[i], fabs(bn_data[i])*1e-3);
  }
}

/*
TYPED_TEST(CuDNNConvolutionBNFixedLayerTest, TestGradientCuDNN) {
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);
  convolution_param->add_kernel_size(3);
  convolution_param->add_stride(2);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  CuDNNConvolutionLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
*/

#endif

}  // namespace caffe
