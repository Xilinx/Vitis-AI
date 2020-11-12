#ifndef CAFFE_UTIL_CONVERT_PROTO_HPP_
#define CAFFE_UTIL_CONVERT_PROTO_HPP_

#include "caffe/blob.hpp"
#include "caffe/caffe.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/proto/quantize.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/quantize.hpp"
#include "opencv2/opencv.hpp"
#include <unordered_set>
#include <unordered_map>

namespace caffe {

NetParameter RemoveLayerByType(const NetParameter &model_in, const string type);

NetParameter RemoveLayerByPhase(const NetParameter &model_in,
                                caffe::Phase phase);

NetParameter RemoveLayerByName(const NetParameter &model_in, const string name);

NetParameter RemoveBlobs(const NetParameter &model_in);

NetParameter RemoveDuplicatedLayerByType(const NetParameter &model_in, const string type);

bool CheckImageDataLayer(const LayerParameter *data_layer);

bool CheckModel(const NetParameter &net_param);

bool CheckModelAndSave(const NetParameter &net_param, const string filename);

template <typename Dtype>
void DataStat(const int n, const Dtype *data, const char* title);

template <typename Dtype>
void BindWeightWithProfile(Blob<float>& new_weight, const Dtype *weight,
                const Dtype *scale, const Dtype *var, float eps);

template <typename Dtype>
void BindWeight(const int n, const int kernel_dim, const Dtype *weight,
                const Dtype *scale, const Dtype *var, Dtype *bind_weight, float eps);

template <typename Dtype>
void BindBias(const int n, const Dtype *mean, const Dtype *scale,
              const Dtype *var, const Dtype *bias_conv, const Dtype *bias_bn,
              Dtype *bind_bias, float eps);

template <typename Dtype>
void ScaleInvVar(const int n, const Dtype *scale, const Dtype *var,
                 Dtype *scale_inv_var, float eps);

vector<vector<bool>> BuildConnections(const NetParameter &net);

void ShowConnections(NetParameter net, vector<vector<bool>> connect);

const string InferNetType(const NetParameter &net_param);

const vector<int> GetInputShape(NetParameter net_param, const string &input_blob);

TransformationParameter GetTransformParam(const NetParameter &net);

int GetNextLayer(const NetParameter &net, const int index, const string type,
                 const vector<vector<bool>> &connect);

int GetPreviousLayer(const NetParameter &net, const int index,
                     const string type, const vector<vector<bool>> &connect);

bool IsInplace(const LayerParameter *layer);

int GetLastInputLayer(const NetParameter &net);

bool IsFixSkipLayer(LayerParameter layer);

// Transform Model with fix_info
typedef map<string, vector<int>> FIX_INFO;
typedef pair<string, vector<int>> FIX_INFO_ITEM;

LayerParameter TransformConvFixed(const LayerParameter &conv_fixed,
                                  int &weight_dpos, int &bias_dpos);

LayerParameter TransformConvBNFixed(const LayerParameter &conv_bn_fixed,
                                    int &weight_dpos, int &bias_dpos);

LayerParameter TransformFCFixed(const LayerParameter &fc_fixed,
                                int &weight_dpos, int &bias_dpos);

LayerParameter TransformBatchNormFixed(const LayerParameter &bn_fixed,
                                       int &weight_dpos, int &bias_dpos);

LayerParameter TransformPReLUFixed(const LayerParameter &bn_fixed,
                                       int &weight_dpos, int &bias_dpos);

LayerParameter TransformDeconvFixed(const LayerParameter &bn_fixed,
                                       int &weight_dpos, int &bias_dpos);

LayerParameter TransformScaleFixed(const LayerParameter &bn_fixed,
                                       int &weight_dpos, int &bias_dpos);

LayerParameter TransformNormalizeFixed(const LayerParameter &bn_fixed,
                                       int &weight_dpos, int &bias_dpos);

void TransformFix2FloatWithFixInfo(NetParameter &net_in, NetParameter &net_out,
                                   FIX_INFO &weight_infos, FIX_INFO &bias_infos,
                                   FIX_INFO &data_infos,
                                   bool keep_fixed_neuron);

vector<int> GetLayerFixInfo(const string &layer_name, FIX_INFO &fix_infos);

vector<int> GetDataFixInfo(const string &blob_name, FIX_INFO &data_infos);

void ConcatVector(vector<int> &vec1, vector<int> &vec2, int times = 1);

void UpdateDataFixInfo(const NetParameter &net_in, FIX_INFO &data_infos);

vector<vector<int>> GetFixInfo(NetParameter &net, FIX_INFO &weight_infos,
                               FIX_INFO &bias_infos, FIX_INFO &data_infos);

void WriteFixInfoToFile(NetParameter &net, vector<vector<int>> fix_infos,
                        string file_name);

void WriteFixInfoToModel(NetParameter &model, vector<vector<int>> fix_infos);

// Merge Model
void MergeConvBatchNorm2Conv(const NetParameter &net_in, NetParameter &net_out,
                             const int conv_index, const int bn_index,
                             vector<bool> &processed, bool binary = false);

void MergeConvBatchNorm2ConvBNFixed(const NetParameter &net_in,
                                    NetParameter &net_out, const int conv_index,
                                    const int bn_index, vector<bool> &processed,
                                    bool binary = false);

void MergeBatchNormConv2Conv(const NetParameter &net_in, NetParameter &net_out,
                             const int bn_index, const int conv_index,
                             vector<bool> &processed, bool binary = false);

void MergeFCBatchNorm2FC(const NetParameter &net_in, NetParameter &net_out,
                             const int fc_index, const int bn_index,
                             vector<bool> &processed, bool binary = false);

void MergeBvlcBatchNormScale2BatchNorm(const NetParameter &net_in,
                                       NetParameter &net_out,
                                       const int bn_index, const int sc_index,
                                       vector<bool> &processed,
                                       bool binary = false);

void MergeBatchNorm2Scale(const NetParameter &net_in, NetParameter &net_out,
                          const int bn_index, vector<bool> &processed,
                          bool binary = false);

void ConvertBvlcBatchNorm(const NetParameter &net_in, NetParameter &net_out,
                          const int bn_index, vector<bool> &processed,
                          bool binary = false);

void ConvertBN2BatchNorm(const NetParameter &net_in, NetParameter &net_out,
                         const int bn_index, vector<bool> &processed,
                         bool binary = false);

// Convert float2fix
BatchNormParameter BuildBatchNormParam();

FixedParameter BuildFixedParam(const int method, const int bit_width);

void InsertFixedNeuron(std::unordered_set<std::string> &fixed_blobs,
                       const std::string &name, NetParameter &net,
                       const int method, const int data_bit, bool isLastDataLayer = false );

void SearchInsertNeuronLayer(const NetParameter &net_in, NetParameter &net_out,
                             const int index, vector<bool> &need_insert_fix,
                             std::unordered_set<std::string> &fixed_blobs,
                             const vector<vector<bool>> &connect,
                             const int method, const int data_bit, 
                             map<string, pair<int,int> > &special_layers);

void ConvertConvolution2Fixed(const NetParameter &net_in, NetParameter &net_out,
                              const int index, vector<bool> &need_insert_fix,
                              const vector<vector<bool>> &connect,
                              std::unordered_set<std::string> &fixed_blobs,
                              vector<bool> &processed, const int method,
                              const int weight_bit, const int data_bit,
                              const bool add_bn, map<string, pair<int,int> > &special_layers);

void ConvertBatchNorm2Fixed(const NetParameter &net_in, NetParameter &net_out,
                            const int index, vector<bool> &need_insert_fix,
                            const vector<vector<bool>> &connect,
                            std::unordered_set<std::string> &fixed_blobs,
                            vector<bool> &processed, const int method,
                            const int weight_bit, const int data_bit, 
                            map<string, pair<int,int> > &special_layers);

void ConvertInnerProduct2Fixed(const NetParameter &net_in,
                               NetParameter &net_out, const int index,
                               vector<bool> &need_insert_fix,
                               const vector<vector<bool>> &connect,
                               std::unordered_set<std::string> &fixed_blobs,
                               vector<bool> &processed, const int method,
                               const int weight_bit, const int data_bit, 
                               map<string, pair<int,int> > &special_layers);

void ConvertPReLU2Fixed(const NetParameter &net_in, NetParameter &net_out,
                        const int index, vector<bool> &need_insert_fix,
                        const vector<vector<bool>> &connect,
                        std::unordered_set<std::string> &fixed_blobs,
                        vector<bool> &processed, const int method,
                        const int weight_bit, const int data_bit, 
                        map<string, pair<int,int> > &special_layers);

void ConvertDeconvolution2Fixed(const NetParameter &net_in,
                                NetParameter &net_out, const int index,
                                vector<bool> &need_insert_fix,
                                const vector<vector<bool>> &connect,
                                std::unordered_set<std::string> &fixed_blobs,
                                vector<bool> &processed, const int method,
                                const int weight_bit, const int data_bit, 
                                map<string, pair<int,int> > &special_layers);

void ConvertScale2Fixed(const NetParameter &net_in,
                                NetParameter &net_out, const int index,
                                vector<bool> &need_insert_fix,
                                const vector<vector<bool>> &connect,
                                std::unordered_set<std::string> &fixed_blobs,
                                vector<bool> &processed, const int method,
                                const int weight_bit, const int data_bit, 
                                map<string, pair<int,int> > &special_layers);

void ConvertBatchNorm2Fixed(const NetParameter &net_in,
                                NetParameter &net_out, const int index,
                                vector<bool> &need_insert_fix,
                                const vector<vector<bool>> &connect,
                                std::unordered_set<std::string> &fixed_blobs,
                                vector<bool> &processed, const int method,
                                const int weight_bit, const int data_bit, 
                                map<string, pair<int,int> > &special_layers);

bool NeedIgnore(const string &layer_name, std::unordered_map<std::string, bool> &ignore_layers);

bool isNum(const string &str);

void handleParam2map(const string str, map<string, pair<int,int> > &amap, const int data_bit);

// Convert train2deploy
void ConvertSoftLoss2Soft(const NetParameter &net, const int soft_index);

NetParameter ConvertTrain2Deploy(const NetParameter &train_net);

int MergeLayers(const NetParameter &net_in, NetParameter &net_out, bool binary,
                bool keep_bn, bool keep_convbn);

int ConvertFloat2Fix(const NetParameter &net_in, NetParameter &net_out,
                     const string &FLAGS_ignore_layers, const string &FLAGS_ignore_layers_file, const int method,
                     const int weight_bit, const int data_bit,
                     const bool add_bn, const string &sigmoided_layers, const string &FLAGS_special_layers);

void InitIgnoreFile(std::unordered_map<std::string, bool> &ignore_layers,
                    const string ignore_layers_file_in,
                    const string ignore_layers_in);

void InitSigmoidedLayers(vector<string> &sigmoided_layers,
                    const string &ignore_layers_in);

void MergeModelAndWeights( NetParameter& model, 
                           NetParameter& weights ); 

} // namespace caffe
#endif // CAFFE_UTIL_CONVERT_PROTO_HPP_
