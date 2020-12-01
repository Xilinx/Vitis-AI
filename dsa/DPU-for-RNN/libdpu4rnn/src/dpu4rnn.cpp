#include <iostream>
#include "dpu4rnn.hpp"
#include "dpu4rnn_imp.hpp"

dpu4rnn::dpu4rnn() {}
dpu4rnn::~dpu4rnn() {}

std::unique_ptr<dpu4rnn> dpu4rnn::create(const std::string& model_name, const int device_id) {
  //std::cout << "model name is " << model_name << std::endl;
  return std::unique_ptr<dpu4rnn>(new dpu4rnnImp(model_name, device_id));
}

