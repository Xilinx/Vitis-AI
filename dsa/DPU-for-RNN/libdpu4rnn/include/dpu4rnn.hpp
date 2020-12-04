#pragma once
#include <string>
#include <vector>
#include <memory>

class dpu4rnn {
 public:
  dpu4rnn();
  dpu4rnn(const std::string& model_name) = delete;
  virtual ~dpu4rnn();

  static std::unique_ptr<dpu4rnn> create(const std::string& model_name,
		  const int device_id = 0);
  virtual void run(const char* input, int in_size,
		  char* output, int frame_num, int batch = 1) = 0;
  virtual int getBatch() = 0;
};
