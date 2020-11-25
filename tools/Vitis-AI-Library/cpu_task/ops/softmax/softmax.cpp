#include <iostream>

#include "vart/op_imp.h"
using namespace std;

namespace {
class SoftmaxOpImp : public vart::OpImp {
 public:
  explicit SoftmaxOpImp(const xir::Op* op);
  virtual ~SoftmaxOpImp();
  SoftmaxOpImp(const SoftmaxOpImp& other) = delete;
  SoftmaxOpImp& operator=(const SoftmaxOpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

SoftmaxOpImp::SoftmaxOpImp(const xir::Op* op) : vart::OpImp(op){};
SoftmaxOpImp::~SoftmaxOpImp() {}
int SoftmaxOpImp::calculate(const std::vector<vart::OpImpArg>& inputs,
                            vart::TensorBuffer* output) {
  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<SoftmaxOpImp>();
}
