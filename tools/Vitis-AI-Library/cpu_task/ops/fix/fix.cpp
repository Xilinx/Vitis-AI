#include <iostream>

#include "vart/op_imp.h"
using namespace std;

namespace {
class fix_OpImp : public vart::OpImp {
 public:
  explicit fix_OpImp(const xir::Op* op);
  virtual ~fix_OpImp();
  fix_OpImp(const fix_OpImp& other) = delete;
  fix_OpImp& operator=(const fix_OpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

fix_OpImp::fix_OpImp(const xir::Op* op) : vart::OpImp(op){};
fix_OpImp::~fix_OpImp() {}
int fix_OpImp::calculate(const std::vector<vart::OpImpArg>& inputs,
                         vart::TensorBuffer* output) {
  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<fix_OpImp>();
}
