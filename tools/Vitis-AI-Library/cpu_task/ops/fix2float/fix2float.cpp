#include <iostream>

#include "vart/op_imp.h"
using namespace std;

namespace {
class Fix2FloatOpImp : public vart::OpImp {
 public:
  explicit Fix2FloatOpImp(const xir::Op* op);
  virtual ~Fix2FloatOpImp();
  Fix2FloatOpImp(const Fix2FloatOpImp& other) = delete;
  Fix2FloatOpImp& operator=(const Fix2FloatOpImp& rhs) = delete;

 public:
  virtual int calculate(const std::vector<vart::OpImpArg>& inputs,
                        vart::TensorBuffer* output) override;
};

Fix2FloatOpImp::Fix2FloatOpImp(const xir::Op* op) : vart::OpImp(op){};
Fix2FloatOpImp::~Fix2FloatOpImp() {}
int Fix2FloatOpImp::calculate(const std::vector<vart::OpImpArg>& inputs,
                              vart::TensorBuffer* output) {
  LOG(INFO) << "hello " << inputs.size() << "output "
            << output->get_tensor()->get_name();

  return 0;
}

}  // namespace
extern "C" vart_op_imp_t vart_init_op_imp(const xir_op_t op) {
  return vart::make_vart_opt_imp<Fix2FloatOpImp>();
}
