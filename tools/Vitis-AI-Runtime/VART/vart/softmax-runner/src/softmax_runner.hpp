// TODOL Licence
#include <memory>

#include "vart/runner.hpp"
#include "vart/tensor_buffer.hpp"
#include "xir/sfm_controller.hpp"
namespace vart {
class SoftmaxRunner : public vart::Runner {
 public:
  explicit SoftmaxRunner(const xir::Subgraph* subgraph, xir::Attrs* attrs);
  SoftmaxRunner(const SoftmaxRunner& other) = delete;

  virtual ~SoftmaxRunner();

 public:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() override;

 private:
  vart::TensorBuffer* prepare_input(vart::TensorBuffer* input);
  vart::TensorBuffer* prepare_output(vart::TensorBuffer* input);
  void start_controller(vart::TensorBuffer* input, vart::TensorBuffer* output);
  void finalize_output(vart::TensorBuffer* internal,
                       vart::TensorBuffer* output);

 private:
  const size_t device_core_id_ =
      0u;  // TODO: scheduler, fix device_core_id_ per runner
  std::shared_ptr<xir::SfmController> controller_;
  std::unique_ptr<vart::TensorBuffer> input_;
  std::unique_ptr<vart::TensorBuffer> output_;
};
}  // namespace vart
