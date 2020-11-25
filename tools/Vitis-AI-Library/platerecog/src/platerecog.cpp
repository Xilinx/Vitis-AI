#include <vitis/ai/platerecog.hpp>

#include "./platerecog_imp.hpp"

namespace vitis {
namespace ai {
PlateRecog::PlateRecog() {}
PlateRecog::~PlateRecog() {}

std::unique_ptr<PlateRecog> PlateRecog::create(
    const std::string &platedetect_model, const std::string &platerecog_model,
    bool need_preprocess) {
  return std::unique_ptr<PlateRecog>(
      new PlateRecogImp(platedetect_model, platerecog_model, need_preprocess));
}
std::unique_ptr<PlateRecog> PlateRecog::create(
    const std::string &platedetect_model, const std::string &platerecog_model,
    xir::Attrs *attrs, bool need_preprocess) {
  return std::unique_ptr<PlateRecog>(
      new PlateRecogImp(platedetect_model, platerecog_model, attrs, need_preprocess));
}
}  // namespace ai
}  // namespace vitis
