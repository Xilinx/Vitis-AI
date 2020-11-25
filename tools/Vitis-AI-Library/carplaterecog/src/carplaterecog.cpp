#include <vitis/ai/carplaterecog.hpp>

#include "./carplaterecog_imp.hpp"

namespace vitis {
namespace ai {
CarPlateRecog::CarPlateRecog() {}
CarPlateRecog::~CarPlateRecog() {}

std::unique_ptr<CarPlateRecog> CarPlateRecog::create(
    const std::string &cardetect_model, const std::string &platedetect_model, const std::string &carplaterecog_model,
    bool need_preprocess) {
  return std::unique_ptr<CarPlateRecog>(
      new CarPlateRecogImp(cardetect_model, platedetect_model, carplaterecog_model, need_preprocess));
}

std::unique_ptr<CarPlateRecog> CarPlateRecog::create(
    const std::string &cardetect_model, const std::string &platedetect_model, const std::string &carplaterecog_model, xir::Attrs *attrs,
    bool need_preprocess) {
  return std::unique_ptr<CarPlateRecog>(
      new CarPlateRecogImp(cardetect_model, platedetect_model, carplaterecog_model, attrs, need_preprocess));

}
}  // namespace ai
}  // namespace vitis
