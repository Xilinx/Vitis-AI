#include <cstdlib>
#include <string>
extern "C" const char* vitis_ai_getenv_s(const char* name);
namespace vitis {
namespace ai {
std::string my_getenv_s(const char* name,
                        const std::string& default_value = "") {
  auto ret = std::string();
  auto p = vitis_ai_getenv_s(name);
  if (p == nullptr) {
    ret = default_value;
  } else {
    ret = p;
#if _WIN32
    free((void*)p);
#endif
  }
  return ret;
}
}  // namespace ai
}  // namespace vitis
