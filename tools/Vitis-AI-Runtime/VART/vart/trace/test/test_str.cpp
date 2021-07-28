#include <iostream>
#include "str.hpp"

int main(void) {
  auto a = vitis::ai::trace::str("aaa");
  auto b = vitis::ai::trace::str("alsdfkajsdf");
  auto c = vitis::ai::trace::str("777");
  auto d = vitis::ai::trace::str("aaa");

  std::cout << "pool size: " << vitis::ai::trace::str_pool_size() << std::endl;
  std::cout << "size: " << sizeof(a) << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << a.to_string() << std::endl;
  std::cout << b.to_string() << std::endl;
  std::cout << c.to_string() << std::endl;
}
