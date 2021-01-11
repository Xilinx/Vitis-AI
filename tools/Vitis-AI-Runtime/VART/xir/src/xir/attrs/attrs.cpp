/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "xir/attrs/attrs.hpp"

#include <type_traits>

#include "xir/attrs/attrs_imp.hpp"

namespace xir {

std::unique_ptr<Attrs> Attrs::create() { return std::make_unique<AttrsImp>(); }

std::unique_ptr<Attrs> Attrs::clone(Attrs* param) {
  return std::make_unique<AttrsImp>(*static_cast<AttrsImp*>(param));
}

template <typename T>
static bool cmp(const any& a, const any& b) {
  return stdx::any_cast<const T&>(a) == stdx::any_cast<const T&>(b);
}

template <typename... T>
static std::unordered_map<std::type_index,
                          std::function<bool(const any&, const any&)>>
make_cmp_fun() {
  return {
      make_pair(std::type_index(typeid(T)), &cmp<T>)...,
  };
}

static std::unordered_map<std::type_index,
                          std::function<bool(const any&, const any&)>>
get_cmp_functions() {
  using bytes_t = std::vector<int8_t>;
  return make_cmp_fun<  //
      bool, int8_t,
      uint8_t,                                                          //
      int16_t, uint16_t,                                                //
      int32_t, uint32_t,                                                //
      int64_t, uint64_t,                                                //
      float, double,                                                    //
      std::string,                                                      //
      bytes_t,                                                          //
      std::vector<bool>,                                                //
      std::vector<int8_t>, std::vector<uint8_t>,                        //
      std::vector<int16_t>, std::vector<uint16_t>,                      //
      std::vector<int32_t>, std::vector<uint32_t>,                      //
      std::vector<int64_t>, std::vector<uint64_t>,                      //
      std::vector<float>, std::vector<double>,                          //
      std::vector<std::string>,                                         //
      std::vector<bytes_t>,                                             //
      std::map<std::string, int8_t>, std::map<std::string, uint8_t>,    //
      std::map<std::string, int16_t>, std::map<std::string, uint16_t>,  //
      std::map<std::string, int32_t>, std::map<std::string, uint32_t>,  //
      std::map<std::string, int64_t>, std::map<std::string, uint64_t>,  //
      std::map<std::string, float>, std::map<std::string, double>,      //
      std::map<std::string, std::string>,                               //
      std::map<std::string, bytes_t>,                                   //
      std::map<std::string, std::vector<bool>>,                         //
      std::map<std::string, std::vector<int8_t>>,                       //
      std::map<std::string, std::vector<uint8_t>>,                      //
      std::map<std::string, std::vector<int16_t>>,                      //
      std::map<std::string, std::vector<uint16_t>>,                     //
      std::map<std::string, std::vector<int32_t>>,                      //
      std::map<std::string, std::vector<uint32_t>>,                     //
      std::map<std::string, std::vector<int64_t>>,                      //
      std::map<std::string, std::vector<uint64_t>>,                     //
      std::map<std::string, std::vector<float>>,                        //
      std::map<std::string, std::vector<double>>,                       //
      std::map<std::string, std::vector<std::string>>,                  //
      std::map<std::string, std::vector<bytes_t>>,                      //
      nullptr_t>();
}

int Attrs::cmp(const any& a, const any& b) {
  auto t1 = std::type_index(a.type());
  auto t2 = std::type_index(b.type());
  if (t1 != t2) {
    return 0;
  }
  auto cmp_functions = get_cmp_functions();
  auto it_cmp = cmp_functions.find(t1);
  if (it_cmp == cmp_functions.end()) {
    return -1;  // uncertain
  }
  return it_cmp->second(a, b);
}
}  // namespace xir

// C API implementations
#include <cstring>

#include "xir/xir.h"
extern "C" xir_attrs_t xir_attrs_create() {
  return static_cast<xir_attrs_t>(xir::Attrs::create().release());
}

extern "C" void xir_attrs_destroy(xir_attrs_t attrs) {
  delete static_cast<xir::Attrs*>(attrs);
}

// p(type_name, c_type, c++ type )
template <typename from_t, typename to_t>
struct convert {
  static to_t conv(const from_t& value) { return value; }
};

template <>
struct convert<xir_string_t, std::string> {
  static std::string conv(xir_string_t value) { return std::string(value); }
};
template <>
struct convert<std::string, xir_string_t> {
  static xir_string_t conv(const std::string& value) { return value.c_str(); }
};
template <>
struct convert<xir_bytes_t, std::vector<int8_t>> {
  static std::vector<int8_t> conv(const xir_bytes_t& value) {
    auto ret = std::vector<int8_t>(value.size);
    std::memcpy(&ret[0], value.data, value.size);
    return ret;
  }
};
template <>
struct convert<std::vector<int8_t>, xir_bytes_t> {
  static xir_bytes_t conv(const std::vector<int8_t>& value) {
    return xir_bytes_t{(int8_t*)(&value[0]), value.size()};
  }
};
template <>
struct convert<bool, int> {
  static bool conv(const int& value) { return value != 0; }
};

#define IMPL_GET(type_name, c_type, cxx_type)                                  \
  extern "C" c_type xir_attrs_get_##type_name(xir_attrs_t attrs,               \
                                              const char* name) {              \
    return convert<cxx_type, c_type>::conv(                                    \
        static_cast<xir::Attrs*>(attrs)->get_attr<const cxx_type&>(            \
            std::string(name)));                                               \
  }
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_GET);

#define IMPL_SET(type_name, c_type, cxx_type)                                  \
  extern "C" void xir_attrs_set_##type_name(xir_attrs_t attrs,                 \
                                            const char* name, c_type value) {  \
    static_cast<xir::Attrs*>(attrs)->set_attr<cxx_type>(                       \
        std::string(name), convert<c_type, cxx_type>::conv(value));            \
  }
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_SET);

#define IMPL_HAS(type_name, c_type, cxx_type)                                  \
  extern "C" int xir_attrs_has_##type_name(xir_attrs_t attrs,                  \
                                           const char* name) {                 \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    return self->has_attr(std::string(name)) &&                                \
           self->get_attr(std::string(name)).type() == typeid(cxx_type);       \
  }
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_HAS);

extern int xir_attrs_has_attr(xir_attrs_t attrs, const char* key) {
  return static_cast<xir::Attrs*>(attrs)->has_attr(std::string(key));
}

#define IMPL_VEC_GET_SIZE(type_name, c_type, cxx_type)                         \
  extern "C" size_t xir_attrs_get_vec_size_##type_name(xir_attrs_t attrs,      \
                                                       const char* name) {     \
    return static_cast<xir::Attrs*>(attrs)                                     \
        ->get_attr<const std::vector<cxx_type>&>(std::string(name))            \
        .size();                                                               \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_VEC_GET_SIZE);

#define IMPL_VEC_GET(type_name, c_type, cxx_type)                              \
  extern "C" c_type xir_attrs_get_vec_##type_name(                             \
      xir_attrs_t attrs, const char* name, size_t idx) {                       \
    return convert<cxx_type, c_type>::conv(                                    \
        static_cast<xir::Attrs*>(attrs)                                        \
            ->get_attr<const std::vector<cxx_type>&>(std::string(name))[idx]); \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_VEC_GET);
#define IMPL_VEC_SET(type_name, c_type, cxx_type)                              \
  extern "C" void xir_attrs_set_vec_##type_name(                               \
      xir_attrs_t attrs, const char* name, size_t idx, c_type value) {         \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    if (!self->has_attr(std::string(name))) {                                  \
      self->set_attr<std::vector<cxx_type>>(std::string(name),                 \
                                            std::vector<cxx_type>(idx + 1u));  \
    }                                                                          \
    std::vector<cxx_type>& vec = const_cast<std::vector<cxx_type>&>(           \
        self->get_attr<const std::vector<cxx_type>&>(std::string(name)));      \
    if (idx >= vec.size()) {                                                   \
      vec.resize(idx + 1u);                                                    \
    }                                                                          \
    vec[idx] = convert<c_type, cxx_type>::conv(value);                         \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_VEC_SET);

#define IMPL_VEC_HAS(type_name, c_type, cxx_type)                              \
  extern "C" int xir_attrs_has_vec_##type_name(xir_attrs_t attrs,              \
                                               const char* name) {             \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    return self->has_attr(std::string(name)) &&                                \
           self->get_attr(std::string(name)).type() ==                         \
               typeid(std::vector<cxx_type>);                                  \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_VEC_HAS);

#define IMP_XIR_ATTRS_GET_MAP_SIZE(type_name, c_type, cxx_type)                \
  extern "C" size_t xir_attrs_get_map_size_##type_name(xir_attrs_t attrs,      \
                                                       const char* name) {     \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    return self                                                                \
        ->get_attr<const std::map<std::string, cxx_type>&>(std::string(name))  \
        .size();                                                               \
  }
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMP_XIR_ATTRS_GET_MAP_SIZE)

#define IMP_XIR_ATTRS_GET_MAP_KEYS(type_name, c_type, cxx_type)                \
  extern "C" void xir_attrs_get_map_keys_##type_name(                          \
      xir_attrs_t attrs, const char* name, const char* keys[]) {               \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    auto& m = self->get_attr<const std::map<std::string, cxx_type>&>(          \
        std::string(name));                                                    \
    int c = 0;                                                                 \
    for (auto& k : m) {                                                        \
      keys[c++] = k.first.c_str();                                             \
    }                                                                          \
  }
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMP_XIR_ATTRS_GET_MAP_KEYS)

#define IMPL_MAP_GET(type_name, c_type, cxx_type)                              \
  extern "C" c_type xir_attrs_get_map_##type_name(                             \
      xir_attrs_t attrs, const char* name, const char* key) {                  \
    auto& it = (static_cast<xir::Attrs*>(attrs)                                \
                    ->get_attr<const std::map<std::string, cxx_type>&>(        \
                        std::string(name))                                     \
                    .at(std::string(key)));                                    \
    return convert<cxx_type, c_type>::conv(it);                                \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_MAP_GET);
#define IMPL_MAP_SET(type_name, c_type, cxx_type)                              \
  extern "C" void xir_attrs_set_map_##type_name(                               \
      xir_attrs_t attrs, const char* name, const char* key, c_type value) {    \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    if (!self->has_attr(std::string(name))) {                                  \
      self->set_attr<std::map<std::string, cxx_type>>(                         \
          std::string(name), std::map<std::string, cxx_type>());               \
    }                                                                          \
    std::map<std::string, cxx_type>& m =                                       \
        const_cast<std::map<std::string, cxx_type>&>(                          \
            self->get_attr<const std::map<std::string, cxx_type>&>(            \
                std::string(name)));                                           \
    m[std::string(key)] = convert<c_type, cxx_type>::conv(value);              \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_MAP_SET);

#define IMPL_MAP_HAS(type_name, c_type, cxx_type)                              \
  extern "C" int xir_attrs_has_map_##type_name(xir_attrs_t attrs,              \
                                               const char* name) {             \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    return self->has_attr(std::string(name)) &&                                \
           self->get_attr(std::string(name)).type() ==                         \
               typeid(std::map<std::string, cxx_type>);                        \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_MAP_HAS);

#define IMP_XIR_ATTRS_GET_MAP_VEC_MSIZE(type_name, c_type, cxx_type)           \
  extern "C" size_t xir_attrs_get_map_vec_msize_##type_name(                   \
      xir_attrs_t attrs, const char* name) {                                   \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    return self                                                                \
        ->get_attr<const std::map<std::string, std::vector<cxx_type>>&>(       \
            std::string(name))                                                 \
        .size();                                                               \
  }
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMP_XIR_ATTRS_GET_MAP_VEC_MSIZE)

#define IMP_XIR_ATTRS_GET_MAP_VEC_VSIZE(type_name, c_type, cxx_type)           \
  extern "C" size_t xir_attrs_get_map_vec_vsize_##type_name(                   \
      xir_attrs_t attrs, const char* name, const char* key) {                  \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    return self                                                                \
        ->get_attr<const std::map<std::string, std::vector<cxx_type>>&>(       \
            std::string(name))                                                 \
        .at(std::string(key))                                                  \
        .size();                                                               \
  }
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMP_XIR_ATTRS_GET_MAP_VEC_VSIZE)

#define IMP_XIR_ATTRS_GET_MAP_VEC_KEYS(type_name, c_type, cxx_type)            \
  extern "C" void xir_attrs_get_map_vec_keys_##type_name(                      \
      xir_attrs_t attrs, const char* name, const char* keys[]) {               \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    auto& m =                                                                  \
        self->get_attr<const std::map<std::string, std::vector<cxx_type>>&>(   \
            std::string(name));                                                \
    int c = 0;                                                                 \
    for (auto& k : m) {                                                        \
      keys[c++] = k.first.c_str();                                             \
    }                                                                          \
  }
XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMP_XIR_ATTRS_GET_MAP_VEC_KEYS)

#define IMPL_MAP_VEC_GET(type_name, c_type, cxx_type)                          \
  extern "C" c_type xir_attrs_get_map_vec_##type_name(                         \
      xir_attrs_t attrs, const char* name, const char* key, size_t idx) {      \
    auto& it =                                                                 \
        (static_cast<xir::Attrs*>(attrs)                                       \
             ->get_attr<const std::map<std::string, std::vector<cxx_type>>&>(  \
                 std::string(name))                                            \
             .at(std::string(key)));                                           \
    return convert<cxx_type, c_type>::conv(it[idx]);                           \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_MAP_VEC_GET);
#define IMPL_MAP_VEC_SET(type_name, c_type, cxx_type)                          \
  extern "C" void xir_attrs_set_map_vec_##type_name(                           \
      xir_attrs_t attrs, const char* name, const char* key, size_t idx,        \
      c_type value) {                                                          \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    if (!self->has_attr(std::string(name))) {                                  \
      self->set_attr<std::map<std::string, std::vector<cxx_type>>>(            \
          std::string(name), std::map<std::string, std::vector<cxx_type>>());  \
    }                                                                          \
    std::map<std::string, std::vector<cxx_type>>& m = const_cast<              \
        std::map<std::string, std::vector<cxx_type>>&>(                        \
        self->get_attr<const std::map<std::string, std::vector<cxx_type>>&>(   \
            std::string(name)));                                               \
    const auto& key_in_cxx = std::string(key);                                 \
    auto it = m.find(key_in_cxx);                                              \
    if (it == m.end()) {                                                       \
      bool ok = false;                                                         \
      std::tie(it, ok) =                                                       \
          m.insert({key_in_cxx, std::vector<cxx_type>(idx + 1u)});             \
    }                                                                          \
    auto& v = it->second;                                                      \
    if (idx >= v.size()) {                                                     \
      v.resize(idx + 1u);                                                      \
    }                                                                          \
    v[idx] = convert<c_type, cxx_type>::conv(value);                           \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_MAP_VEC_SET);

#define IMPL_MAP_VEC_HAS(type_name, c_type, cxx_type)                          \
  extern "C" int xir_attrs_has_map_vec_##type_name(xir_attrs_t attrs,          \
                                                   const char* name) {         \
    auto self = static_cast<xir::Attrs*>(attrs);                               \
    return self->has_attr(std::string(name)) &&                                \
           self->get_attr(std::string(name)).type() ==                         \
               typeid(std::map<std::string, std::vector<cxx_type>>);           \
  }

XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES(IMPL_MAP_VEC_HAS);
