/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

// if included by op_def.hpp, it is not sued.
static inline bool cmp_type_id(const std::type_info& i1,
                               const std::type_info& i2) {
  auto ret = i1 == i2;
#ifndef NDEBUG
  auto ret2 = std::type_index(i1) == std::type_index(i2);
  auto ret3 = i1.name() == i2.name();
  // for debugging purpose
  LOG_IF(INFO, false) << i1.name() << " == " << i2.name() << " => " << ret
                      << " " << ret2 << ' ' << ret3;
#endif
  return ret;
}

[[maybe_unused]] xir_attr_value_t convert<any, xir_attr_value_t>::conv(
    const any& value) {
  auto& type = value.type();
  xir_attr_value_t ret;
  if (0) {
  }

#define IMP_GET_IMP(name, c_type, cxx_type)                                    \
  else if (cmp_type_id(typeid(cxx_type), type)) {                              \
    ret.tag = XIR_ATTR_TYPE_TAG_##name;                                        \
    ret.u.name##_value =                                                       \
        convert<cxx_type, c_type>::conv(any_cast<const cxx_type&>(value));     \
  }
#define IMP_GET_IMP_VEC(name, c_type, cxx_type)                                \
  else if (typeid(vector<cxx_type>) == type) {                                 \
    ret.tag = XIR_ATTR_TYPE_TAG_VEC_##name;                                    \
    ret.u.vec_value = to_xir_iter_t<cxx_type>::conv(                           \
        any_cast<const vector<cxx_type>&>(value));                             \
  }

#define IMP_GET_IMP_MAP(name, c_type, cxx_type)                                \
  else if (cmp_type_id(typeid(map<string, cxx_type>), type)) {                 \
    ret.tag = XIR_ATTR_TYPE_TAG_MAP_##name;                                    \
    ret.u.map_value = to_xir_map_iter_t<cxx_type>::conv(                       \
        any_cast<const map<string, cxx_type>&>(value));                        \
  }

#define IMP_GET_IMP_MAP_VEC(name, c_type, cxx_type)                            \
  else if (typeid(map<string, vector<cxx_type>>) == type) {                    \
    ret.tag = XIR_ATTR_TYPE_TAG_MAP_VEC_##name;                                \
    ret.u.map_value = to_xir_map_iter_t<vector<cxx_type>>::conv(               \
        any_cast<const map<string, vector<cxx_type>>&>(value));                \
  }

  XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(IMP_GET_IMP)
  XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(IMP_GET_IMP_VEC)
  XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(IMP_GET_IMP_MAP)
  XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(IMP_GET_IMP_MAP_VEC)

  else {
    ret.tag = XIR_ATTR_TYPE_TAG_NONE;
    // suppress coverity complain
    ret.u.map_value = nullptr;
  }
  return ret;
}

any convert<xir_attr_value_t, any>::conv(xir_attr_value_t value) {
  auto ret = any();
  if (0) {
  }
#define IMP_SET_IMP(name, c_type, cxx_type)                                    \
  else if (value.tag == XIR_ATTR_TYPE_TAG_##name) {                            \
    ret = convert<xir_attr_value_t, cxx_type>::conv(value);                    \
  }                                                                            \
  else if (value.tag == XIR_ATTR_TYPE_TAG_VEC_##name) {                        \
    ret = to_vec<cxx_type>::conv(value.u.vec_value);                           \
  }                                                                            \
  else if (value.tag == XIR_ATTR_TYPE_TAG_MAP_##name) {                        \
    ret = to_map<cxx_type>::conv(value.u.map_value);                           \
  }                                                                            \
  else if (value.tag == XIR_ATTR_TYPE_TAG_MAP_VEC_##name) {                    \
    ret = to_map<vector<cxx_type>>::conv(value.u.map_value);                   \
  }
  XIR_ATTRS_SUPPORTED_PRIMITIVE_TYPES2(IMP_SET_IMP)
  else {
  }
  return ret;
}

template <typename cxx_type>
xir_attr_value_t convert<vector<cxx_type>, xir_attr_value_t>::conv(
    const vector<cxx_type>& value) {
  xir_attr_value_t ret;
  ret.tag = type_hint_t<cxx_type>::vec_tag;
  ret.u.vec_value = to_xir_iter_t<cxx_type>::conv(value);
  return ret;
}

template <typename cxx_type>
vector<cxx_type> convert<xir_attr_value_t, vector<cxx_type>>::conv(
    xir_attr_value_t value) {
  return to_vec<cxx_type>::conv(value.u.vec_value);
};

// Local Variables:
// mode:c++
// coding: utf-8-unix
// End:
