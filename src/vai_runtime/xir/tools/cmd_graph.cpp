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

#include "cmd_graph.hpp"

#include <iostream>
#include <tuple>

using namespace std;
template <typename... T>
struct SupportedAttryTypes {
  using types = std::tuple<T...>;
};

using bytes_t = std::vector<int8_t>;

using ListOfSupportedAttryType1 = SupportedAttryTypes<
    /** begin supported list of attr types */
    bool,                                             //
    int8_t,                                           //
    uint8_t,                                          //
    int16_t, uint16_t,                                //
    int32_t, uint32_t,                                //
    int64_t, uint64_t,                                //
    float, double,                                    //
    std::string,                                      //
    bytes_t,                                          //
    std::vector<bool>,                                //
    std::vector<int8_t>, std::vector<uint8_t>,        //
    std::vector<int16_t>, std::vector<uint16_t>,      //
    std::vector<int32_t>, std::vector<uint32_t>,      //
    std::vector<int64_t>, std::vector<uint64_t>,      //
    std::vector<float>, std::vector<double>,          //
    std::vector<std::string>,                         //
    std::vector<bytes_t>,                             //
    std::map<std::string, int8_t>,                    //
    std::map<std::string, uint8_t>,                   //
    std::map<std::string, int16_t>,                   //
    std::map<std::string, uint16_t>,                  //
    std::map<std::string, int32_t>,                   //
    std::map<std::string, uint32_t>,                  //
    std::map<std::string, int64_t>,                   //
    std::map<std::string, uint64_t>,                  //
    std::map<std::string, float>,                     //
    std::map<std::string, double>,                    //
    std::map<std::string, std::string>,               //
    std::map<std::string, bytes_t>,                   //
    std::map<std::string, std::vector<bool>>,         //
    std::map<std::string, std::vector<int8_t>>,       //
    std::map<std::string, std::vector<uint8_t>>,      //
    std::map<std::string, std::vector<int16_t>>,      //
    std::map<std::string, std::vector<uint16_t>>,     //
    std::map<std::string, std::vector<int32_t>>,      //
    std::map<std::string, std::vector<uint32_t>>,     //
    std::map<std::string, std::vector<int64_t>>,      //
    std::map<std::string, std::vector<uint64_t>>,     //
    std::map<std::string, std::vector<float>>,        //
    std::map<std::string, std::vector<double>>,       //
    std::map<std::string, std::vector<std::string>>,  //
    std::map<std::string, std::vector<bytes_t>>,      //
    nullptr_t>;

using ListOfSupportedAttryType =
    SupportedAttryTypes<std::string,                        //
                        std::map<std::string, std::string>  //
                        >;

template <typename Op>
struct Apply {
  std::string do_it(const std::any& x) {
    return do_it(ListOfSupportedAttryType(), x);
  }
  template <typename T0, typename... T>
  std::string do_it(SupportedAttryTypes<T0, T...> tag, const std::any& x) {
    if (x.type() == typeid(T0)) {
      return Op()(std::any_cast<T0>(x));
    }
    return do_it(SupportedAttryTypes<T...>(), x);
  }
  std::string do_it(SupportedAttryTypes<> tag, const std::any& x) {
    return "NA";
  };
};

template <typename T, class = void>
struct is_cout_able : public std::false_type {};
template <typename T>
struct is_cout_able<
    T, std::void_t<decltype(declval<std::ostream&>() << declval<T>())>>
    : public std::true_type {};

template <typename T>
constexpr bool is_cout_able_v = is_cout_able<T>::value;

template <typename K, typename V>
std::enable_if_t<is_cout_able_v<K> && is_cout_able_v<V>, std::ostream&>
operator<<(std::ostream& out, const std::map<K, V>& v) {
  out << "{";
  for (const auto& x : v) {
    out << "\n\t\"" << x.first << "\" = " << x.second;
  }
  out << "\n}";
  return out;
}

struct ToString {
  template <typename T>
  std::enable_if_t<is_cout_able_v<T>, std::string> operator()(const T& x) {
    std::ostringstream str;
    str << x;
    return str.str();
  }
  template <typename T>
  std::enable_if_t<!is_cout_able_v<T>, std::string> operator()(const T& x) {
    std::ostringstream str;
    str << "unknwon type: " << typeid(x).name() << " and " << is_cout_able_v<T>;
    return str.str();
  }
};

static void show_attr(const xir::Attrs* attr) {
  for (auto key : attr->get_keys()) {
    cout << '"' << key
         << "\" = " << Apply<ToString>().do_it(attr->get_attr(key));
    cout << endl;
  }
}

CmdGraph::CmdGraph(const std::string& name) : Cmd(name) {}

int CmdGraph::main(int argc, char* argv[]) {
  auto xmodel = std::string(argv[1]);
  auto graph = xir::Graph::deserialize(xmodel);
  show_attr(graph->get_attrs().get());
  return 0;
}

std::string CmdGraph::help() const {
  std::ostringstream str;
  str << "xir " << get_name()
      << " <xmodel>\n\t"
         "show graph properties\n"
      << endl;
  return str.str();
}
