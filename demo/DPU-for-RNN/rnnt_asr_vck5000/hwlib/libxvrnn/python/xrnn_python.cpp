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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <lstm_xrnn.h>

namespace py = pybind11;

static char* array_int8_to_pointer(py::array_t<int8_t>& a) {
  char* p = (char*)a.mutable_data();
  return p;
}

static char* array_int32_to_pointer(py::array_t<int32_t>& a) {
  char* p = (char*)a.mutable_data();
  return p;
}
static char* array_int16_to_pointer(py::array_t<int16_t>& a) {
  char* p = (char*)a.mutable_data();
  return p;
}

PYBIND11_MODULE(xrnn_py, m) {
  m.doc() = "lstm_xrnn";
  //m.def("XRNN", &xrnn::lstm_run, "run lstm xrnn");
  auto xrnn_runner = py::class_<xrnn>(m, "xrnn")
	  .def(py::init<char*>())

          .def("rnnt_reflash_ddr",
            [](xrnn* self){
                 self->rnnt_reflash_ddr();
               }
          )
          .def("rnnt_update_ddr",
            [](xrnn* self,
               py::array_t<int32_t> in,
               int in_s,
               unsigned int offset){
                 char* in_p = array_int32_to_pointer(in);
                 self->rnnt_update_ddr(in_p, 4*in_s, offset);
               }
          )
          .def("rnnt_download_ddr",
            [](xrnn* self,
               py::array_t<int8_t> out,
               int out_s,
               unsigned int offset){
                 char* out_p = array_int8_to_pointer(out);
                 self->rnnt_download_ddr(out_p, out_s, offset); 
               }
          )
	  .def("lstm_run",
	    [](xrnn* self,
	       py::array_t<int8_t> input,
	       int in_size,
	       py::array_t<int16_t> output,
	       int out_size,
	       int frame_num){
	         char* input_p = array_int8_to_pointer(input);
	         char* output_p = array_int16_to_pointer(output);
	         self->lstm_run(input_p, in_size, output_p, out_size*2, frame_num);
	       } 
	  );
}
