/*
 * Copyright (C) 2020 - 2021 Xilinx, Inc.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * XILINX BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of the Xilinx shall not be used
 * in advertising or otherwise to promote the sale, use or other dealings in
 * this Software without prior written authorization from Xilinx.
 */ 

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp> 
#ifndef __XAPM_HPP__
#include "xperfmon.hpp"
#endif

#define PFM_1_CLK 399990000
#define PFM_2_CLK 399990000
#define PFM_3_CLK 399990000

using namespace boost::python;

std::vector<apm_t> toVector(list l){ 
  std::vector<apm_t> a;
  for (int i = 0; i < len(l); i++){
    apm_t x = {.addr = extract<uintptr_t>(l[i][0]), 
           .slot = extract<size_t>(l[i][1]), 
           .msr_read = extract<uint32_t>(l[i][2]), 
           .msr_write = extract<uint32_t>(l[i][3]), 
           .smcr = extract<uint32_t>(l[i][4]),
       	   .apm_num = extract<uint32_t>(l[i][5]),
       	   .clock_freq = extract<uint32_t>(l[i][6])};
    a.push_back(x);
  }
  return a;        
}

boost::python::list setup(uint32_t core_clk){
	uint32_t base_addr = 0xA4050000;
	std::vector<uint32_t> HDMI_VMIX_SMC_0_RD = {base_addr, 0, 0x44, 0x03, 0x200, 0, core_clk};
	std::vector<uint32_t> HDMI_VMIX_SMC_1_RD = {base_addr, 1, 0x44, 0x2300, 0x210, 0, core_clk};
	std::vector<uint32_t> PL_ACCEL_RD = {base_addr, 2, 0x44, 0x630000, 0x220, 0, core_clk};
	std::vector<uint32_t> PL_ACCEL_WR = {base_addr, 3, 0x44, 0x62000000, 0x230, 0, core_clk};
	std::vector<uint32_t> AIE_ACCEL_RD = {base_addr, 4, 0x48, 0x83, 0x240, 0, core_clk};
	std::vector<uint32_t> AIE_ACCEL_WR = {base_addr, 5, 0x48, 0xC200, 0x250, 0, core_clk};
	std::vector<uint32_t> MIPI_WRITE = {base_addr, 6, 0x48, 0xA20000, 0x260, 0, core_clk};
	std::vector<uint32_t> HDMI_PRIMARY_PLANE_RD = {base_addr, 7, 0x48, 0x43000000, 0x270, 0, core_clk};
	std::vector<std::vector<uint32_t>> preset = {
		HDMI_VMIX_SMC_0_RD,
		HDMI_VMIX_SMC_1_RD,
		PL_ACCEL_RD,
		PL_ACCEL_WR,
		AIE_ACCEL_RD,
		AIE_ACCEL_WR,
		MIPI_WRITE,
		HDMI_PRIMARY_PLANE_RD
	};
	boost::python::list fin;
	for (int i = 0; i < preset.size(); i++){
		boost::python::list temp;
		for (int j = 0; j < preset[i].size(); j++){
			temp.append(preset[i][j]);
		}
		fin.append(temp);
	}

	return fin;
}

BOOST_PYTHON_MODULE(libxperfmon)
{
    class_<APM>("APM", init<list>())
       .def_readwrite("port", &APM::port)
       .def("getThroughput", &APM::getThroughput)
    ;
    scope().attr("preset_pfm1") = setup(PFM_1_CLK);
    scope().attr("preset_pfm2") = setup(PFM_2_CLK);
    scope().attr("preset_pfm3") = setup(PFM_3_CLK);
    scope().attr("MB/s") = 0;
    scope().attr("Gbps") = 1;

}
