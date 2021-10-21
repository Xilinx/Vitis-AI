.. 
   Copyright 2020 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


Regular Expression Acceleration
********************************

Getting Started
===============

In order to prepare the framework for use, it is first necessary to build it.
As mentioned in L1 regex-VM, we re-used VM instructions and compiler from popular `Oniguruma`_ library,
which is the foundation of current Ruby `regex implementation`_.

.. _`Oniguruma`: https://github.com/kkos/oniguruma.git

.. _`regex implementation`: https://github.com/k-takata/Onigmo

To fit the instructions into FPGA and achieve reasonable performance, we cannot simply move the original Oniguruma-like
OPs into our implementation without optimizing it.
Thus, the post-compiled OPs for FPGA is quite different from the one in Oniguruma.
A software compiler which is responsible for tranforming the post-compiled OPs to our desired instructions is
provided in ``L1/src/sw``.

Builds the ``libxfcompile.so`` (the software compiler) simply by:

.. code-block:: sh

    cd xf_DataAnalytics/L3/tests/re_test/re_compile
    make

After the build is complete, ``libxfcompile.so`` should be available in ``L3/tests/re_test/re_compile/lib/lib``.

Limitation
----------

We only support the ``match`` process in the hardware design, the return values should be a group of the start/end offset pairs if zero or more characters at the beginning of the string match the regular expression pattern.

The difference between ``match`` and ``search`` can be explained as:

.. NOTE::
    Pattern: "app"

    String: "Pineapple"

    Match result: Mismatch [-1, -1]

    Search result: Match [4, 7]

Example Usage
=============

At first, you have to set up the global parameters to specify the hardware and software scale you want to build.
Take those parameters in ``L3/tests/re_test/kernel/general_config.hpp`` as an example.

.. NOTE::

    The hardware related parameters will be used to generate the hardware ``reEngineKernel``.

    The software ones will be used to generate the L3 ``RegexEngine`` objects.

Secondly, you definitely want to compile the regex pattern to see if it is a valid one and supprted by our regex-VM
hardware before triggering the regex matching process.

.. code-block:: cpp
    
    // define your RE pattern
    std::string pattern = "your-regular-expression-pattern";
    // instantiate RegexEngine object
    xf::data_analytics::text::re::RegexEngine reInst(path_to_xclbin, 0, // device config
            instr_depth, cclass_nm, cpgp_nm, // re limits
            msg_sz, slice_sz, slice_nm); // processing needs
    // corresponding error code
    xf::data_analytics::text::re::ErrCode err_code;
    // compile RE pattern
    err_code = reInst.compile(pattern);
    // error occurs
    if (err_code != 0) {
        return -1;
    }
    
Then, you should allocate memory for each buffer.
The utilities are provided in ``L3/include/sw/xf_data_analytics/text``

.. code-block:: cpp
    
    // enumerations
    enum {
        // Max number of messages in a section
        MAX_MSG_DEPTH = 250000000,
        // Max length of message in byte
        // XXX: actual max supported message length is defined in general configuration (general_config.hpp)
        // This one is used to split the input into several processing sections
        MAX_MSG_LEN = 65536,
        // Max number of lines in a single section
        MAX_LNM = 6000000,
        // 20 for 19 capturing groups at most
        MAX_OUT_DEPTH = MAX_LNM * 20
    };

    // utility for allocating the buffers
    x_utils::MM mm;
    // message buffer
    uint64_t* msg_buff = mm.aligned_alloc<uint64_t>(MAX_MSG_DEPTH);
    // offset address buffer
    uint32_t* offt_buff = mm.aligned_alloc<uint32_t>(MAX_LNM);
    // length of each message buffer
    uint16_t* len_buff = mm.aligned_alloc<uint16_t>(MAX_LNM);
    // output buffer
    uint32_t* out_buff = mm.aligned_alloc<uint32_t>(MAX_OUT_DEPTH);


Feeds each buffer according to the format provided in ``L3/tests/re_test/host/main.cpp``,
and call the mathcing process by:

.. code-block:: cpp

    // make sure the number of length (lnm) is greater than 0
    err_code = reInst.match(lnm, msg_buff, offt_buff, len_buff, out_buff);

After the matching process complete, you'll get the corresponding results in ``out_buff`` with the format:

.. image:: /images/outbuff_format.png
    :alt: Result Buffer Format
    :width: 80%
    :align: center

Finally, do what you want with the results, like asserting whether a line of log is matched or extracting the captured
sub-strings with the begin/end offsets provided in each capturing group.
