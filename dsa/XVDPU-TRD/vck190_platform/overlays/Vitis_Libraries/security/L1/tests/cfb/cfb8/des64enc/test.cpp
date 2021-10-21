/*
 * Copyright 2019 Xilinx, Inc.
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

#include "test.hpp"
#include "xf_security/cfb.hpp"

void test(hls::stream<ap_uint<64> >& plaintext,
          hls::stream<bool>& plaintext_e,
          hls::stream<ap_uint<64> >& cipherkey,
          hls::stream<ap_uint<64> >& initialization_vector,
          hls::stream<ap_uint<64> >& ciphertext,
          hls::stream<bool>& ciphertext_e) {
    xf::security::desCfb8Encrypt(plaintext, plaintext_e, cipherkey, initialization_vector, ciphertext, ciphertext_e);
}
