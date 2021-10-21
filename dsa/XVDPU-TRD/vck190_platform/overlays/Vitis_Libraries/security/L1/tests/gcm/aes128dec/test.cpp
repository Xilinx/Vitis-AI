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
#include "xf_security/gcm.hpp"

void test(hls::stream<ap_uint<128> >& ciphertext,
          hls::stream<ap_uint<128> >& cipherkey,
          hls::stream<ap_uint<96> >& IV,
          hls::stream<ap_uint<128> >& AAD,
          hls::stream<ap_uint<64> >& AAD_length,
          hls::stream<ap_uint<64> >& ciphertext_length,
          hls::stream<bool>& end_length,
          hls::stream<ap_uint<128> >& plaintext,
          hls::stream<ap_uint<64> >& plaintext_length,
          hls::stream<ap_uint<128> >& tag,
          hls::stream<bool>& end_tag) {
    xf::security::aes128GcmDecrypt(ciphertext, cipherkey, IV, AAD, AAD_length, ciphertext_length, end_length, plaintext,
                                   plaintext_length, tag, end_tag);
}
