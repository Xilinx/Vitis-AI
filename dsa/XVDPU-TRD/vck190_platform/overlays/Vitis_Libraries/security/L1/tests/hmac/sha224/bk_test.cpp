
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

#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#define XF_SECURITY_DECRYPT_DEBUG 1
#include <openssl/sha.h>

// number of times to perform the test in different message and length
#define NUM_TESTS 2 // 200
// the result hash value in byte
//#define HASH_SIZE 16
// the size of each message word in byte
#define MSG_SIZE 4
// the size of the digest in byte
#define HASH_SIZE 28
// the max size of the message in byte
#define MAX_MSG 256

#include <ap_int.h>
#include <hls_stream.h>
#include "xf_security/hmac.hpp"
#include "xf_security/sha224_256.hpp"

#define KEYW (8 * MSG_SIZE)
#define MSGW (8 * MSG_SIZE)
#define HSHW (8 * HASH_SIZE)
#define BLOCK_SIZE 64

typedef ap_uint<64> u64;

template <int msgW, int hshW>
struct sha224_wrapper {
    static void hash(hls::stream<ap_uint<msgW> >& msgStrm,
                     hls::stream<u64>& lenStrm,
                     hls::stream<bool>& eLenStrm,
                     hls::stream<ap_uint<224> >& hshStrm,
                     hls::stream<bool>& eHshStrm) {
        xf::security::sha224<msgW>(msgStrm, lenStrm, eLenStrm, hshStrm, eHshStrm);
    }
};

void test_hmac_sha224(hls::stream<ap_uint<KEYW> >& keyStrm,
                      hls::stream<u64>& lenKeyStrm,
                      hls::stream<ap_uint<MSGW> >& msgStrm,
                      hls::stream<u64>& lenStrm,
                      hls::stream<bool>& eLenStrm,
                      hls::stream<ap_uint<HSHW> >& hshStrm,
                      hls::stream<bool>& eHshStrm) {
    xf::security::hmac<KEYW, MSGW, HSHW, BLOCK_SIZE, sha224_wrapper>(keyStrm, lenKeyStrm, msgStrm, lenStrm, eLenStrm,
                                                                     hshStrm, eHshStrm);
}

struct Test {
    std::string key;
    std::string msg;
    unsigned char hash[HASH_SIZE];
    Test(const char* k, const char* m, const void* h) : key(k), msg(m) { memcpy(hash, h, HASH_SIZE); }
};

// print hash value
std::string hash2str(unsigned char* h, int len) {
    std::ostringstream oss;
    std::string retstr;

    // check output
    oss.str("");
    oss << std::hex;
    for (int i = 0; i < len; i++) {
        oss << std::setw(2) << std::setfill('0') << (unsigned)h[i];
    }
    retstr = oss.str();
    return retstr;
}

template <int W>
unsigned int string2Strm(std::string data, std::string title, hls::stream<ap_uint<W> >& strm) {
    ap_uint<W> oneWord;
    unsigned int n = 0;
    unsigned int cnt = 0;
    // write msg stream word by word
    std::cout << title << " = " << data << "   len=" << data.length() << std::endl;
    for (std::string::size_type i = 0; i < data.length(); i++) {
        if (n == 0) {
            oneWord = 0;
        }
        oneWord.range(7 + 8 * n, 8 * n) = (unsigned)(data[i]);
        n++;
        if (n == W / 8) {
            strm.write(oneWord);
            std::cout << std::hex << oneWord;
            ++cnt;
            n = 0;
        }
    }
    // deal with the condition that we didn't hit a boundary of the last word
    if (n != 0) {
        strm.write(oneWord);
        std::cout << std::hex << oneWord;
        ++cnt;
    }
    std::cout << std::endl;
    return cnt;
}
// compute golden hmac
void hmacSHA224(const unsigned char* key,
                unsigned int keyLen,
                const unsigned char* message,
                unsigned int msgLen,
                unsigned char* h) {
    // hmac(key,msg) =  hash(
    //                         kopad^k1 || hash(
    //                                            (kipad^k1) || msg)
    //                                         )
    //                       )
    //
    //  k1 =  len(key) < blocksize ? {key,0} ? { hash(key),0 };
    //  The width of k1 is blocksize bits.

    unsigned char kone[BLOCK_SIZE + 8] = {0};
    unsigned char kipad[BLOCK_SIZE + 8] = {0};
    unsigned char kopad[BLOCK_SIZE + 8] = {0};
    unsigned char kmsg[BLOCK_SIZE + MAX_MSG + 8] = {0};
    unsigned char khsh[BLOCK_SIZE + HASH_SIZE + 8] = {0};
    unsigned char h1[HASH_SIZE + 8] = {0};
    unsigned char h2[HASH_SIZE + 8] = {0};

    if (keyLen > BLOCK_SIZE) {
        SHA224((const unsigned char*)key, keyLen, (unsigned char*)h1);
        memcpy(kone, h1, HASH_SIZE);
    } else
        memcpy(kone, key, keyLen);

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        kipad[i] = (unsigned int)(kone[i]) ^ 0x36;
        kopad[i] = (unsigned int)(kone[i]) ^ 0x5c;
    }
    std::cout << "main kipad:" << std::endl;
    hash2str(kipad, BLOCK_SIZE);
    std::cout << "main kopad:" << std::endl;
    hash2str(kopad, BLOCK_SIZE);
    memcpy(kmsg, kipad, BLOCK_SIZE);
    memcpy(kmsg + BLOCK_SIZE, message, msgLen);
    SHA224((const unsigned char*)kmsg, BLOCK_SIZE + msgLen, (unsigned char*)h2);

    std::cout << "main kmsg:" << std::endl;
    hash2str(kmsg, BLOCK_SIZE + msgLen);
    memcpy(khsh, kopad, BLOCK_SIZE);
    memcpy(khsh + BLOCK_SIZE, h2, HASH_SIZE);
    SHA224((const unsigned char*)khsh, BLOCK_SIZE + HASH_SIZE, (unsigned char*)h);
}

int main() {
    std::cout << "********************************" << std::endl;
    std::cout << "   Testing hmac+SHA224 on HLS project   " << std::endl;
    std::cout << "********************************" << std::endl;

    // the original message to be digested
    const char message[] = "The quick brown fox jumps over the lazy dog. Its hmac is 80070713463e7749b90c2dc24911e275";
    //	const char message[] = "The quick brown fox jumps over the lazy dog"; // Its hmac is
    // 80070713463e7749b90c2dc24911e275"; 	const char message[] = "ABCDEFGH";
    const char key[] = "key";
    std::vector<Test> tests;
    // generate golden
    for (unsigned int i = 0; i < NUM_TESTS; i++) {
        unsigned int len = i % 128;
        char k[128] = {0};
        char m[128] = {0};
        if (len != 0) {
            memcpy(k, key, len);
            memcpy(m, message, len);
        }
        k[len] = 0;
        m[len] = 0;
        unsigned char h[HASH_SIZE] = "";
        Test t(k, m, h);
        hmacSHA224((const unsigned char*)k, t.key.length(), (const unsigned char*)m, t.msg.length(), (unsigned char*)h);

        tests.push_back(Test(k, m, h));
    }
    unsigned int nerror = 0;
    unsigned int ncorrect = 0;

    hls::stream<ap_uint<KEYW> > keyStrm("keyStrm");
#pragma HLS stream variable = key_strm depth = 128
    hls::stream<ap_uint<64> > lenKeyStrm("lenKeyStrm");
#pragma HLS stream variable = lenKeyStrm depth = 128
    hls::stream<ap_uint<MSGW> > msgStrm("msgStrm");
#pragma HLS stream variable = msgStrm depth = 128
    hls::stream<ap_uint<64> > lenMsgStrm("lenMsgStrm");
#pragma HLS stream variable = lenMsgStrm depth = 128
    hls::stream<bool> endLenStrm("endLenStrm");
#pragma HLS stream variable = endLenStrm depth = 128
    hls::stream<ap_uint<HSHW> > hshStrm("hshStrm");
#pragma HLS stream variable = hshStrm depth = 128
    hls::stream<bool> endHshStrm("endHshStrm");
#pragma HLS stream variable = endHshStrm depth = 128

    // generate input message words
    for (std::vector<Test>::const_iterator test = tests.begin(); test != tests.end(); test++) {
        string2Strm<KEYW>((test->key), "key", keyStrm);
        string2Strm<MSGW>((test->msg), "msg", msgStrm);
        // inform the prmitive how many bytes do we have in this message
        lenKeyStrm.write((unsigned long long)((*test).key.length()));
        lenMsgStrm.write((unsigned long long)((*test).msg.length()));
        endLenStrm.write(false);
    }
    endLenStrm.write(true);
    // call fpga module
    test_hmac_sha224(keyStrm, lenKeyStrm, msgStrm, lenMsgStrm, endLenStrm, hshStrm, endHshStrm);

    // check result
    for (std::vector<Test>::const_iterator test = tests.begin(); test != tests.end(); test++) {
        ap_uint<HSHW> digest = hshStrm.read();
        bool x = endHshStrm.read();
        std::cout << "output:   " << std::endl;
        unsigned char hash[HSHW];
        for (unsigned int i = 0; i < HASH_SIZE; i++) {
            hash[i] = (unsigned char)(digest.range(7 + 8 * i, 8 * i).to_int() & 0xff);
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)(hash[i]);
        }

        if (memcmp((*test).hash, hash, HASH_SIZE)) {
            ++nerror;
            std::cout << "fpga   : " << hash2str((unsigned char*)hash, HASH_SIZE) << std::endl;
            std::cout << "golden : " << hash2str((unsigned char*)(*test).hash, HASH_SIZE) << std::endl;
        } else {
            ++ncorrect;
        }

        std::cout << std::endl;
    }

    bool x = endHshStrm.read();
    std::cout << std::endl;
    if (nerror) {
        std::cout << "FAIL: " << std::dec << nerror << " errors found." << std::endl;
    } else {
        std::cout << "PASS: " << std::dec << ncorrect << " inputs verified, no error found." << std::endl;
    }

    return nerror;
}
