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

#ifndef __SYNTHESIS__
#include <new>
#include <cstdlib>
#endif

#include "XAcc_jpegdecoder.hpp"
#include "XAcc_jfifparser.hpp"

// ------------------------------------------------------------
#ifndef __SYNTHESIS__

#if __linux
template <typename T>
char* aligned_alloc(std::size_t num) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, num * sizeof(char))) throw std::bad_alloc();
    return reinterpret_cast<char*>(ptr);
}
#endif
// ------------------------------------------------------------

// load the data file (.txt, .bin, .jpg ...)to ptr
template <typename T>
int load_dat(T*& data, const std::string& name, int& size) {
    uint64_t n;
    std::string fn = name;
    FILE* f = fopen(fn.c_str(), "rb");
    std::cout << "WARNING: " << fn << " will be opened for binary read." << std::endl;
    if (!f) {
        std::cerr << "ERROR: " << fn << " cannot be opened for binary read." << std::endl;
        return -1;
    }

    fseek(f, 0, SEEK_END);
    n = (uint64_t)ftell(f);
    if (n > MAX_DEC_PIX) {
        return 1;
    }
    // data = (T*)aligned_alloc(n);
    data = (T*)malloc(MAX_DEC_PIX);
    fseek(f, 0, SEEK_SET);
    size = fread(data, sizeof(char), n, f);
    fclose(f);
    std::cout << n << " entries read from " << fn << std::endl;

    return 0;
}

// ------------------------------------------------------------
// get the arg
#include <algorithm>
#include <string>
#include <vector>
class ArgParser {
   public:
    ArgParser(int& argc, const char* argv[]) {
        for (int i = 1; i < argc; ++i) mTokens.push_back(std::string(argv[i]));
    }
    bool getCmdOption(const std::string option, std::string& value) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end() && ++itr != this->mTokens.end()) {
            value = *itr;
            return true;
        }
        return false;
    }
    bool getCmdOption(const std::string option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->mTokens.begin(), this->mTokens.end(), option);
        if (itr != this->mTokens.end())
            return true;
        else
            return false;
    }

   private:
    std::vector<std::string> mTokens;
};

// ------------------------------------------------------------
// for tmp application
int16_t* hls_block = (int16_t*)malloc(sizeof(int16_t) * MAX_NUM_COLOR * MAXCMP_BC * 64);

// ************************************************************
int main(int argc, const char* argv[]) {
    std::cout << "\n------------ Test for decode image.jpg  -------------\n";
    std::string optValue;
    std::string JPEGFile;
    std::string in_dir = "./"; // no use by now

    // cmd arg parser.
    ArgParser parser(argc, argv);

    if (parser.getCmdOption("-JPEGFile", optValue)) {
        JPEGFile = optValue;
    } else {
        std::cout << "WARNING: JPEG file not specified for this test. use "
                     "'-JPEGFile' to specified it. \n";
    }

    // load data to simulate the ddr data
    int size;
    uint8_t* datatoDDR;

    int err = load_dat(datatoDDR, JPEGFile, size);
    if (err) {
        printf("Alloc buf failed!\n");
        return err;
    }

    // call SYNTHESIS top
    hls::stream<ap_uint<24> > block_strm;
    xf::codec::hls_compInfo hls_cmpnfo[MAX_NUM_COLOR];
    uint8_t hls_mbs[MAX_NUM_COLOR];

    // 0: decode jfif successful
    // 1: marker in jfif is not in expectation
    // 2: huffman table is not in expectation
    int rtn = 0;

    // 0: decode huffman successful
    // 1: huffman data is not in expectation
    bool rtn2 = false;

    uint32_t hls_mcuc;
    uint16_t hls_mcuh;
    uint16_t hls_mcuv;
    uint8_t hls_cs_cmpc;
    xf::codec::img_info imgInfo;
    xf::codec::decOutput pout;
    imgInfo.hls_cs_cmpc = 0; // init
    // L1 top
    // parser_jpg_top((ap_uint<CH_W>*)datatoDDR, (int)size, hls_mcuc, hls_cmpnfo, block_strm, rtn);
    // L2 top
    kernel_parser_decoder((ap_uint<CH_W>*)datatoDDR, (int)size, imgInfo, hls_cmpnfo, block_strm, rtn, rtn2, &pout);

    // for image info
    int hls_sfv[MAX_NUM_COLOR];
    int hls_sfh[MAX_NUM_COLOR];
    // int hls_mbs[MAX_NUM_COLOR];
    int hls_bcv[MAX_NUM_COLOR];
    int hls_bch[MAX_NUM_COLOR];
    int hls_bc[MAX_NUM_COLOR];
    for (int i = 0; i < MAX_NUM_COLOR; i++) {
#pragma HLS PIPELINE II = 1
        hls_sfv[i] = hls_cmpnfo[i].sfv;
        hls_sfh[i] = hls_cmpnfo[i].sfh;
        hls_mbs[i] = hls_cmpnfo[i].mbs;
        hls_bcv[i] = hls_cmpnfo[i].bcv;
        hls_bch[i] = hls_cmpnfo[i].bch;
        hls_bc[i] = hls_cmpnfo[i].bc;
    }

    // todo merge to syn-code
    int status = 0;
    if (rtn || rtn2) {
        status = 2;
        printf("Warning: Decoding the bad case input file!\n");
        if (rtn == 1) {
            printf("Warning: [code 1] marker in jfif is not in expectation!\n");
        } else if (rtn == 2) {
            printf("Warning: [code 2] huffman table is not in expectation!\n");
        } else {
            if (rtn2) {
                printf("Warning: [code 3] huffman data is not in expectation!\n");
            }
        }
        printf("Info: Ready to decode next input file!\n");
    }

    xf::codec::details::hls_next_mcupos2(block_strm, hls_block, hls_sfv, hls_sfh, hls_mbs, hls_bch[0], hls_bc[0],
                                         imgInfo.hls_mcuc, imgInfo.hls_cs_cmpc, rtn2, status);

    int k_dpos[3];

    printf("****the end 3 blocks before zigzag are : \n");
    for (int i_cmp = 0; i_cmp < imgInfo.hls_cs_cmpc; i_cmp++) {
        k_dpos[i_cmp] = hls_cmpnfo[i_cmp].bc - 1;
        for (int bpos = 0; bpos < 64; bpos++) {
            printf(" %.4x, ", hls_block[i_cmp * hls_cmpnfo[0].bc * 64 + k_dpos[i_cmp] * 64 + bpos]);
            if (bpos % 16 == 15) {
                printf("\n ");
            }
        }
    }
#if 0
  int i_cmp = 0;
  for(k_dpos[i_cmp]=0; k_dpos[i_cmp] <= hls_cmpnfo[i_cmp].bc - 1; k_dpos[i_cmp]++){
	  printf(  " %.4x, ",hls_block[i_cmp*hls_cmpnfo[ 0 ].bc*64 + k_dpos[i_cmp]*64 + 0] );
		if(k_dpos[i_cmp]%16==15){
			printf(  "\n ");
		}
  }
#endif

    free(datatoDDR);
    free(hls_block);

    std::cout << "Ready for next image!\n ";
}
#endif

// ************************************************************
