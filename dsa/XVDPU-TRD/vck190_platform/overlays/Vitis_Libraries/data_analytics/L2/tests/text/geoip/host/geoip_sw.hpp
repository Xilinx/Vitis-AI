#include <fstream>
#include "ap_int.h"
#include <vector>

#define TH1 16
#define TH2 4
#define Bank1 32
#define Bank2 24
#define NIP 1024 * 64

typedef ap_uint<32> uint32;
typedef ap_uint<64> uint64;
typedef ap_uint<512> uint512;
const uint64_t MAX_LEN = 4294967296 / 64;
const uint64_t N16 = 65536;

// read geoIP file from disk to in-memory buffer
// Here, the buffer format is better in arrow-style, including 32-bit offset and 8-bit data
int readGeoIP(std::string geoipFile, std::vector<std::string>& geoip) {
    std::cout << "read file: " << geoipFile << "\n";
    std::ifstream inFile(geoipFile, std::ios::in);
    // add check to make sure the file is opened correctly.
    if (!inFile.is_open()) {
        fprintf(stderr, "ERROR: %s file open failed!\n", geoipFile.c_str());
        return -1;
    } else {
        // read the file line-by-line
        std::string lineStr;
        getline(inFile, lineStr);
        while (getline(inFile, lineStr)) {
            geoip.push_back(lineStr);
        }
        std::cout << "geoip database size is " << geoip.size() << std::endl;
        inFile.close();
        return 0;
    }
}

// store data in specified memory format
int geoipConvert(
    std::vector<std::string> geoip, uint64_t* netHigh, uint512* netLow, uint32_t* ipBegin, uint32_t* ipEnd) {
    // get IP sub-string
    std::vector<std::string> networkArray;
    for (unsigned int i = 0; i < geoip.size(); i++) {
        // std::cout << lineStr << std::endl;
        std::string lineStr = geoip[i];

        std::string str = lineStr.substr(0, lineStr.find(','));
        networkArray.push_back(str);
    }

    // the value of last a1a2
    unsigned int netsHigh16Cnt = -1;
    // store the row-number for each
    unsigned int* netsHigh16 = aligned_alloc<unsigned int>(N16);
    // low-16 bit IP and 5-bit mask
    unsigned int* netsLow21 = aligned_alloc<unsigned int>(MAX_LEN);

    // store the flag for 0.0 to 255.255
    unsigned char* netsBFlag = aligned_alloc<unsigned char>(N16);
    // initialize the netsBFlag buffer
    for (uint64_t i = 0; i < N16; i++) {
        netsBFlag[i] = 0;
    }
    // parse the IP and convert to int
    for (unsigned int i = 0; i < networkArray.size(); i++) {
        std::string network = networkArray[i];
        std::string str;
        std::istringstream sin(network);
        int j = 0;

        // put the variable close to the place it is used.
        unsigned int b = 0;
        unsigned int a1a2 = 0;
        unsigned int a3a4 = 0;

        // extract the a1a2, a3a4, and mask
        while (getline(sin, str, '/')) {
            if (j == 0) {
                std::istringstream sip(str);
                std::string strip;

                // get the value of a1a2 and a3a4.
                int k = 0;
                while (getline(sip, strip, '.')) {
                    if (k < 2) {
                        if (k == 0)
                            a1a2 = std::stoi(strip) * 256;
                        else
                            a1a2 = std::stoi(strip) + a1a2;
                    } else {
                        if (k == 2)
                            a3a4 = std::stoi(strip) * 256;
                        else
                            a3a4 = std::stoi(strip) + a3a4;
                    }
                    k++;
                    netsLow21[i] = a3a4;
                }

            } else {
                // get the mask value
                b = std::stoi(str);
                if (b <= 16) {
                    for (int ib = 0; ib < (1 << (16 - b)); ib++) {
                        netsBFlag[a1a2 + ib] = 1;
                    }
                }
                // store the mask value
                // the high 5-bit is mask value
                // the low 16-bit is a3a4
                netsLow21[i] = netsLow21[i] + (b - 1) * 0x10000;
            }
            // for reach row, store the range of IP
            ipBegin[i] = (a1a2 << 16) + a3a4;
            ipEnd[i] = ipBegin[i] + (1 << (32 - b));
            j++;
        }

        for (unsigned int d = netsHigh16Cnt; d < a1a2; d++) {
            // store the row number
            // the high 64-bit store the row number
            // the low 1-bit store the flag
            netsHigh16[d + 1] = i;
            if ((netsBFlag[d + 1] == 1) && (d + 1 != a1a2))
                netHigh[d + 1] = ((i - 1) << 1) + netsBFlag[d + 1];
            else
                netHigh[d + 1] = (i << 1) + netsBFlag[d + 1];
        }
        // the last a1a2
        netsHigh16Cnt = a1a2;
    }

    // set the left value to be same with last one.
    for (unsigned int i = netsHigh16Cnt + 1; i < N16; i++) {
        netsHigh16[i] = networkArray.size();
        netHigh[i] = networkArray.size() * 2;
    }

    // number of 512-bit data
    uint64_t cnt512 = 0;

    // record the last index of newLow buffer
    unsigned int indexLow = 0;

    // the high 32-bit is the address of a3a4
    // the low 32-bit is the row number
    netHigh[0] += (cnt512 << 32);
    for (unsigned int i = 1; i <= netsHigh16Cnt + 1; i++) {
        unsigned int offsetLow = 0;
        uint512 tmp = 0;
        // if 0 < the row number with same a1a2 < 24*16 (16 is burst length???)
        if (netsHigh16[i] - netsHigh16[i - 1] <= Bank2 * TH1 && netsHigh16[i] != netsHigh16[i - 1]) {
            // the data width is 512, it could store 24 21-bit data
            // the number of 512-bit data
            unsigned int cntNet = (netsHigh16[i] - netsHigh16[i - 1] + Bank2 - 1) / Bank2;

            unsigned int lastIndex = indexLow;

            // the number of 512-bit data
            cnt512 += cntNet;
            // the high 504-bits stores 24 21-bit data and the low 8-bits store the number of data
            for (unsigned int j = netsHigh16[i - 1]; j < netsHigh16[i]; j++) {
                if (offsetLow == Bank2) {
                    tmp.range(7, 0) = Bank2;
                    netLow[indexLow++] = tmp;
                    tmp = 0;
                    offsetLow = 0;
                }
                tmp.range(offsetLow * 21 + 20 + 8, offsetLow * 21 + 8) = netsLow21[j];
                offsetLow++;
            }
            tmp.range(7, 0) = offsetLow;
            netLow[indexLow++] = tmp;
            tmp = 0;
            // start
            assert(indexLow - lastIndex == cntNet);
            // if the row number with same a1a2 > 24*16
            // then it needs multiple burst reads (burst length is 4?)
        } else if (netsHigh16[i] != netsHigh16[i - 1]) {
            // burst read number
            unsigned int cntNet = (netsHigh16[i] - netsHigh16[i - 1] + Bank2 * TH2 - 1) / (Bank2 * TH2);
            // the index number
            // 512-bit store 32 16-bit index
            // the index store the start a3a4 value for each read block
            unsigned int idxCnt = (cntNet + Bank1 - 2) / Bank1;
            cnt512 += idxCnt;

            unsigned int lastIndex = indexLow;
            for (unsigned int j = 1; j < cntNet; j++) {
                if (offsetLow == Bank1) {
                    netLow[indexLow++] = tmp;
                    tmp = 0;
                    offsetLow = 0;
                }
                tmp.range(offsetLow * 16 + 15, offsetLow * 16) = netsLow21[netsHigh16[i - 1] + Bank2 * TH2 * j];
                offsetLow++;
            }
            netLow[indexLow++] = tmp;
            tmp = 0;
            offsetLow = 0;

            assert(indexLow - lastIndex == idxCnt);
            if (indexLow - lastIndex != idxCnt)
                std::cout << "netsHigh16[" << i - 1 << "]=" << netsHigh16[i - 1] << ",netsHigh16[" << i
                          << "]=" << netsHigh16[i] << ",idxCnt=" << idxCnt << ",lastIndex=" << lastIndex
                          << ",indexLow=" << indexLow << ",cntNet=" << cntNet << ",calcu error\n";

            lastIndex = indexLow;
            // store the a3a4 and mask value
            // calculate the number of 512-bit data
            cntNet = (netsHigh16[i] - netsHigh16[i - 1] + Bank2 - 1) / Bank2;
            cnt512 += cntNet;
            for (unsigned int j = netsHigh16[i - 1]; j < netsHigh16[i]; j++) {
                if (offsetLow == Bank2) {
                    tmp.range(7, 0) = Bank2;
                    netLow[indexLow++] = tmp;
                    tmp = 0;
                    offsetLow = 0;
                }
                tmp.range(offsetLow * 21 + 20 + 8, offsetLow * 21 + 8) = netsLow21[j];
                offsetLow++;
            }
            tmp.range(7, 0) = offsetLow;
            netLow[indexLow++] = tmp;
            tmp = 0;
            assert(indexLow - lastIndex == cntNet);
        }
        // the address of netLowbuff for a1a2.
        netHigh[i] += (cnt512 << 32);
    }
    std::cout << "netsLow21 actual use buffer size is " << indexLow << std::endl;

    // the left one use the same value with last one.
    for (unsigned int i = netsHigh16Cnt + 1; i < N16; i++) {
        netHigh[i] += (cnt512 << 32);
    }
    delete[] netsBFlag;
    delete[] netsHigh16;
    delete[] netsLow21;
    return 0;
}

// check the result is correct
int geoip_check(uint32_t ipNum,
                uint32_t* ip,
                uint32_t* id,
                uint32_t* ipBegin,
                uint32_t* ipEnd,
                std::vector<std::string> geoip,
                std::vector<std::string>& geoip_out) {
    int nerr = 0;
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    for (unsigned int i = 0; i < ipNum; i++) {
        if (id[i] < MAX_LEN) {
            unsigned begin = ipBegin[id[i]];
            unsigned end = ipEnd[id[i]];
            if ((ip[i] < begin) || (ip[i] > end)) {
                std::cout << "ip[" << i << "]=" << ip[i] << ", id=" << id[i] << ", begin=" << begin << ", end=" << end
                          << std::endl;
                nerr++;
            }
            geoip_out.push_back(geoip[id[i]]);
            // std::cout << "geoip[" << id[i] << "]=" << geoip[id[i]] << ", ip[" << i << "]=" << ip[i] << std::endl;
        } else {
            if (id[i] != 0xFFFFFFFF) {
                std::cout << "ip[" << i << "]=" << ip[i] << ", id[i]=" << id[i] << std::endl;
                nerr++;
            }
            geoip_out.push_back("");
        }
    }
    gettimeofday(&t2, 0);
    std::cout << "copy geoip data " << tvdiff(&t1, &t2) / 1000.0 << "ms" << std::endl;
    if (!nerr) std::cout << "[INFO] geoip correct!\n";
    return nerr;
}
