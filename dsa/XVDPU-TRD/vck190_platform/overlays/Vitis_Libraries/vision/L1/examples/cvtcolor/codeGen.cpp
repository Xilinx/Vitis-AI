#include <algorithm>
#include <locale>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <sstream>
#include <assert.h>

// Image Dimensions

const std::string gBenchHeader = R"INLINE_CODE(
#include "common/xf_headers.hpp"
#define ERROR_THRESHOLD 2

)INLINE_CODE";

const std::string gBenchMain = R"INLINE_CODE(
int main(int argc, char** argv) {

    cv::Mat imgInput0, imgInput1, imgInput2;
    cv::Mat refOutput0, refOutput1, refOutput2;
    cv::Mat errImg0, errImg1, errImg2;

)INLINE_CODE";


const std::string gBenchFooter = R"INLINE_CODE(

    float err_per;
xf::cv::analyzeDiff(errImg0, ERROR_THRESHOLD, err_per);

if (err_per > 3.0f) {
    fprintf(stderr, "\n1st Image Test Failed\n");
}
#if (IYUV2NV12 || RGBA2NV12 || RGBA2NV21 || UYVY2NV12 || YUYV2NV12 || NV122IYUV || NV212IYUV || IYUV2YUV4 || \
     NV122YUV4 || NV212YUV4 || RGBA2IYUV || RGBA2YUV4 || UYVY2IYUV || YUYV2IYUV || NV122NV21 || NV212NV12)
xf::cv::analyzeDiff(errImg1, ERROR_THRESHOLD, err_per);
if (err_per > 3.0f) {
    fprintf(stderr, "\n2nd Image Test Failed\n");
    return 1;
}

#endif
#if (IYUV2YUV4 || NV122IYUV || NV122YUV4 || NV212IYUV || NV212YUV4 || RGBA2IYUV || RGBA2YUV4 || UYVY2IYUV || YUYV2IYUV)
xf::cv::analyzeDiff(errImg2, ERROR_THRESHOLD, err_per);
if (err_per > 3.0f) {
    fprintf(stderr, "\n3rd Image Test Failed\n");
    return 1;
}
#endif
/* ## *************************************************************** ##*/
return 0;
}
)INLINE_CODE";

const std::string gCopyRight = R"INLINE_CODE(/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.*You may obtain a copy of the License at *
        *http : // www.apache.org/licenses/LICENSE-2.0
                **Unless required by applicable law or
    agreed to in writing,
    software *distributed under the License is distributed on an "AS IS" BASIS,
    *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
    either express or
        implied.*See the License for the specific language governing permissions and*limitations under the License.*/

    ) INLINE_CODE ";

    const std::string gHeaderConfig = R "INLINE_CODE(
#include "hls_stream.h"
#include "ap_int.h"
#include "xf_config_params.h"
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_cvt_color.hpp"
#include "imgproc/xf_cvt_color_1.hpp"
//#include "imgproc/xf_rgb2hsv.hpp"
//#include "imgproc/xf_bgr2hsv.hpp"
// Has to be set when synthesizing
#define _XF_SYNTHESIS_ 1

    // Image Dimensions
    static constexpr int WIDTH = 1920;
static constexpr int HEIGHT = 1080;

#if (IYUV2NV12 || NV122IYUV || NV212IYUV || NV122YUV4 || NV212YUV4 || UYVY2NV12 || UYVY2NV21 || YUYV2NV12 ||           \
     YUYV2NV21 || RGBA2NV12 || RGBA2NV21 || RGB2NV12 || RGB2NV21 || NV122RGBA || NV212RGB || NV212RGBA || NV122RGB ||  \
     NV122BGR || NV212BGR || NV122YUYV || NV212YUYV || NV122UYVY || NV212UYVY || NV122NV21 || NV212NV12 || BGR2NV12 || \
     BGR2NV21)
#if NO
static constexpr int NPC1 = XF_NPPC1;
static constexpr int NPC2 = XF_NPPC1;
#endif
#if RO
static constexpr int NPC1 = XF_NPPC8;
static constexpr int NPC2 = XF_NPPC4;
#endif

#else
#if NO
static constexpr int NPC1 = XF_NPPC1;
static constexpr int NPC2 = XF_NPPC1;
#else
static constexpr int NPC1 = XF_NPPC8;
static constexpr int NPC2 = XF_NPPC8;
#endif
#endif

)INLINE_CODE";

struct PortType {
    std::string type;
    std::string cv_type;
    std::string height;
    std::string width;
    std::string npc;
    int ptr_width;
    int im_read_mode;

    int channels() {
        char C = type[type.size() - 1];
        std::stringstream ss;
        ss << C;
        int Cint;
        ss >> Cint;
        return Cint;
    }

    int bitwidth() {
        int i = 3;
        int width = 0;
        while ((type[i] >= '0') && (type[i] <= '9')) {
            width = 10 * width + (type[i] - '0');
            i++;
        }
        return width;
    }

    std::string ptrwidth() {
        std::stringstream ss;
        ss << ptr_width << "*" << npc;
        return ss.str();
    }
};

std::ostream& operator<<(std::ostream& out, const PortType& type) {
    out << type.type << ", " << type.height << ", " << type.width << ", " << type.npc;
    return out;
}

struct ImgType {
    std::string name;
    std::vector<PortType> portVec;
};

// 1 Port
/* type         cv type     height    width    npc    ptr_width cv::imread parameter*/
static ImgType BGR = {"BGR", {{"XF_8UC3", "CV_8UC3", "HEIGHT", "WIDTH", "NPC1", 32, 1}}};
static ImgType GRAY = {"GRAY", {{"XF_8UC1", "CV_8UC1", "HEIGHT", "WIDTH", "NPC1", 8, 0}}};
static ImgType HLS = {"HLS", {{"XF_8UC3", "CV_8UC3", "HEIGHT", "WIDTH", "NPC1", 32, 1}}};
static ImgType HSV = {"HSV", {{"XF_8UC3", "CV_8UC3", "HEIGHT", "WIDTH", "NPC1", 32, 1}}};
static ImgType RGB = {"RGB", {{"XF_8UC3", "CV_8UC3", "HEIGHT", "WIDTH", "NPC1", 32, 1}}};
static ImgType RGBA = {"RGBA", {{"XF_8UC4", "CV_8UC4", "HEIGHT", "WIDTH", "NPC1", 32, 1}}};
static ImgType UYVY = {"UYVY", {{"XF_16UC1", "CV_16UC1", "HEIGHT", "WIDTH", "NPC1", 16, -1}}};
static ImgType XYZ = {"XYZ", {{"XF_8UC3", "CV_8UC3", "HEIGHT", "WIDTH", "NPC1", 32, 1}}};
static ImgType YCRCB = {"YCrCb", {{"XF_8UC3", "CV_8UC3", "HEIGHT", "WIDTH", "NPC1", 32, 1}}};
static ImgType YUYV = {"YUYV", {{"XF_16UC1", "CV_16UC1", "HEIGHT", "WIDTH", "NPC1", 16, -1}}};

// 2 Ports
static ImgType NV12 = {"NV12",
                       {{"XF_8UC1", "CV_8UC1", "HEIGHT", "WIDTH", "NPC1", 8, 0},
                        {"XF_8UC2", "CV_16UC1", "(HEIGHT / 2)", "(WIDTH / 2)", "NPC2", 16, -1}}};

static ImgType NV21 = {"NV21",
                       {{"XF_8UC1", "CV_8UC1", "HEIGHT", "WIDTH", "NPC1", 8, 0},
                        {"XF_8UC2", "CV_16UC1", "(HEIGHT / 2)", "(WIDTH / 2)", "NPC2", 16, -1}}};

// 3 Ports
static ImgType IYUV = {"IYUV",
                       {{"XF_8UC1", "CV_8UC1", "HEIGHT", "WIDTH", "NPC1", 8, 0},
                        {"XF_8UC1", "CV_8UC1", "(HEIGHT / 4)", "WIDTH", "NPC1", 8, 0},
                        {"XF_8UC1", "CV_8UC1", "(HEIGHT / 4)", "WIDTH", "NPC1", 8, 0}}};

static ImgType YUV4 = {"YUV4",
                       {{"XF_8UC1", "CV_8UC1", "HEIGHT", "WIDTH", "NPC1", 8, 0},
                        {"XF_8UC1", "CV_8UC1", "HEIGHT", "WIDTH", "NPC1", 8, 0},
                        {"XF_8UC1", "CV_8UC1", "HEIGHT", "WIDTH", "NPC1", 8, 0}}};

std::string tolower(std::string data) {
    std::transform(data.begin(), data.end(), data.begin(), [](unsigned char c) { return std::tolower(c); });
    return data;
}

struct comp {
    bool operator()(const std::string& lhs, const std::string& rhs) const { return tolower(lhs) < tolower(rhs); }
};

static std::map<std::string, ImgType, comp> ports;
void initPorts() {
    ports[BGR.name] = BGR;
    ports[GRAY.name] = GRAY;
    ports[HLS.name] = HLS;
    ports[HSV.name] = HSV;
    ports[RGB.name] = RGB;
    ports[RGBA.name] = RGBA;
    ports[UYVY.name] = UYVY;
    ports[XYZ.name] = XYZ;
    ports[YCRCB.name] = YCRCB;
    ports[YUYV.name] = YUYV;

    ports[NV12.name] = NV12;
    ports[NV21.name] = NV21;

    ports[IYUV.name] = IYUV;
    ports[YUV4.name] = YUV4;
}

std::vector<std::string> merge(ImgType A, ImgType B) {
    // Currently only 4 parameters are passed to XF::CV call.
    // Of theese 4 parameters 1st (Image type) and 4rth (NPC) only need merging
    // 2nd (HIEGHT) and 3rd(WIDTH) are passed as is
    std::vector<std::string> ret;
    std::set<std::string> unqSetA;
    // Merge types
    for (int j = 0; j < A.portVec.size(); j++) {
        if (A.portVec.size() == 1)
            ret.push_back(A.portVec[j].type);
        else if (unqSetA.find(A.portVec[j].type) == unqSetA.end()) {
            ret.push_back(A.portVec[j].type);
            unqSetA.insert(A.portVec[j].type);
        }
    }

    for (int j = 0; j < B.portVec.size(); j++) {
        if (B.portVec.size() == 1)
            ret.push_back(B.portVec[j].type);
        else if (unqSetA.find(B.portVec[j].type) == unqSetA.end()) {
            ret.push_back(B.portVec[j].type);
            unqSetA.insert(B.portVec[j].type);
        }
    }

    // HEIGHT
    ret.push_back(A.portVec[0].height);

    // WIDTH
    ret.push_back(A.portVec[0].width);

    // Merge NPC
    std::set<std::string> unqSet;
    for (int j = 0; j < A.portVec.size(); j++) {
        if (unqSet.find(A.portVec[j].npc) == unqSet.end()) {
            ret.push_back(A.portVec[j].npc);
            unqSet.insert(A.portVec[j].npc);
        }
    }

    for (int j = 0; j < B.portVec.size(); j++) {
        if (unqSet.find(B.portVec[j].npc) == unqSet.end()) {
            ret.push_back(B.portVec[j].npc);
            unqSet.insert(B.portVec[j].npc);
        }
    }

    return ret;
}

long gnIndent = 0;
template <typename T>
void printIndent(T& ofs) {
    for (int i = 0; i < gnIndent; i++) ofs << " ";
}

void printVitisDepthExpr(ImgType In, bool bFrom, std::ofstream& ofs) {
    for (int i = 0; i < In.portVec.size(); i++) {
        gnIndent = 4;
        printIndent(ofs);
        gnIndent = 0;

        std::string sdepth = "((";
        sdepth += In.portVec[i].height;
        sdepth += ")*(";
        sdepth += In.portVec[i].width;
        sdepth += ")*(XF_PIXELWIDTH(";
        sdepth += In.portVec[i].type;
        sdepth += ",";
        sdepth += In.portVec[i].npc;
        sdepth += ")))/(";
        sdepth += In.portVec[i].ptrwidth();
        sdepth += ")";

        ofs << "static constexpr int __XF_DEPTH";
        if (bFrom)
            ofs << "_INP";
        else
            ofs << "_OUT";
        ofs << "_" << i;
        ofs << " = " << sdepth << ";" << std::endl;
    }
}

void printVitisPortPragma(ImgType In, bool bFrom, std::ofstream& ofs) {
    for (int i = 0; i < In.portVec.size(); i++) {
        gnIndent = 4;
        printIndent(ofs);
        gnIndent = 0;

        ofs << "#pragma HLS INTERFACE m_axi      port=";
        std::ostringstream name;
        if (bFrom)
            name << "imgInput";
        else
            name << "imgOutput";
        if (In.portVec.size() > 1) name << i;

        ofs << name.str();
        gnIndent = (12 - name.str().size());
        printIndent(ofs);
        gnIndent = 0;
        if (bFrom)
            ofs << "offset=slave  bundle=gmem_in" << i << "  depth=";
        else
            ofs << "offset=slave  bundle=gmem_out" << i << " depth=";

        ofs << "__XF_DEPTH";
        if (bFrom)
            ofs << "_INP";
        else
            ofs << "_OUT";
        ofs << "_" << i;
        ofs << std::endl;
    }

    if (!bFrom) {
        gnIndent = 4;
        printIndent(ofs);
        ofs << "#pragma HLS INTERFACE s_axilite  port=return" << std::endl;
        gnIndent = 0;
    }
}

/* This function generates port for accel function*/
void printPort(ImgType In, bool bFrom, std::ostringstream& ss) {
    for (int i = 0; i < In.portVec.size(); i++) {
        if ((!bFrom) || (i != 0)) printIndent(ss);

#ifdef VITIS
        ss << "ap_uint<";
        ss << In.portVec[i].ptrwidth();
        ss << ">* ";
#else
        ss << "xf::cv::Mat<";
        ss << In.portVec[i] << ">& ";
#endif
        if (bFrom)
            ss << "imgInput";
        else
            ss << "imgOutput";

        if (In.portVec.size() > 1) ss << i;
        ss << ",\n";
    }
}

/*This function when called with bDecl = true generates local xf::cv::Mat declarations for Vitis flow
 * When called with bDecl = false generates the formal arguements for xf::cv::<CVT function>
 */
void printPortUsage_VitisDecl(ImgType In, bool bFrom, bool bDecl, std::ofstream& ofs) {
    for (int i = 0; i < In.portVec.size(); i++) {
        if (bDecl) {
#ifdef VITIS
            gnIndent = 4;
            printIndent(ofs);
            gnIndent = 0;
            ofs << "xf::cv::Mat<";
            ofs << In.portVec[i] << "> ";
#else
            assert(bDecl == false); // for non vitis code gen, bdecl should never be true
#endif
        }

        std::ostringstream name;
#ifdef VITIS
        name << "_";
#endif
        if (bFrom)
            name << "imgInput";
        else
            name << "imgOutput";
        if (In.portVec.size() > 1) name << i;

        ofs << name.str();
        if (bDecl) {
#ifdef VITIS
            ofs << "(" << In.portVec[i].height << ", " << In.portVec[i].width << ");";
#else
            ofs << ",";
#endif
            ofs << std::endl;
        } else
            ofs << ", ";
    }
}

/*This function generates conversion functions from ap_unit<> to xf::cv::Mat*/
void printConversions(ImgType In, bool bFrom, std::ofstream& ofs) {
#ifndef VITIS
    assert(0); // Function meant for only VITIS
#endif

    for (int i = 0; i < In.portVec.size(); i++) {
        gnIndent = 4;
        printIndent(ofs);
        gnIndent = 0;
        if (bFrom)
            ofs << "xf::cv::Array2xfMat<";
        else
            ofs << "xf::cv::xfMat2Array<";

        ofs << In.portVec[i].ptrwidth() << ", ";
        ofs << In.portVec[i];
        ofs << ">";

        ofs << "(";
        // Arguements
        std::ostringstream name_src;
        std::ostringstream name_dst;

        if (bFrom) {
            name_src << "imgInput";
            name_dst << "_imgInput";
        } else {
            name_src << "_imgOutput";
            name_dst << "imgOutput";
        }

        if (In.portVec.size() > 1) {
            name_src << i;
            name_dst << i;
        }

        ofs << name_src.str() << ", " << name_dst.str();
        ofs << ");" << std::endl;
    }
}

void printErrorImageInBench(ImgType In, std::ofstream& ofs) {
    for (int i = 0; i < In.portVec.size(); i++) {
        std::stringstream ss;
        ss << "errImg" << i;

        std::string name = ss.str();
        gnIndent = 4;
        printIndent(ofs);
        ofs << "cv::Size S" << i << "(" << In.portVec[i].width << ", " << In.portVec[i].height << ");" << std::endl;

        printIndent(ofs);
        ofs << name << ".create(S" << i << ", " << In.portVec[i].cv_type << ");" << std::endl;
        gnIndent = 0;
    }
}

void printInputOrRefImageInBench(int& i, const std::string name, PortType port, std::ofstream& ofs) {
    gnIndent = 4;
    printIndent(ofs);

    ofs << name << " = cv::imread(argv[" << i << "], " << port.im_read_mode << ");" << std::endl;

    printIndent(ofs);
    ofs << "if (!" << name << ".data) {" << std::endl;

    gnIndent = 8;
    printIndent(ofs);
    ofs << "fprintf(stderr, " Can't open image %s\n", argv[i]);" << std::endl;

        printIndent(ofs);
    ofs << "return -1;" << std::endl;

    gnIndent = 4;
    printIndent(ofs);
    ofs << "}" << std::endl;
    gnIndent = 0;
}

bool gbDisableRGBToBGR = false;
/*for some conversions RGB is not converted to BGR in test bench.
 * BGR2RGB
 * GRAY2RGB
 * XYZ2RGB
 * YCrCb2RGB
 * HLS2RGB
 * HSV2RGB
 */
void printImageInBench(ImgType In, bool bRef, int& argCnt, std::ofstream& ofs) {
    for (int i = 0; i < In.portVec.size(); i++) {
        std::stringstream ss;
        if (bRef)
            ss << "refOutput" << i;
        else
            ss << "imgInput" << i;

        std::string name = ss.str();
        printInputOrRefImageInBench(argCnt, name, In.portVec[i], ofs);
        argCnt++;

        gnIndent = 4;
        if (bRef == false) {
            if (In.name == "RGBA") {
                printIndent(ofs);
                ofs << "cvtColor(" << name << ", " << name << ", CV_BGR2RGBA);" << std::endl;
            } else if ((gbDisableRGBToBGR == false) && (In.name == "RGB")) {
                printIndent(ofs);
                ofs << "cvtColor(" << name << ", " << name << ", CV_BGR2RGB);" << std::endl;
            }
        }

        printIndent(ofs);
        std::stringstream tss;
        tss << "ap_uint<" << In.portVec[i].ptrwidth() << ">";
        std::string type_str = tss.str();

        std::stringstream new_name_ss;
        if (bRef)
            new_name_ss << "_imgOutput" << i;
        else
            new_name_ss << "_imgInput" << i;
        std::string new_name = new_name_ss.str();

        ofs << type_str << "* " << new_name << " = "
            << "(" << type_str << " *)";
        if (bRef) {
            ofs << "malloc(" << In.portVec[i].height << "*" << In.portVec[i].width << "*" << In.portVec[i].ptr_width
                << ");" << std::endl;
        } else {
            ofs << name << ".data;" << std::endl;
        }
        gnIndent = 0;
        ofs << std::endl;
    }
}

void printOutputImageInBench(ImgType In, std::ofstream& ofs) {
    ofs << std::endl;
    for (int i = 0; i < In.portVec.size(); i++) {
        std::stringstream ss;
        ss << "imgOutput" << i;
        std::string name = ss.str();

        gnIndent = 4;
        printIndent(ofs);
        ofs << "cv::Mat " << name << "(" << In.portVec[i].height << ", " << In.portVec[i].width << ", "
            << In.portVec[i].cv_type << ");" << std::endl;
        gnIndent = 0;
    }
}

void printCall(ImgType From, ImgType To, std::ofstream& ofs) {
    ofs << std::endl;
    std::ostringstream funcCall;
    funcCall << "cvtcolor_" << tolower(From.name) << "2" << tolower(To.name);

    funcCall << "(";
    for (int i = 0; i < From.portVec.size(); i++) {
        std::stringstream ss;
        ss << "_imgInput" << i;

        funcCall << ss.str() << ", ";
    }

    for (int i = 0; i < To.portVec.size(); i++) {
        std::stringstream ss;
        ss << "_imgOutput" << i;

        funcCall << ss.str() << ", ";
    }

    long nPos = funcCall.tellp();
    funcCall.seekp(nPos - 2);
    funcCall.write(");", 2);

    gnIndent = 4;
    printIndent(ofs);
    ofs << funcCall.str();
    ofs << std::endl << std::endl;
    gnIndent = 0;
}

void printDiff(ImgType To, std::ofstream& ofs) {
    ofs << std::endl;
    for (int i = 0; i < To.portVec.size(); i++) {
        std::stringstream refss;
        std::stringstream outss;
        std::stringstream errss;

        refss << "refOutput" << i;
        outss << "imgOutput" << i;
        errss << "errImg" << i;

        gnIndent = 4;
        printIndent(ofs);
        ofs << "cv::absdiff(" << refss.str() << ", " << outss.str() << ", " << errss.str() << ");" << std::endl;
        gnIndent = 0;
    }
}

void printOutputs(ImgType To, std::ofstream& ofs) {
    for (int i = 0; i < To.portVec.size(); i++) {
        std::stringstream ss;
        ss << "imgOutput" << i;

        std::string name1 = ss.str();
        std::string name2 = "_" + name1;

        gnIndent = 4;
        printIndent(ofs);
        ofs << name1 << ".data = (unsigned char*)" << name2 << ";" << std::endl;
        if (To.name == "RGBA") {
            printIndent(ofs);
            ofs << "cvtColor(" << name1 << ", " << name1 << ", CV_RGBA2BGR);" << std::endl;
        } else if (To.name == "RGB") {
            printIndent(ofs);
            ofs << "cvtColor(" << name1 << ", " << name1 << ", CV_RGB2BGR);" << std::endl;
        }
    }
    ofs << std::endl;
}

void printImageWrite(ImgType To, bool bErr, std::ofstream& ofs) {
    ofs << std::endl;
    for (int i = 0; i < To.portVec.size(); i++) {
        std::stringstream ss;
        ss << "imgOutput" << i;
        std::string name1 = ss.str();

        gnIndent = 4;
        printIndent(ofs);
        ofs << "cv::imwrite(";
        std::stringstream fss;
        if (bErr)
            fss << "err_";
        else
            fss << "out_";
        if ((To.name == "NV12") || (To.name == "NV21")) {
            if (i == 0)
                fss << "Y";
            else
                fss << "UV";
        } else if ((To.name == "IYUV") || (To.name == "YUV4")) {
            if (i == 0)
                fss << "Y";
            else if (i == 1)
                fss << "U";
            else
                fss << "V";
        } else {
            fss << To.name;
        }
        fss << ".png";
        ofs << "\"" << fss.str() << "\", " << name1 << ");" << std::endl;
    }
}

#ifdef VITIS
int genBench(std::string& from, std::string& to, std::ofstream& ofs) {
    std::map<std::string, ImgType, comp>::iterator it_from = ports.find(from);
    std::map<std::string, ImgType, comp>::iterator it_to = ports.find(to);

    bool bErr = false;
    if (it_from == ports.end()) {
        fprintf(stderr, "Unrecognozed from target %s\n", from);
        bErr = true;
    }

    if (it_to == ports.end()) {
        fprintf(stderr, "Unrecognozed from target %s\n", from);
        bErr = true;
    }
    if (bErr) return -1;

    // Code generator
    //#ifdef
    ofs << "#if " << it_from->first << "2" << it_to->first << std::endl;
    int argCnt = 1;
    if (it_to->second.name == "RGB") {
        if ((it_from->second.name == "BGR") || (it_from->second.name == "GRAY") || (it_from->second.name == "XYZ") ||
            (it_from->second.name == "YCrCb") || (it_from->second.name == "HLS") || (it_from->second.name == "HSV"))
            gbDisableRGBToBGR = true;
        else
            gbDisableRGBToBGR = false;
    }

    printImageInBench(it_from->second, false, argCnt, ofs);
    printImageInBench(it_to->second, true, argCnt, ofs);
    printOutputImageInBench(it_to->second, ofs);
    printCall(it_from->second, it_to->second, ofs);
    printOutputs(it_to->second, ofs);
    printImageWrite(it_to->second, false, ofs);
    printErrorImageInBench(it_to->second, ofs);
    printDiff(it_to->second, ofs);
    printImageWrite(it_to->second, true, ofs);
    ofs << "#endif" << std::endl;

    return 0;
}
#endif

int genCode(std::string& from, std::string& to, std::ofstream& ofs, std::ofstream& ofs_h) {
    std::map<std::string, ImgType, comp>::iterator it_from = ports.find(from);
    std::map<std::string, ImgType, comp>::iterator it_to = ports.find(to);

    bool bErr = false;
    if (it_from == ports.end()) {
        fprintf(stderr, "Unrecognozed from target %s\n", from);
        bErr = true;
    }

    if (it_to == ports.end()) {
        fprintf(stderr, "Unrecognozed from target %s\n", from);
        bErr = true;
    }
    if (bErr) return -1;

    // Code generator
    //#ifdef
    ofs << "#if " << it_from->first << "2" << it_to->first << std::endl;

    // Function name
    std::ostringstream funcHeader;
    funcHeader << "void cvtcolor_";
    funcHeader << tolower(from);
    funcHeader << 2;
    funcHeader << tolower(to);
    funcHeader << "(";

    // Function ports
    gnIndent = funcHeader.str().size();
    printPort(it_from->second, true, funcHeader);
    printPort(it_to->second, false, funcHeader);
    gnIndent = 0;
    long nPos = funcHeader.tellp();
    funcHeader.seekp(nPos - 2);
    funcHeader.write(");", 2);
    ofs_h << funcHeader.str() << std::endl;

    funcHeader.seekp(nPos - 2);
    funcHeader.write(") ", 2);
    ofs << funcHeader.str();
    ofs << "{" << std::endl;

#ifdef VITIS
    ofs << std::endl;
    printVitisDepthExpr(it_from->second, true, ofs);
    printVitisDepthExpr(it_to->second, false, ofs);
    ofs << std::endl;

    gnIndent = 4;
    printIndent(ofs);
    ofs << "// clang-format off" << std::endl;
    gnIndent = 0;
    printVitisPortPragma(it_from->second, true, ofs);
    printVitisPortPragma(it_to->second, false, ofs);

    gnIndent = 4;
    printIndent(ofs);
    ofs << "// clang-format on" << std::endl;
    gnIndent = 0;
    ofs << std::endl;

    printPortUsage_VitisDecl(it_from->second, true, true, ofs);
    printPortUsage_VitisDecl(it_to->second, false, true, ofs);
    ofs << std::endl;
#endif

    // XF CV inner call #[
    gnIndent = 4;
    printIndent(ofs);
    ofs << "// clang-format off" << std::endl;
    printIndent(ofs);
    ofs << "#pragma HLS DATAFLOW" << std::endl;
    printIndent(ofs);
    ofs << "// clang-format on" << std::endl;
    ofs << std::endl;
    gnIndent = 0;

#ifdef VITIS
    printConversions(it_from->second, true, ofs);
#endif

    ofs << std::endl;
    gnIndent = 4;
    printIndent(ofs);
    gnIndent = 0;
    ofs << "xf::cv::" << tolower(from) << "2" << tolower(to) << "<";
    std::vector<std::string> mergeParam = merge(it_from->second, it_to->second);
    for (int i = 0; i < mergeParam.size(); i++) {
        ofs << mergeParam[i];
        if (i < (mergeParam.size() - 1)) ofs << ", ";
    }
    ofs << ">";

    ofs << "(";
    printPortUsage_VitisDecl(it_from->second, true, false, ofs);
    printPortUsage_VitisDecl(it_to->second, false, false, ofs);
    nPos = ofs.tellp();
    ofs.seekp(nPos - 2);
    ofs.write(");", 2);
    ofs << std::endl;
    ofs << std::endl;
//#]

#ifdef VITIS
    printConversions(it_to->second, false, ofs);
#endif
    ofs << "}" << std::endl;
    ofs << "#endif" << std::endl;
}

int main() {
    initPorts();

    std::string output_gen = "XF_CVT_COLOR_ACCEL_GEN";
    std::string output_conf = "XF_CVT_COLOR_CONFIG_GEN";

#ifdef VITIS
    std::string output_bench = "XF_CVT_COLOR_TB_GEN";
    output_gen += "_VITIS";
    output_conf += "_VITIS";
    output_bench += "_VITIS";
#endif
    std::ifstream ifs("convConfigs", std::ifstream::in);
    if (!ifs.is_open()) {
        fprintf(stderr, "Unable to open file convConfigs. Exiting ...\n ");
        return -1;
    }

    std::string output_cpp = tolower(output_gen) + ".cpp";
    std::string output_h = tolower(output_conf) + ".h";
    std::ofstream ofs(output_cpp.c_str(), std::ofstream::out);
    std::ofstream ofs_h(output_h.c_str(), std::ofstream::out);

#ifdef VITIS
    std::string output_bench_cpp = tolower(output_bench) + ".cpp";
    std::ofstream ofs_bench(output_bench_cpp.c_str(), std::ofstream::out);
#endif

    // Global comments and headers
    ofs << gCopyRight;
    ofs << "#include "
        << "\"" << output_h << "\"" << std::endl
        << std::endl;

    ofs_h << gCopyRight;
    ofs_h << "#ifndef "
          << "_" << output_conf << "_H_" << std::endl;
    ofs_h << "#define "
          << "_" << output_conf << "_H_" << std::endl;
    ofs_h << gHeaderConfig;

#ifdef VITIS
    ofs_bench << gCopyRight;
    ofs_bench << gBenchHeader;
    ofs_bench << "#include "
              << "\"" << output_h << "\"" << std::endl
              << std::endl;
    ofs_bench << gBenchMain;
#endif

    int stat = 0;
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string from;
        std::string to;
        ss >> from >> to;

        stat |= genCode(from, to, ofs, ofs_h);
#ifdef VITIS
        stat |= genBench(from, to, ofs_bench);
#endif
    }
#ifdef VITIS
    ofs_bench << gBenchFooter;
    ofs_bench.close();
#endif
    ofs_h << "#endif" << std::endl;
    ofs.close();
    ofs_h.close();
    ifs.close();

    return stat;
}
