#include <iostream>
#include "adf/adf_api/AIEControlConfig.h"
#include "kernels.h"


/************************** Graph Configurations  *****************************/

  adf::GraphConfig GraphConfigurations[] = {
  //{id, name, graphLoadElfFunc, graphInitFunc, graphDebugHalt, coreColumns, coreRows, iterMemColumns, iterMemRows, iterMemAddrs, triggered, plKernelInstanceNames, plAxiLiteModes, plDriverStartFuncs, plDriverCheckIPDoneFuncs}
    {0, "convgraph", nullptr, nullptr, nullptr, {6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17}, {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7}, {6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17}, {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7}, {0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804, 0x3804}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {}, {}, {}, {},  }, 
    {1, "resize_norm", nullptr, nullptr, nullptr, {36}, {0}, {35}, {0}, {0x7544}, {0}, {}, {}, {}, {},  }, 
  };
  const int NUM_GRAPH = 2;


/************************** RTP Configurations  *****************************/

  adf::RTPConfig RTPConfigurations[] = {
  //{portId, aliasId, portName, aliasName, graphId, isInput, isAsync, isConnect, elemType, numBytes, isPL, hasLock, selectorColumn, selectorRow, selectorAddr, selectorLockId, pingColumn, pingRow, pingAddr, pongColumn, pongRow, pongAddr, pongLockId, plKernelInstanceName, plParameterIndex, plDriverWriteRTP, plDriverReadRTP}
    {810, 800, "resize_norm.k.in[1]", "resize_norm.a0", 1, true, true, false, (adf::RTPConfig::elementType)3, 4, false, true, 36, 0, 0x40, 0, 36, 0, 0x1000, 1, 36, 0, 0x2000, 2, "", -1, nullptr, nullptr},
    {811, 801, "resize_norm.k.in[2]", "resize_norm.a1", 1, true, true, false, (adf::RTPConfig::elementType)3, 4, false, true, 35, 0, 0x23c0, 0, 36, 0, 0x6000, 3, 36, 0, 0xc0, 4, "", -1, nullptr, nullptr},
    {812, 802, "resize_norm.k.in[3]", "resize_norm.a2", 1, true, true, false, (adf::RTPConfig::elementType)3, 4, false, true, 36, 0, 0, 5, 36, 0, 0x3fe0, 6, 36, 0, 0xa0, 7, "", -1, nullptr, nullptr},
    {813, 803, "resize_norm.k.in[4]", "resize_norm.a3", 1, true, true, false, (adf::RTPConfig::elementType)3, 4, false, true, 36, 0, 0x60, 8, 36, 0, 0x2020, 9, 36, 0, 0x20, 10, "", -1, nullptr, nullptr},
    {814, 804, "resize_norm.k.in[5]", "resize_norm.b0", 1, true, true, false, (adf::RTPConfig::elementType)3, 4, false, true, 35, 0, 0x23e0, 1, 36, 0, 0x2040, 11, 36, 0, 0x80, 12, "", -1, nullptr, nullptr},
    {815, 805, "resize_norm.k.in[6]", "resize_norm.b1", 1, true, true, false, (adf::RTPConfig::elementType)3, 4, false, true, 36, 0, 0x140, 13, 35, 0, 0x2400, 2, 35, 0, 0x4000, 3, "", -1, nullptr, nullptr},
    {816, 806, "resize_norm.k.in[7]", "resize_norm.b2", 1, true, true, false, (adf::RTPConfig::elementType)3, 4, false, true, 35, 0, 0x6000, 4, 35, 0, 0x3fe0, 5, 35, 0, 0x5fe0, 6, "", -1, nullptr, nullptr},
    {817, 807, "resize_norm.k.in[8]", "resize_norm.b3", 1, true, true, false, (adf::RTPConfig::elementType)3, 4, false, true, 35, 0, 0x6020, 7, 36, 0, 0x4020, 14, 36, 0, 0x120, 15, "", -1, nullptr, nullptr},
  };
  const int NUM_RTP = 8;

/************************** PLIO Configurations  *****************************/

  adf::PLIOConfig PLIOConfigurations[] = {
  //{id, name, loginal_name, shim_column, slaveOrMaster, streamId}
    {0, "convgraph.datain_a[0]", "XvDPU_v1/M_AXIS_PL2MEb0_r0_c0", 8, 0, 0},
    {1, "convgraph.datain_a[1]", "XvDPU_v1/M_AXIS_PL2MEb0_r0_c1", 9, 0, 0},
    {2, "convgraph.datain_a[2]", "XvDPU_v1/M_AXIS_PL2MEb0_r2_c0", 8, 0, 4},
    {3, "convgraph.datain_a[3]", "XvDPU_v1/M_AXIS_PL2MEb0_r2_c1", 9, 0, 4},
    {4, "convgraph.datain_a[4]", "XvDPU_v1/M_AXIS_PL2MEb0_r4_c0", 6, 0, 4},
    {5, "convgraph.datain_a[5]", "XvDPU_v1/M_AXIS_PL2MEb0_r4_c1", 7, 0, 4},
    {6, "convgraph.datain_a[6]", "XvDPU_v1/M_AXIS_PL2MEb0_r6_c0", 6, 0, 0},
    {7, "convgraph.datain_a[7]", "XvDPU_v1/M_AXIS_PL2MEb0_r6_c1", 7, 0, 0},
    {8, "convgraph.datain_a[8]", "XvDPU_v1/M_AXIS_PL2MEb1_r0_c0", 12, 0, 0},
    {9, "convgraph.datain_a[9]", "XvDPU_v1/M_AXIS_PL2MEb1_r0_c1", 13, 0, 4},
    {10, "convgraph.datain_a[10]", "XvDPU_v1/M_AXIS_PL2MEb1_r2_c0", 12, 0, 4},
    {11, "convgraph.datain_a[11]", "XvDPU_v1/M_AXIS_PL2MEb1_r2_c1", 13, 0, 0},
    {12, "convgraph.datain_a[12]", "XvDPU_v1/M_AXIS_PL2MEb1_r4_c0", 10, 0, 0},
    {13, "convgraph.datain_a[13]", "XvDPU_v1/M_AXIS_PL2MEb1_r4_c1", 11, 0, 0},
    {14, "convgraph.datain_a[14]", "XvDPU_v1/M_AXIS_PL2MEb1_r6_c0", 10, 0, 4},
    {15, "convgraph.datain_a[15]", "XvDPU_v1/M_AXIS_PL2MEb1_r6_c1", 11, 0, 4},
    {16, "convgraph.datain_a[16]", "XvDPU_v1/M_AXIS_PL2MEb2_r0_c0", 16, 0, 4},
    {17, "convgraph.datain_a[17]", "XvDPU_v1/M_AXIS_PL2MEb2_r0_c1", 17, 0, 0},
    {18, "convgraph.datain_a[18]", "XvDPU_v1/M_AXIS_PL2MEb2_r2_c0", 16, 0, 0},
    {19, "convgraph.datain_a[19]", "XvDPU_v1/M_AXIS_PL2MEb2_r2_c1", 17, 0, 4},
    {20, "convgraph.datain_a[20]", "XvDPU_v1/M_AXIS_PL2MEb2_r4_c0", 14, 0, 4},
    {21, "convgraph.datain_a[21]", "XvDPU_v1/M_AXIS_PL2MEb2_r4_c1", 15, 0, 4},
    {22, "convgraph.datain_a[22]", "XvDPU_v1/M_AXIS_PL2MEb2_r6_c0", 14, 0, 0},
    {23, "convgraph.datain_a[23]", "XvDPU_v1/M_AXIS_PL2MEb2_r6_c1", 15, 0, 0},
    {24, "convgraph.datain_b[0]", "XvDPU_v1/M_AXIS_WM2MEb0_g0_l0", 20, 0, 0},
    {25, "convgraph.datain_b[1]", "XvDPU_v1/M_AXIS_WM2MEb0_g0_l1", 41, 0, 0},
    {26, "convgraph.datain_b[2]", "XvDPU_v1/M_AXIS_WM2MEb0_g1_l0", 23, 0, 0},
    {27, "convgraph.datain_b[3]", "XvDPU_v1/M_AXIS_WM2MEb0_g1_l1", 19, 0, 4},
    {28, "convgraph.datain_b[4]", "XvDPU_v1/M_AXIS_WM2MEb0_g2_l0", 18, 0, 0},
    {29, "convgraph.datain_b[5]", "XvDPU_v1/M_AXIS_WM2MEb0_g2_l1", 35, 0, 0},
    {30, "convgraph.datain_b[6]", "XvDPU_v1/M_AXIS_WM2MEb0_g3_l0", 19, 0, 0},
    {31, "convgraph.datain_b[7]", "XvDPU_v1/M_AXIS_WM2MEb0_g3_l1", 27, 0, 0},
    {32, "convgraph.dataout[0]", "XvDPU_v1/M_AXIS_ME2PLb0_n0_r0_c1", 7, 1, 0},
    {33, "convgraph.dataout[1]", "XvDPU_v1/M_AXIS_ME2PLb0_n0_r1_c0", 6, 1, 0},
    {34, "convgraph.dataout[8]", "XvDPU_v1/M_AXIS_ME2PLb0_n1_r0_c1", 9, 1, 0},
    {35, "convgraph.dataout[9]", "XvDPU_v1/M_AXIS_ME2PLb0_n1_r1_c0", 8, 1, 0},
    {36, "convgraph.dataout[2]", "XvDPU_v1/M_AXIS_ME2PLb0_n0_r2_c1", 7, 1, 5},
    {37, "convgraph.dataout[3]", "XvDPU_v1/M_AXIS_ME2PLb0_n0_r3_c0", 6, 1, 5},
    {38, "convgraph.dataout[10]", "XvDPU_v1/M_AXIS_ME2PLb0_n1_r2_c1", 9, 1, 5},
    {39, "convgraph.dataout[11]", "XvDPU_v1/M_AXIS_ME2PLb0_n1_r3_c0", 8, 1, 5},
    {40, "convgraph.dataout[4]", "XvDPU_v1/M_AXIS_ME2PLb0_n0_r4_c1", 7, 1, 1},
    {41, "convgraph.dataout[5]", "XvDPU_v1/M_AXIS_ME2PLb0_n0_r5_c0", 6, 1, 1},
    {42, "convgraph.dataout[12]", "XvDPU_v1/M_AXIS_ME2PLb0_n1_r4_c1", 9, 1, 1},
    {43, "convgraph.dataout[13]", "XvDPU_v1/M_AXIS_ME2PLb0_n1_r5_c0", 8, 1, 1},
    {44, "convgraph.dataout[6]", "XvDPU_v1/M_AXIS_ME2PLb0_n0_r6_c1", 7, 1, 3},
    {45, "convgraph.dataout[7]", "XvDPU_v1/M_AXIS_ME2PLb0_n0_r7_c0", 6, 1, 3},
    {46, "convgraph.dataout[14]", "XvDPU_v1/M_AXIS_ME2PLb0_n1_r6_c1", 9, 1, 3},
    {47, "convgraph.dataout[15]", "XvDPU_v1/M_AXIS_ME2PLb0_n1_r7_c0", 8, 1, 3},
    {48, "convgraph.dataout[16]", "XvDPU_v1/M_AXIS_ME2PLb1_n0_r0_c1", 11, 1, 0},
    {49, "convgraph.dataout[17]", "XvDPU_v1/M_AXIS_ME2PLb1_n0_r1_c0", 10, 1, 0},
    {50, "convgraph.dataout[24]", "XvDPU_v1/M_AXIS_ME2PLb1_n1_r0_c1", 13, 1, 0},
    {51, "convgraph.dataout[25]", "XvDPU_v1/M_AXIS_ME2PLb1_n1_r1_c0", 12, 1, 0},
    {52, "convgraph.dataout[18]", "XvDPU_v1/M_AXIS_ME2PLb1_n0_r2_c1", 11, 1, 5},
    {53, "convgraph.dataout[19]", "XvDPU_v1/M_AXIS_ME2PLb1_n0_r3_c0", 10, 1, 5},
    {54, "convgraph.dataout[26]", "XvDPU_v1/M_AXIS_ME2PLb1_n1_r2_c1", 13, 1, 5},
    {55, "convgraph.dataout[27]", "XvDPU_v1/M_AXIS_ME2PLb1_n1_r3_c0", 12, 1, 5},
    {56, "convgraph.dataout[20]", "XvDPU_v1/M_AXIS_ME2PLb1_n0_r4_c1", 11, 1, 1},
    {57, "convgraph.dataout[21]", "XvDPU_v1/M_AXIS_ME2PLb1_n0_r5_c0", 10, 1, 1},
    {58, "convgraph.dataout[28]", "XvDPU_v1/M_AXIS_ME2PLb1_n1_r4_c1", 13, 1, 1},
    {59, "convgraph.dataout[29]", "XvDPU_v1/M_AXIS_ME2PLb1_n1_r5_c0", 12, 1, 1},
    {60, "convgraph.dataout[22]", "XvDPU_v1/M_AXIS_ME2PLb1_n0_r6_c1", 11, 1, 3},
    {61, "convgraph.dataout[23]", "XvDPU_v1/M_AXIS_ME2PLb1_n0_r7_c0", 10, 1, 3},
    {62, "convgraph.dataout[30]", "XvDPU_v1/M_AXIS_ME2PLb1_n1_r6_c1", 13, 1, 3},
    {63, "convgraph.dataout[31]", "XvDPU_v1/M_AXIS_ME2PLb1_n1_r7_c0", 12, 1, 3},
    {64, "convgraph.dataout[32]", "XvDPU_v1/M_AXIS_ME2PLb2_n0_r0_c1", 15, 1, 0},
    {65, "convgraph.dataout[33]", "XvDPU_v1/M_AXIS_ME2PLb2_n0_r1_c0", 14, 1, 0},
    {66, "convgraph.dataout[40]", "XvDPU_v1/M_AXIS_ME2PLb2_n1_r0_c1", 17, 1, 0},
    {67, "convgraph.dataout[41]", "XvDPU_v1/M_AXIS_ME2PLb2_n1_r1_c0", 16, 1, 0},
    {68, "convgraph.dataout[34]", "XvDPU_v1/M_AXIS_ME2PLb2_n0_r2_c1", 15, 1, 5},
    {69, "convgraph.dataout[35]", "XvDPU_v1/M_AXIS_ME2PLb2_n0_r3_c0", 14, 1, 5},
    {70, "convgraph.dataout[42]", "XvDPU_v1/M_AXIS_ME2PLb2_n1_r2_c1", 17, 1, 5},
    {71, "convgraph.dataout[43]", "XvDPU_v1/M_AXIS_ME2PLb2_n1_r3_c0", 16, 1, 5},
    {72, "convgraph.dataout[36]", "XvDPU_v1/M_AXIS_ME2PLb2_n0_r4_c1", 15, 1, 1},
    {73, "convgraph.dataout[37]", "XvDPU_v1/M_AXIS_ME2PLb2_n0_r5_c0", 14, 1, 1},
    {74, "convgraph.dataout[44]", "XvDPU_v1/M_AXIS_ME2PLb2_n1_r4_c1", 17, 1, 1},
    {75, "convgraph.dataout[45]", "XvDPU_v1/M_AXIS_ME2PLb2_n1_r5_c0", 16, 1, 1},
    {76, "convgraph.dataout[38]", "XvDPU_v1/M_AXIS_ME2PLb2_n0_r6_c1", 15, 1, 3},
    {77, "convgraph.dataout[39]", "XvDPU_v1/M_AXIS_ME2PLb2_n0_r7_c0", 14, 1, 3},
    {78, "convgraph.dataout[46]", "XvDPU_v1/M_AXIS_ME2PLb2_n1_r6_c1", 17, 1, 3},
    {79, "convgraph.dataout[47]", "XvDPU_v1/M_AXIS_ME2PLb2_n1_r7_c0", 16, 1, 3},
    {80, "resize_norm.in1", "DataIn0", 36, 0, 0},
    {81, "resize_norm.out1", "DataOut0", 36, 1, 0},
  };
  const int NUM_PLIO = 82;


/************************** ADF API initializer *****************************/

  class InitializeAIEControlXRT
  {
  public:
    InitializeAIEControlXRT()
    {
      std::cout<<"Initializing ADF API..."<<std::endl;
#ifdef __EXCLUDE_PL_CONTROL__
      bool exclude_pl_control = true;
#else
      bool exclude_pl_control = false;
#endif
      adf::initializeConfigurations(nullptr, 0, 0, 0,
                                    GraphConfigurations, NUM_GRAPH,
                                    RTPConfigurations, NUM_RTP,
                                    nullptr, 0,
                                    nullptr, 0,
                                    nullptr, 0,
                                    nullptr, 0,
                                    nullptr, 0,
                                    PLIOConfigurations, NUM_PLIO,
                                    nullptr, 0, 0, nullptr,
                                    false, exclude_pl_control, false, nullptr,
                                    true, 2);

    }
  } initAIEControlXRT;



#if !defined(__CDO__)

// Kernel Stub Definition
template<> void ResizeNormRunner<1920, 2, 224, 1, 224>::run(input_window<unsigned char> *,output_window<signed char> *,int,int,int,int,int,int,int,int) { /* Stub */ } 
  void super_kernel_casc_output(input_window<int> *,input_window<int> *,input_stream<acc48_t> *,output_window<int> *) { /* Stub */ } 
  void super_kernel_input_casc(input_window<int> *,input_window<int> *,output_stream<acc48_t> *) { /* Stub */ } 
#endif
