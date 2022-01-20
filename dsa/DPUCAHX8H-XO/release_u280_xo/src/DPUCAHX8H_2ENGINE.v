//Copyright 1986-2018 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2018.3 (lin64) Build 2405991 Thu Dec  6 23:36:41 MST 2018
//Date        : Tue Jul 16 16:54:57 2019
//Host        : cnn_s79 running 64-bit CentOS Linux release 7.5.1804 (Core)
//Command     : generate_target v3e.bd
//Design      : DPUCAHX8H_2ENGINE
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module DPUCAHX8H_2ENGINE
   (DPU_AXI_0_araddr,
    DPU_AXI_0_arburst,
    DPU_AXI_0_arcache,
    DPU_AXI_0_arid,
    DPU_AXI_0_arlen,
    DPU_AXI_0_arlock,
    DPU_AXI_0_arprot,
    DPU_AXI_0_arqos,
    DPU_AXI_0_arready,
    DPU_AXI_0_arsize,
    DPU_AXI_0_arvalid,
    DPU_AXI_0_awaddr,
    DPU_AXI_0_awburst,
    DPU_AXI_0_awcache,
    DPU_AXI_0_awid,
    DPU_AXI_0_awlen,
    DPU_AXI_0_awlock,
    DPU_AXI_0_awprot,
    DPU_AXI_0_awqos,
    DPU_AXI_0_awready,
    DPU_AXI_0_awsize,
    DPU_AXI_0_awvalid,
    DPU_AXI_0_bid,
    DPU_AXI_0_bready,
    DPU_AXI_0_bresp,
    DPU_AXI_0_bvalid,
    DPU_AXI_0_rdata,
    DPU_AXI_0_rid,
    DPU_AXI_0_rlast,
    DPU_AXI_0_rready,
    DPU_AXI_0_rresp,
    DPU_AXI_0_rvalid,
    DPU_AXI_0_wdata,
    DPU_AXI_0_wid,
    DPU_AXI_0_wlast,
    DPU_AXI_0_wready,
    DPU_AXI_0_wstrb,
    DPU_AXI_0_wvalid,
    DPU_AXI_1_araddr,
    DPU_AXI_1_arburst,
    DPU_AXI_1_arcache,
    DPU_AXI_1_arid,
    DPU_AXI_1_arlen,
    DPU_AXI_1_arlock,
    DPU_AXI_1_arprot,
    DPU_AXI_1_arqos,
    DPU_AXI_1_arready,
    DPU_AXI_1_arsize,
    DPU_AXI_1_arvalid,
    DPU_AXI_1_awaddr,
    DPU_AXI_1_awburst,
    DPU_AXI_1_awcache,
    DPU_AXI_1_awid,
    DPU_AXI_1_awlen,
    DPU_AXI_1_awlock,
    DPU_AXI_1_awprot,
    DPU_AXI_1_awqos,
    DPU_AXI_1_awready,
    DPU_AXI_1_awsize,
    DPU_AXI_1_awvalid,
    DPU_AXI_1_bid,
    DPU_AXI_1_bready,
    DPU_AXI_1_bresp,
    DPU_AXI_1_bvalid,
    DPU_AXI_1_rdata,
    DPU_AXI_1_rid,
    DPU_AXI_1_rlast,
    DPU_AXI_1_rready,
    DPU_AXI_1_rresp,
    DPU_AXI_1_rvalid,
    DPU_AXI_1_wdata,
    DPU_AXI_1_wid,
    DPU_AXI_1_wlast,
    DPU_AXI_1_wready,
    DPU_AXI_1_wstrb,
    DPU_AXI_1_wvalid,
    DPU_AXI_I0_araddr,
    DPU_AXI_I0_arburst,
    DPU_AXI_I0_arcache,
    DPU_AXI_I0_arid,
    DPU_AXI_I0_arlen,
    DPU_AXI_I0_arlock,
    DPU_AXI_I0_arprot,
    DPU_AXI_I0_arqos,
    DPU_AXI_I0_arready,
    DPU_AXI_I0_arsize,
    DPU_AXI_I0_arvalid,
    DPU_AXI_I0_awaddr,
    DPU_AXI_I0_awburst,
    DPU_AXI_I0_awcache,
    DPU_AXI_I0_awid,
    DPU_AXI_I0_awlen,
    DPU_AXI_I0_awlock,
    DPU_AXI_I0_awprot,
    DPU_AXI_I0_awqos,
    DPU_AXI_I0_awready,
    DPU_AXI_I0_awsize,
    DPU_AXI_I0_awvalid,
    DPU_AXI_I0_bid,
    DPU_AXI_I0_bready,
    DPU_AXI_I0_bresp,
    DPU_AXI_I0_bvalid,
    DPU_AXI_I0_rdata,
    DPU_AXI_I0_rid,
    DPU_AXI_I0_rlast,
    DPU_AXI_I0_rready,
    DPU_AXI_I0_rresp,
    DPU_AXI_I0_rvalid,
    DPU_AXI_I0_wdata,
    DPU_AXI_I0_wid,
    DPU_AXI_I0_wlast,
    DPU_AXI_I0_wready,
    DPU_AXI_I0_wstrb,
    DPU_AXI_I0_wvalid,
    DPU_AXI_W0_araddr,
    DPU_AXI_W0_arburst,
    DPU_AXI_W0_arcache,
    DPU_AXI_W0_arid,
    DPU_AXI_W0_arlen,
    DPU_AXI_W0_arlock,
    DPU_AXI_W0_arprot,
    DPU_AXI_W0_arqos,
    DPU_AXI_W0_arready,
    DPU_AXI_W0_arsize,
    DPU_AXI_W0_arvalid,
    DPU_AXI_W0_rdata,
    DPU_AXI_W0_rid,
    DPU_AXI_W0_rlast,
    DPU_AXI_W0_rready,
    DPU_AXI_W0_rresp,
    DPU_AXI_W0_rvalid,
    DPU_AXI_W1_araddr,
    DPU_AXI_W1_arburst,
    DPU_AXI_W1_arcache,
    DPU_AXI_W1_arid,
    DPU_AXI_W1_arlen,
    DPU_AXI_W1_arlock,
    DPU_AXI_W1_arprot,
    DPU_AXI_W1_arqos,
    DPU_AXI_W1_arready,
    DPU_AXI_W1_arsize,
    DPU_AXI_W1_arvalid,
    DPU_AXI_W1_rdata,
    DPU_AXI_W1_rid,
    DPU_AXI_W1_rlast,
    DPU_AXI_W1_rready,
    DPU_AXI_W1_rresp,
    DPU_AXI_W1_rvalid,
    ap_clk,
    ap_clk_2,
    ap_rst_n,
    ap_rst_n_2,
    interrupt,
    s_axi_control_araddr,
    s_axi_control_arprot,
    s_axi_control_arready,
    s_axi_control_arvalid,
    s_axi_control_awaddr,
    s_axi_control_awprot,
    s_axi_control_awready,
    s_axi_control_awvalid,
    s_axi_control_bready,
    s_axi_control_bresp,
    s_axi_control_bvalid,
    s_axi_control_rdata,
    s_axi_control_rready,
    s_axi_control_rresp,
    s_axi_control_rvalid,
    s_axi_control_wdata,
    s_axi_control_wready,
    s_axi_control_wstrb,
    s_axi_control_wvalid);
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk CLK" *)
(* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF DPU_AXI_0:DPU_AXI_1:DPU_AXI_I0:DPU_AXI_W0:DPU_AXI_W1:s_axi_control, ASSOCIATED_RESET ap_rst_n" *)
  input ap_clk;
(* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ap_clk_2 CLK" *)
(* X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ap_rst_n_2" *)
  input ap_clk_2;
(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n RST" *)
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input ap_rst_n;
(* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ap_rst_n_2 RST" *)
(* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
  input ap_rst_n_2;
//(* X_INTERFACE_PARAMETER = "CLK_DOMAIN <value>,PHASE <value>,MAX_BURST_LENGTH <value>,NUM_WRITE_OUTSTANDING <value>,NUM_READ_OUTSTANDING <value>,SUPPORTS_NARROW_BURST <value>,READ_WRITE_MODE <value>,BUSER_WIDTH <value>,RUSER_WIDTH <value>,WUSER_WIDTH <value>,ARUSER_WIDTH <value>,AWUSER_WIDTH <value>,ADDR_WIDTH <value>,ID_WIDTH <value>,FREQ_HZ <value>,PROTOCOL <value>,DATA_WIDTH <value>,HAS_BURST <value>,HAS_CACHE <value>,HAS_LOCK <value>,HAS_PROT <value>,HAS_QOS <value>,HAS_REGION <value>,HAS_WSTRB <value>,HAS_BRESP <value>,HAS_RRESP <value>" *)
(* X_INTERFACE_PARAMETER = "MAX_BURST_LENGTH 128,NUM_WRITE_OUTSTANDING 32,NUM_READ_OUTSTANDING 32" *)
  output [32:0]DPU_AXI_0_araddr;
  output [1:0]DPU_AXI_0_arburst;
  output [3:0]DPU_AXI_0_arcache;
  output [5:0]DPU_AXI_0_arid;
  output [3:0]DPU_AXI_0_arlen;
  output [1:0]DPU_AXI_0_arlock;
  output [2:0]DPU_AXI_0_arprot;
  output [3:0]DPU_AXI_0_arqos;
  input DPU_AXI_0_arready;
  output [2:0]DPU_AXI_0_arsize;
  output DPU_AXI_0_arvalid;
  output [32:0]DPU_AXI_0_awaddr;
  output [1:0]DPU_AXI_0_awburst;
  output [3:0]DPU_AXI_0_awcache;
  output [5:0]DPU_AXI_0_awid;
  output [3:0]DPU_AXI_0_awlen;
  output [1:0]DPU_AXI_0_awlock;
  output [2:0]DPU_AXI_0_awprot;
  output [3:0]DPU_AXI_0_awqos;
  input DPU_AXI_0_awready;
  output [2:0]DPU_AXI_0_awsize;
  output DPU_AXI_0_awvalid;
  input [5:0]DPU_AXI_0_bid;
  output DPU_AXI_0_bready;
  input [1:0]DPU_AXI_0_bresp;
  input DPU_AXI_0_bvalid;
  input [255:0]DPU_AXI_0_rdata;
  input [5:0]DPU_AXI_0_rid;
  input DPU_AXI_0_rlast;
  output DPU_AXI_0_rready;
  input [1:0]DPU_AXI_0_rresp;
  input DPU_AXI_0_rvalid;
  output [255:0]DPU_AXI_0_wdata;
  output [5:0]DPU_AXI_0_wid;
  output DPU_AXI_0_wlast;
  input DPU_AXI_0_wready;
  output [31:0]DPU_AXI_0_wstrb;
  output DPU_AXI_0_wvalid;
(* X_INTERFACE_PARAMETER = "MAX_BURST_LENGTH 128,NUM_WRITE_OUTSTANDING 32,NUM_READ_OUTSTANDING 32" *)
  output [32:0]DPU_AXI_1_araddr;
  output [1:0]DPU_AXI_1_arburst;
  output [3:0]DPU_AXI_1_arcache;
  output [5:0]DPU_AXI_1_arid;
  output [3:0]DPU_AXI_1_arlen;
  output [1:0]DPU_AXI_1_arlock;
  output [2:0]DPU_AXI_1_arprot;
  output [3:0]DPU_AXI_1_arqos;
  input DPU_AXI_1_arready;
  output [2:0]DPU_AXI_1_arsize;
  output DPU_AXI_1_arvalid;
  output [32:0]DPU_AXI_1_awaddr;
  output [1:0]DPU_AXI_1_awburst;
  output [3:0]DPU_AXI_1_awcache;
  output [5:0]DPU_AXI_1_awid;
  output [3:0]DPU_AXI_1_awlen;
  output [1:0]DPU_AXI_1_awlock;
  output [2:0]DPU_AXI_1_awprot;
  output [3:0]DPU_AXI_1_awqos;
  input DPU_AXI_1_awready;
  output [2:0]DPU_AXI_1_awsize;
  output DPU_AXI_1_awvalid;
  input [5:0]DPU_AXI_1_bid;
  output DPU_AXI_1_bready;
  input [1:0]DPU_AXI_1_bresp;
  input DPU_AXI_1_bvalid;
  input [255:0]DPU_AXI_1_rdata;
  input [5:0]DPU_AXI_1_rid;
  input DPU_AXI_1_rlast;
  output DPU_AXI_1_rready;
  input [1:0]DPU_AXI_1_rresp;
  input DPU_AXI_1_rvalid;
  output [255:0]DPU_AXI_1_wdata;
  output [5:0]DPU_AXI_1_wid;
  output DPU_AXI_1_wlast;
  input DPU_AXI_1_wready;
  output [31:0]DPU_AXI_1_wstrb;
  output DPU_AXI_1_wvalid;
(* X_INTERFACE_PARAMETER = "MAX_BURST_LENGTH 128,NUM_WRITE_OUTSTANDING 32,NUM_READ_OUTSTANDING 32" *)
  output [32:0]DPU_AXI_I0_araddr;
  output [1:0]DPU_AXI_I0_arburst;
  output [3:0]DPU_AXI_I0_arcache;
  output [5:0]DPU_AXI_I0_arid;
  output [3:0]DPU_AXI_I0_arlen;
  output [1:0]DPU_AXI_I0_arlock;
  output [2:0]DPU_AXI_I0_arprot;
  output [3:0]DPU_AXI_I0_arqos;
  input DPU_AXI_I0_arready;
  output [2:0]DPU_AXI_I0_arsize;
  output DPU_AXI_I0_arvalid;
  output [32:0]DPU_AXI_I0_awaddr;
  output [1:0]DPU_AXI_I0_awburst;
  output [3:0]DPU_AXI_I0_awcache;
  output [5:0]DPU_AXI_I0_awid;
  output [3:0]DPU_AXI_I0_awlen;
  output [1:0]DPU_AXI_I0_awlock;
  output [2:0]DPU_AXI_I0_awprot;
  output [3:0]DPU_AXI_I0_awqos;
  input DPU_AXI_I0_awready;
  output [2:0]DPU_AXI_I0_awsize;
  output DPU_AXI_I0_awvalid;
  input [5:0]DPU_AXI_I0_bid;
  output DPU_AXI_I0_bready;
  input [1:0]DPU_AXI_I0_bresp;
  input DPU_AXI_I0_bvalid;
  input [255:0]DPU_AXI_I0_rdata;
  input [5:0]DPU_AXI_I0_rid;
  input DPU_AXI_I0_rlast;
  output DPU_AXI_I0_rready;
  input [1:0]DPU_AXI_I0_rresp;
  input DPU_AXI_I0_rvalid;
  output [255:0]DPU_AXI_I0_wdata;
  output [5:0]DPU_AXI_I0_wid;
  output DPU_AXI_I0_wlast;
  input DPU_AXI_I0_wready;
  output [31:0]DPU_AXI_I0_wstrb;
  output DPU_AXI_I0_wvalid;
(* X_INTERFACE_PARAMETER = "MAX_BURST_LENGTH 128,NUM_WRITE_OUTSTANDING 32,NUM_READ_OUTSTANDING 32" *)
  output [32:0]DPU_AXI_W0_araddr;
  output [1:0]DPU_AXI_W0_arburst;
  output [3:0]DPU_AXI_W0_arcache;
  output [5:0]DPU_AXI_W0_arid;
  output [3:0]DPU_AXI_W0_arlen;
  output [1:0]DPU_AXI_W0_arlock;
  output [2:0]DPU_AXI_W0_arprot;
  output [3:0]DPU_AXI_W0_arqos;
  input DPU_AXI_W0_arready;
  output [2:0]DPU_AXI_W0_arsize;
  output DPU_AXI_W0_arvalid;
  input [255:0]DPU_AXI_W0_rdata;
  input [5:0]DPU_AXI_W0_rid;
  input DPU_AXI_W0_rlast;
  output DPU_AXI_W0_rready;
  input [1:0]DPU_AXI_W0_rresp;
  input DPU_AXI_W0_rvalid;
(* X_INTERFACE_PARAMETER = "MAX_BURST_LENGTH 128,NUM_WRITE_OUTSTANDING 32,NUM_READ_OUTSTANDING 32" *)
  output [32:0]DPU_AXI_W1_araddr;
  output [1:0]DPU_AXI_W1_arburst;
  output [3:0]DPU_AXI_W1_arcache;
  output [5:0]DPU_AXI_W1_arid;
  output [3:0]DPU_AXI_W1_arlen;
  output [1:0]DPU_AXI_W1_arlock;
  output [2:0]DPU_AXI_W1_arprot;
  output [3:0]DPU_AXI_W1_arqos;
  input DPU_AXI_W1_arready;
  output [2:0]DPU_AXI_W1_arsize;
  output DPU_AXI_W1_arvalid;
  input [255:0]DPU_AXI_W1_rdata;
  input [5:0]DPU_AXI_W1_rid;
  input DPU_AXI_W1_rlast;
  output DPU_AXI_W1_rready;
  input [1:0]DPU_AXI_W1_rresp;
  input DPU_AXI_W1_rvalid;
  output [0:0]interrupt;
  input [11:0]s_axi_control_araddr;
  input [2:0]s_axi_control_arprot;
  output s_axi_control_arready;
  input s_axi_control_arvalid;
  input [11:0]s_axi_control_awaddr;
  input [2:0]s_axi_control_awprot;
  output s_axi_control_awready;
  input s_axi_control_awvalid;
  input s_axi_control_bready;
  output [1:0]s_axi_control_bresp;
  output s_axi_control_bvalid;
  output [31:0]s_axi_control_rdata;
  input s_axi_control_rready;
  output [1:0]s_axi_control_rresp;
  output s_axi_control_rvalid;
  input [31:0]s_axi_control_wdata;
  output s_axi_control_wready;
  input [3:0]s_axi_control_wstrb;
  input s_axi_control_wvalid;

  wire [32:0]DPU_AXI_0_araddr;
  wire [1:0]DPU_AXI_0_arburst;
  wire [3:0]DPU_AXI_0_arcache;
  wire [5:0]DPU_AXI_0_arid;
  wire [3:0]DPU_AXI_0_arlen;
  wire [1:0]DPU_AXI_0_arlock;
  wire [2:0]DPU_AXI_0_arprot;
  wire [3:0]DPU_AXI_0_arqos;
  wire DPU_AXI_0_arready;
  wire [2:0]DPU_AXI_0_arsize;
  wire DPU_AXI_0_arvalid;
  wire [32:0]DPU_AXI_0_awaddr;
  wire [1:0]DPU_AXI_0_awburst;
  wire [3:0]DPU_AXI_0_awcache;
  wire [5:0]DPU_AXI_0_awid;
  wire [3:0]DPU_AXI_0_awlen;
  wire [1:0]DPU_AXI_0_awlock;
  wire [2:0]DPU_AXI_0_awprot;
  wire [3:0]DPU_AXI_0_awqos;
  wire DPU_AXI_0_awready;
  wire [2:0]DPU_AXI_0_awsize;
  wire DPU_AXI_0_awvalid;
  wire [5:0]DPU_AXI_0_bid;
  wire DPU_AXI_0_bready;
  wire [1:0]DPU_AXI_0_bresp;
  wire DPU_AXI_0_bvalid;
  wire [255:0]DPU_AXI_0_rdata;
  wire [5:0]DPU_AXI_0_rid;
  wire DPU_AXI_0_rlast;
  wire DPU_AXI_0_rready;
  wire [1:0]DPU_AXI_0_rresp;
  wire DPU_AXI_0_rvalid;
  wire [255:0]DPU_AXI_0_wdata;
  wire [5:0]DPU_AXI_0_wid;
  wire DPU_AXI_0_wlast;
  wire DPU_AXI_0_wready;
  wire [31:0]DPU_AXI_0_wstrb;
  wire DPU_AXI_0_wvalid;
  wire [32:0]DPU_AXI_1_araddr;
  wire [1:0]DPU_AXI_1_arburst;
  wire [3:0]DPU_AXI_1_arcache;
  wire [5:0]DPU_AXI_1_arid;
  wire [3:0]DPU_AXI_1_arlen;
  wire [1:0]DPU_AXI_1_arlock;
  wire [2:0]DPU_AXI_1_arprot;
  wire [3:0]DPU_AXI_1_arqos;
  wire DPU_AXI_1_arready;
  wire [2:0]DPU_AXI_1_arsize;
  wire DPU_AXI_1_arvalid;
  wire [32:0]DPU_AXI_1_awaddr;
  wire [1:0]DPU_AXI_1_awburst;
  wire [3:0]DPU_AXI_1_awcache;
  wire [5:0]DPU_AXI_1_awid;
  wire [3:0]DPU_AXI_1_awlen;
  wire [1:0]DPU_AXI_1_awlock;
  wire [2:0]DPU_AXI_1_awprot;
  wire [3:0]DPU_AXI_1_awqos;
  wire DPU_AXI_1_awready;
  wire [2:0]DPU_AXI_1_awsize;
  wire DPU_AXI_1_awvalid;
  wire [5:0]DPU_AXI_1_bid;
  wire DPU_AXI_1_bready;
  wire [1:0]DPU_AXI_1_bresp;
  wire DPU_AXI_1_bvalid;
  wire [255:0]DPU_AXI_1_rdata;
  wire [5:0]DPU_AXI_1_rid;
  wire DPU_AXI_1_rlast;
  wire DPU_AXI_1_rready;
  wire [1:0]DPU_AXI_1_rresp;
  wire DPU_AXI_1_rvalid;
  wire [255:0]DPU_AXI_1_wdata;
  wire [5:0]DPU_AXI_1_wid;
  wire DPU_AXI_1_wlast;
  wire DPU_AXI_1_wready;
  wire [31:0]DPU_AXI_1_wstrb;
  wire DPU_AXI_1_wvalid;
  wire [32:0]DPU_AXI_I0_araddr;
  wire [1:0]DPU_AXI_I0_arburst;
  wire [3:0]DPU_AXI_I0_arcache;
  wire [5:0]DPU_AXI_I0_arid;
  wire [3:0]DPU_AXI_I0_arlen;
  wire [1:0]DPU_AXI_I0_arlock;
  wire [2:0]DPU_AXI_I0_arprot;
  wire [3:0]DPU_AXI_I0_arqos;
  wire DPU_AXI_I0_arready;
  wire [2:0]DPU_AXI_I0_arsize;
  wire DPU_AXI_I0_arvalid;
  wire [32:0]DPU_AXI_I0_awaddr;
  wire [1:0]DPU_AXI_I0_awburst;
  wire [3:0]DPU_AXI_I0_awcache;
  wire [5:0]DPU_AXI_I0_awid;
  wire [3:0]DPU_AXI_I0_awlen;
  wire [1:0]DPU_AXI_I0_awlock;
  wire [2:0]DPU_AXI_I0_awprot;
  wire [3:0]DPU_AXI_I0_awqos;
  wire DPU_AXI_I0_awready;
  wire [2:0]DPU_AXI_I0_awsize;
  wire DPU_AXI_I0_awvalid;
  wire [5:0]DPU_AXI_I0_bid;
  wire DPU_AXI_I0_bready;
  wire [1:0]DPU_AXI_I0_bresp;
  wire DPU_AXI_I0_bvalid;
  wire [255:0]DPU_AXI_I0_rdata;
  wire [5:0]DPU_AXI_I0_rid;
  wire DPU_AXI_I0_rlast;
  wire DPU_AXI_I0_rready;
  wire [1:0]DPU_AXI_I0_rresp;
  wire DPU_AXI_I0_rvalid;
  wire [255:0]DPU_AXI_I0_wdata;
  wire [5:0]DPU_AXI_I0_wid;
  wire DPU_AXI_I0_wlast;
  wire DPU_AXI_I0_wready;
  wire [31:0]DPU_AXI_I0_wstrb;
  wire DPU_AXI_I0_wvalid;
  wire [32:0]DPU_AXI_W0_araddr;
  wire [1:0]DPU_AXI_W0_arburst;
  wire [3:0]DPU_AXI_W0_arcache;
  wire [5:0]DPU_AXI_W0_arid;
  wire [3:0]DPU_AXI_W0_arlen;
  wire [1:0]DPU_AXI_W0_arlock;
  wire [2:0]DPU_AXI_W0_arprot;
  wire [3:0]DPU_AXI_W0_arqos;
  wire DPU_AXI_W0_arready;
  wire [2:0]DPU_AXI_W0_arsize;
  wire DPU_AXI_W0_arvalid;
  wire [255:0]DPU_AXI_W0_rdata;
  wire [5:0]DPU_AXI_W0_rid;
  wire DPU_AXI_W0_rlast;
  wire DPU_AXI_W0_rready;
  wire [1:0]DPU_AXI_W0_rresp;
  wire DPU_AXI_W0_rvalid;
  wire [32:0]DPU_AXI_W1_araddr;
  wire [1:0]DPU_AXI_W1_arburst;
  wire [3:0]DPU_AXI_W1_arcache;
  wire [5:0]DPU_AXI_W1_arid;
  wire [3:0]DPU_AXI_W1_arlen;
  wire [1:0]DPU_AXI_W1_arlock;
  wire [2:0]DPU_AXI_W1_arprot;
  wire [3:0]DPU_AXI_W1_arqos;
  wire DPU_AXI_W1_arready;
  wire [2:0]DPU_AXI_W1_arsize;
  wire DPU_AXI_W1_arvalid;
  wire [255:0]DPU_AXI_W1_rdata;
  wire [5:0]DPU_AXI_W1_rid;
  wire DPU_AXI_W1_rlast;
  wire DPU_AXI_W1_rready;
  wire [1:0]DPU_AXI_W1_rresp;
  wire DPU_AXI_W1_rvalid;
  wire ap_clk;
  wire ap_clk_2;
  wire ap_rst_n;
  wire ap_rst_n_2;
  wire [0:0]interrupt;
  wire [11:0]s_axi_control_araddr;
  wire [2:0]s_axi_control_arprot;
  wire s_axi_control_arready;
  wire s_axi_control_arvalid;
  wire [11:0]s_axi_control_awaddr;
  wire [2:0]s_axi_control_awprot;
  wire s_axi_control_awready;
  wire s_axi_control_awvalid;
  wire s_axi_control_bready;
  wire [1:0]s_axi_control_bresp;
  wire s_axi_control_bvalid;
  wire [31:0]s_axi_control_rdata;
  wire s_axi_control_rready;
  wire [1:0]s_axi_control_rresp;
  wire s_axi_control_rvalid;
  wire [31:0]s_axi_control_wdata;
  wire s_axi_control_wready;
  wire [3:0]s_axi_control_wstrb;
  wire s_axi_control_wvalid;

  v3e_bd v3e_bd_i              (
        .DPU_AXI_0_araddr      ( DPU_AXI_0_araddr      ) ,
        .DPU_AXI_0_arburst     ( DPU_AXI_0_arburst     ) ,
        .DPU_AXI_0_arcache     ( DPU_AXI_0_arcache     ) ,
        .DPU_AXI_0_arid        ( DPU_AXI_0_arid        ) ,
        .DPU_AXI_0_arlen       ( DPU_AXI_0_arlen       ) ,
        .DPU_AXI_0_arlock      ( DPU_AXI_0_arlock      ) ,
        .DPU_AXI_0_arprot      ( DPU_AXI_0_arprot      ) ,
        .DPU_AXI_0_arqos       ( DPU_AXI_0_arqos       ) ,
        .DPU_AXI_0_arready     ( DPU_AXI_0_arready     ) ,
        .DPU_AXI_0_arsize      ( DPU_AXI_0_arsize      ) ,
        .DPU_AXI_0_arvalid     ( DPU_AXI_0_arvalid     ) ,
        .DPU_AXI_0_awaddr      ( DPU_AXI_0_awaddr      ) ,
        .DPU_AXI_0_awburst     ( DPU_AXI_0_awburst     ) ,
        .DPU_AXI_0_awcache     ( DPU_AXI_0_awcache     ) ,
        .DPU_AXI_0_awid        ( DPU_AXI_0_awid        ) ,
        .DPU_AXI_0_awlen       ( DPU_AXI_0_awlen       ) ,
        .DPU_AXI_0_awlock      ( DPU_AXI_0_awlock      ) ,
        .DPU_AXI_0_awprot      ( DPU_AXI_0_awprot      ) ,
        .DPU_AXI_0_awqos       ( DPU_AXI_0_awqos       ) ,
        .DPU_AXI_0_awready     ( DPU_AXI_0_awready     ) ,
        .DPU_AXI_0_awsize      ( DPU_AXI_0_awsize      ) ,
        .DPU_AXI_0_awvalid     ( DPU_AXI_0_awvalid     ) ,
        .DPU_AXI_0_bid         ( DPU_AXI_0_bid         ) ,
        .DPU_AXI_0_bready      ( DPU_AXI_0_bready      ) ,
        .DPU_AXI_0_bresp       ( DPU_AXI_0_bresp       ) ,
        .DPU_AXI_0_bvalid      ( DPU_AXI_0_bvalid      ) ,
        .DPU_AXI_0_rdata       ( DPU_AXI_0_rdata       ) ,
        .DPU_AXI_0_rid         ( DPU_AXI_0_rid         ) ,
        .DPU_AXI_0_rlast       ( DPU_AXI_0_rlast       ) ,
        .DPU_AXI_0_rready      ( DPU_AXI_0_rready      ) ,
        .DPU_AXI_0_rresp       ( DPU_AXI_0_rresp       ) ,
        .DPU_AXI_0_rvalid      ( DPU_AXI_0_rvalid      ) ,
        .DPU_AXI_0_wdata       ( DPU_AXI_0_wdata       ) ,
        .DPU_AXI_0_wid         ( DPU_AXI_0_wid         ) ,
        .DPU_AXI_0_wlast       ( DPU_AXI_0_wlast       ) ,
        .DPU_AXI_0_wready      ( DPU_AXI_0_wready      ) ,
        .DPU_AXI_0_wstrb       ( DPU_AXI_0_wstrb       ) ,
        .DPU_AXI_0_wvalid      ( DPU_AXI_0_wvalid      ) ,
        .DPU_AXI_1_araddr      ( DPU_AXI_1_araddr      ) ,
        .DPU_AXI_1_arburst     ( DPU_AXI_1_arburst     ) ,
        .DPU_AXI_1_arcache     ( DPU_AXI_1_arcache     ) ,
        .DPU_AXI_1_arid        ( DPU_AXI_1_arid        ) ,
        .DPU_AXI_1_arlen       ( DPU_AXI_1_arlen       ) ,
        .DPU_AXI_1_arlock      ( DPU_AXI_1_arlock      ) ,
        .DPU_AXI_1_arprot      ( DPU_AXI_1_arprot      ) ,
        .DPU_AXI_1_arqos       ( DPU_AXI_1_arqos       ) ,
        .DPU_AXI_1_arready     ( DPU_AXI_1_arready     ) ,
        .DPU_AXI_1_arsize      ( DPU_AXI_1_arsize      ) ,
        .DPU_AXI_1_arvalid     ( DPU_AXI_1_arvalid     ) ,
        .DPU_AXI_1_awaddr      ( DPU_AXI_1_awaddr      ) ,
        .DPU_AXI_1_awburst     ( DPU_AXI_1_awburst     ) ,
        .DPU_AXI_1_awcache     ( DPU_AXI_1_awcache     ) ,
        .DPU_AXI_1_awid        ( DPU_AXI_1_awid        ) ,
        .DPU_AXI_1_awlen       ( DPU_AXI_1_awlen       ) ,
        .DPU_AXI_1_awlock      ( DPU_AXI_1_awlock      ) ,
        .DPU_AXI_1_awprot      ( DPU_AXI_1_awprot      ) ,
        .DPU_AXI_1_awqos       ( DPU_AXI_1_awqos       ) ,
        .DPU_AXI_1_awready     ( DPU_AXI_1_awready     ) ,
        .DPU_AXI_1_awsize      ( DPU_AXI_1_awsize      ) ,
        .DPU_AXI_1_awvalid     ( DPU_AXI_1_awvalid     ) ,
        .DPU_AXI_1_bid         ( DPU_AXI_1_bid         ) ,
        .DPU_AXI_1_bready      ( DPU_AXI_1_bready      ) ,
        .DPU_AXI_1_bresp       ( DPU_AXI_1_bresp       ) ,
        .DPU_AXI_1_bvalid      ( DPU_AXI_1_bvalid      ) ,
        .DPU_AXI_1_rdata       ( DPU_AXI_1_rdata       ) ,
        .DPU_AXI_1_rid         ( DPU_AXI_1_rid         ) ,
        .DPU_AXI_1_rlast       ( DPU_AXI_1_rlast       ) ,
        .DPU_AXI_1_rready      ( DPU_AXI_1_rready      ) ,
        .DPU_AXI_1_rresp       ( DPU_AXI_1_rresp       ) ,
        .DPU_AXI_1_rvalid      ( DPU_AXI_1_rvalid      ) ,
        .DPU_AXI_1_wdata       ( DPU_AXI_1_wdata       ) ,
        .DPU_AXI_1_wid         ( DPU_AXI_1_wid         ) ,
        .DPU_AXI_1_wlast       ( DPU_AXI_1_wlast       ) ,
        .DPU_AXI_1_wready      ( DPU_AXI_1_wready      ) ,
        .DPU_AXI_1_wstrb       ( DPU_AXI_1_wstrb       ) ,
        .DPU_AXI_1_wvalid      ( DPU_AXI_1_wvalid      ) ,
        .DPU_AXI_I0_araddr      ( DPU_AXI_I0_araddr      ) ,
        .DPU_AXI_I0_arburst     ( DPU_AXI_I0_arburst     ) ,
        .DPU_AXI_I0_arcache     ( DPU_AXI_I0_arcache     ) ,
        .DPU_AXI_I0_arid        ( DPU_AXI_I0_arid        ) ,
        .DPU_AXI_I0_arlen       ( DPU_AXI_I0_arlen       ) ,
        .DPU_AXI_I0_arlock      ( DPU_AXI_I0_arlock      ) ,
        .DPU_AXI_I0_arprot      ( DPU_AXI_I0_arprot      ) ,
        .DPU_AXI_I0_arqos       ( DPU_AXI_I0_arqos       ) ,
        .DPU_AXI_I0_arready     ( DPU_AXI_I0_arready     ) ,
        .DPU_AXI_I0_arsize      ( DPU_AXI_I0_arsize      ) ,
        .DPU_AXI_I0_arvalid     ( DPU_AXI_I0_arvalid     ) ,
        .DPU_AXI_I0_awaddr      ( DPU_AXI_I0_awaddr      ) ,
        .DPU_AXI_I0_awburst     ( DPU_AXI_I0_awburst     ) ,
        .DPU_AXI_I0_awcache     ( DPU_AXI_I0_awcache     ) ,
        .DPU_AXI_I0_awid        ( DPU_AXI_I0_awid        ) ,
        .DPU_AXI_I0_awlen       ( DPU_AXI_I0_awlen       ) ,
        .DPU_AXI_I0_awlock      ( DPU_AXI_I0_awlock      ) ,
        .DPU_AXI_I0_awprot      ( DPU_AXI_I0_awprot      ) ,
        .DPU_AXI_I0_awqos       ( DPU_AXI_I0_awqos       ) ,
        .DPU_AXI_I0_awready     ( DPU_AXI_I0_awready     ) ,
        .DPU_AXI_I0_awsize      ( DPU_AXI_I0_awsize      ) ,
        .DPU_AXI_I0_awvalid     ( DPU_AXI_I0_awvalid     ) ,
        .DPU_AXI_I0_bid         ( DPU_AXI_I0_bid         ) ,
        .DPU_AXI_I0_bready      ( DPU_AXI_I0_bready      ) ,
        .DPU_AXI_I0_bresp       ( DPU_AXI_I0_bresp       ) ,
        .DPU_AXI_I0_bvalid      ( DPU_AXI_I0_bvalid      ) ,
        .DPU_AXI_I0_rdata       ( DPU_AXI_I0_rdata       ) ,
        .DPU_AXI_I0_rid         ( DPU_AXI_I0_rid         ) ,
        .DPU_AXI_I0_rlast       ( DPU_AXI_I0_rlast       ) ,
        .DPU_AXI_I0_rready      ( DPU_AXI_I0_rready      ) ,
        .DPU_AXI_I0_rresp       ( DPU_AXI_I0_rresp       ) ,
        .DPU_AXI_I0_rvalid      ( DPU_AXI_I0_rvalid      ) ,
        .DPU_AXI_I0_wdata       ( DPU_AXI_I0_wdata       ) ,
        .DPU_AXI_I0_wid         ( DPU_AXI_I0_wid         ) ,
        .DPU_AXI_I0_wlast       ( DPU_AXI_I0_wlast       ) ,
        .DPU_AXI_I0_wready      ( DPU_AXI_I0_wready      ) ,
        .DPU_AXI_I0_wstrb       ( DPU_AXI_I0_wstrb       ) ,
        .DPU_AXI_I0_wvalid      ( DPU_AXI_I0_wvalid      ) ,
        .DPU_AXI_W0_araddr     ( DPU_AXI_W0_araddr     ) ,
        .DPU_AXI_W0_arburst    ( DPU_AXI_W0_arburst    ) ,
        .DPU_AXI_W0_arcache    ( DPU_AXI_W0_arcache    ) ,
        .DPU_AXI_W0_arid       ( DPU_AXI_W0_arid       ) ,
        .DPU_AXI_W0_arlen      ( DPU_AXI_W0_arlen      ) ,
        .DPU_AXI_W0_arlock     ( DPU_AXI_W0_arlock     ) ,
        .DPU_AXI_W0_arprot     ( DPU_AXI_W0_arprot     ) ,
        .DPU_AXI_W0_arqos      ( DPU_AXI_W0_arqos      ) ,
        .DPU_AXI_W0_arready    ( DPU_AXI_W0_arready    ) ,
        .DPU_AXI_W0_arsize     ( DPU_AXI_W0_arsize     ) ,
        .DPU_AXI_W0_arvalid    ( DPU_AXI_W0_arvalid    ) ,
        .DPU_AXI_W0_rdata      ( DPU_AXI_W0_rdata      ) ,
        .DPU_AXI_W0_rid        ( DPU_AXI_W0_rid        ) ,
        .DPU_AXI_W0_rlast      ( DPU_AXI_W0_rlast      ) ,
        .DPU_AXI_W0_rready     ( DPU_AXI_W0_rready     ) ,
        .DPU_AXI_W0_rresp      ( DPU_AXI_W0_rresp      ) ,
        .DPU_AXI_W0_rvalid     ( DPU_AXI_W0_rvalid     ) ,
        .DPU_AXI_W1_araddr     ( DPU_AXI_W1_araddr     ) ,
        .DPU_AXI_W1_arburst    ( DPU_AXI_W1_arburst    ) ,
        .DPU_AXI_W1_arcache    ( DPU_AXI_W1_arcache    ) ,
        .DPU_AXI_W1_arid       ( DPU_AXI_W1_arid       ) ,
        .DPU_AXI_W1_arlen      ( DPU_AXI_W1_arlen      ) ,
        .DPU_AXI_W1_arlock     ( DPU_AXI_W1_arlock     ) ,
        .DPU_AXI_W1_arprot     ( DPU_AXI_W1_arprot     ) ,
        .DPU_AXI_W1_arqos      ( DPU_AXI_W1_arqos      ) ,
        .DPU_AXI_W1_arready    ( DPU_AXI_W1_arready    ) ,
        .DPU_AXI_W1_arsize     ( DPU_AXI_W1_arsize     ) ,
        .DPU_AXI_W1_arvalid    ( DPU_AXI_W1_arvalid    ) ,
        .DPU_AXI_W1_rdata      ( DPU_AXI_W1_rdata      ) ,
        .DPU_AXI_W1_rid        ( DPU_AXI_W1_rid        ) ,
        .DPU_AXI_W1_rlast      ( DPU_AXI_W1_rlast      ) ,
        .DPU_AXI_W1_rready     ( DPU_AXI_W1_rready     ) ,
        .DPU_AXI_W1_rresp      ( DPU_AXI_W1_rresp      ) ,
        .DPU_AXI_W1_rvalid     ( DPU_AXI_W1_rvalid     ) ,
        .ap_clk                ( ap_clk                ) ,
        .ap_clk_2              ( ap_clk_2              ) ,
        .ap_rst_n              ( ap_rst_n              ) ,
        .ap_rst_n_2            ( ap_rst_n_2            ) ,
        .interrupt             ( interrupt             ) ,
        .s_axi_control_araddr  ( s_axi_control_araddr  ) ,
        .s_axi_control_arprot  ( s_axi_control_arprot  ) ,
        .s_axi_control_arready ( s_axi_control_arready ) ,
        .s_axi_control_arvalid ( s_axi_control_arvalid ) ,
        .s_axi_control_awaddr  ( s_axi_control_awaddr  ) ,
        .s_axi_control_awprot  ( s_axi_control_awprot  ) ,
        .s_axi_control_awready ( s_axi_control_awready ) ,
        .s_axi_control_awvalid ( s_axi_control_awvalid ) ,
        .s_axi_control_bready  ( s_axi_control_bready  ) ,
        .s_axi_control_bresp   ( s_axi_control_bresp   ) ,
        .s_axi_control_bvalid  ( s_axi_control_bvalid  ) ,
        .s_axi_control_rdata   ( s_axi_control_rdata   ) ,
        .s_axi_control_rready  ( s_axi_control_rready  ) ,
        .s_axi_control_rresp   ( s_axi_control_rresp   ) ,
        .s_axi_control_rvalid  ( s_axi_control_rvalid  ) ,
        .s_axi_control_wdata   ( s_axi_control_wdata   ) ,
        .s_axi_control_wready  ( s_axi_control_wready  ) ,
        .s_axi_control_wstrb   ( s_axi_control_wstrb   ) ,
        .s_axi_control_wvalid  ( s_axi_control_wvalid  )
    ) ;

endmodule
