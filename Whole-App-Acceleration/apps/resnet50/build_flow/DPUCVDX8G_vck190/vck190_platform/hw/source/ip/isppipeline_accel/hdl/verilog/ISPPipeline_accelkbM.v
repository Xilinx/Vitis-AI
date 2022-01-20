// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================

`timescale 1ns/1ps

module ISPPipeline_accelkbM #(
parameter
    ID                = 0,
    NUM_STAGE         = 1,
    din0_WIDTH       = 32,
    din1_WIDTH       = 32,
    din2_WIDTH       = 32,
    din3_WIDTH       = 32,
    din4_WIDTH       = 32,
    din5_WIDTH       = 32,
    din6_WIDTH       = 32,
    din7_WIDTH       = 32,
    din8_WIDTH       = 32,
    din9_WIDTH       = 32,
    din10_WIDTH         = 32,
    dout_WIDTH            = 32
)(
    input  [9 : 0]     din0,
    input  [9 : 0]     din1,
    input  [9 : 0]     din2,
    input  [9 : 0]     din3,
    input  [9 : 0]     din4,
    input  [9 : 0]     din5,
    input  [9 : 0]     din6,
    input  [9 : 0]     din7,
    input  [9 : 0]     din8,
    input  [9 : 0]     din9,
    input  [3 : 0]    din10,
    output [9 : 0]   dout);

// puts internal signals
wire [3 : 0]     sel;
// level 1 signals
wire [9 : 0]         mux_1_0;
wire [9 : 0]         mux_1_1;
wire [9 : 0]         mux_1_2;
wire [9 : 0]         mux_1_3;
wire [9 : 0]         mux_1_4;
// level 2 signals
wire [9 : 0]         mux_2_0;
wire [9 : 0]         mux_2_1;
wire [9 : 0]         mux_2_2;
// level 3 signals
wire [9 : 0]         mux_3_0;
wire [9 : 0]         mux_3_1;
// level 4 signals
wire [9 : 0]         mux_4_0;

assign sel = din10;

// Generate level 1 logic
assign mux_1_0 = (sel[0] == 0)? din0 : din1;
assign mux_1_1 = (sel[0] == 0)? din2 : din3;
assign mux_1_2 = (sel[0] == 0)? din4 : din5;
assign mux_1_3 = (sel[0] == 0)? din6 : din7;
assign mux_1_4 = (sel[0] == 0)? din8 : din9;

// Generate level 2 logic
assign mux_2_0 = (sel[1] == 0)? mux_1_0 : mux_1_1;
assign mux_2_1 = (sel[1] == 0)? mux_1_2 : mux_1_3;
assign mux_2_2 = mux_1_4;

// Generate level 3 logic
assign mux_3_0 = (sel[2] == 0)? mux_2_0 : mux_2_1;
assign mux_3_1 = mux_2_2;

// Generate level 4 logic
assign mux_4_0 = (sel[3] == 0)? mux_3_0 : mux_3_1;

// output logic
assign dout = mux_4_0;

endmodule
