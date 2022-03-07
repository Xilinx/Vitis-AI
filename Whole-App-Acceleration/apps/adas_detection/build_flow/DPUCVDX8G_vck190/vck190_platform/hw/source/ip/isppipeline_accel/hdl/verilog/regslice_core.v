// ==============================================================
// Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC v2020.2 (64-bit)
// Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
// ==============================================================

`timescale 1ns/1ps

module regslice_both
#(parameter 
    DataWidth=32
)(
    input ap_clk ,
    input ap_rst,

    input [DataWidth-1:0] data_in , 
    input vld_in , 
    output ack_in ,
    output [DataWidth-1:0] data_out, 
    output vld_out,
    input ack_out,
    output apdone_blk
);
 
localparam W = DataWidth+1;

wire [W-1:0] cdata;
wire cstop;
wire [W-1:0] idata;
wire istop;
wire [W-1:0] odata;
wire ostop;

reg [1:0] count;

ibuf #(
  .W(W)
)
ibuf_inst(
  .clk(ap_clk),
  .reset(ap_rst),
  .idata(idata),
  .istop(istop),
  .cdata(cdata),
  .cstop(cstop)
);
 
 
obuf #(
  .W(W)
)
obuf_inst(
  .clk(ap_clk),
  .reset(ap_rst),
  .cdata(cdata),
  .cstop(cstop),
  .odata(odata),
  .ostop(ostop)
);

assign idata = {vld_in, data_in};
assign ack_in = ~istop;

assign vld_out = odata[W-1];
assign data_out = odata[W-2:0];
assign ostop = ~ack_out;

// count, indicate how many data in the regslice.
// 00 - null
// 10 - 0
// 11 - 1
// 01 - 2
always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        count <= 2'd0;
    end else begin
        if ((((2'd2 == count) & (1'b0 == vld_in)) | ((2'd3 == count) & (1'b0 == vld_in) & (1'b1 == ack_out)))) begin
            count <= 2'd2;
        end else if ((((2'd1 == count) & (1'b0 == ack_out)) | ((2'd3 == count) & (1'b0 == ack_out) & (1'b1 == vld_in)))) begin
            count <= 2'd1;
        end else if (((~((1'b0 == vld_in) & (1'b1 == ack_out)) & ~((1'b0 == ack_out) & (1'b1 == vld_in)) & (2'd3 == count)) | ((2'd1 == count) & (1'b1 == ack_out)) | ((2'd2 == count) & (1'b1 == vld_in)))) begin
            count <= 2'd3;
        end else begin
            count <= 2'd2;
        end
    end
end

assign apdone_blk = ((count == 2'd3 && ack_out == 1'b0) | (count == 2'd1));

endmodule // both


module regslice_forward 
#(parameter 
    DataWidth=32
)(
    input ap_clk ,
    input ap_rst,

    input [DataWidth-1:0] data_in , 
    input vld_in , 
    output ack_in ,
    output [DataWidth-1:0] data_out, 
    output vld_out,
    input ack_out,
    output apdone_blk
);
 
localparam W = DataWidth+1;

wire [W-1:0] cdata;
wire cstop;
wire [W-1:0] idata;
wire istop;
wire [W-1:0] odata;
wire ostop;

obuf #(
  .W(W)
)
obuf_inst(
  .clk(ap_clk),
  .reset(ap_rst),
  .cdata(idata),
  .cstop(istop),
  .odata(odata),
  .ostop(ostop)
);

assign idata = {vld_in, data_in};
assign ack_in = ~istop;

assign vld_out = odata[W-1];
assign data_out = odata[W-2:0];
assign ostop = ~ack_out;

assign apdone_blk = ((ap_rst == 1'b0)&(1'b0 == ack_out)&(1'b1 == vld_out));

endmodule //forward


module regslice_reverse 
#(parameter 
    DataWidth=32
)(
    input ap_clk ,
    input ap_rst,

    input [DataWidth-1:0] data_in , 
    input vld_in , 
    output ack_in ,
    output [DataWidth-1:0] data_out, 
    output vld_out,
    input ack_out,
    output apdone_blk
);
 
localparam W = DataWidth+1;

wire [W-1:0] cdata;
wire cstop;
wire [W-1:0] idata;
wire istop;
wire [W-1:0] odata;
wire ostop;

ibuf #(
  .W(W)
)
ibuf_inst(
  .clk(ap_clk),
  .reset(ap_rst),
  .idata(idata),
  .istop(istop),
  .cdata(odata),
  .cstop(ostop)
);
 
assign idata = {vld_in, data_in};
assign ack_in = ~istop;

assign vld_out = odata[W-1];
assign data_out = odata[W-2:0];
assign ostop = ~ack_out;

assign apdone_blk = ((ap_rst == 1'b0)&(ack_in == 1'b0));

endmodule //reverse

module regslice_both_w1 
#(parameter 
    DataWidth=32
)(
    input ap_clk ,
    input ap_rst,

    input data_in , 
    input vld_in , 
    output ack_in ,
    output data_out, 
    output vld_out,
    input ack_out,
    output apdone_blk
);
 
localparam W = 2;

wire [W-1:0] cdata;
wire cstop;
wire [W-1:0] idata;
wire istop;
wire [W-1:0] odata;
wire ostop;

reg [1:0] count;

ibuf #(
  .W(W)
)
ibuf_inst(
  .clk(ap_clk),
  .reset(ap_rst),
  .idata(idata),
  .istop(istop),
  .cdata(cdata),
  .cstop(cstop)
);
 
 
obuf #(
  .W(W)
)
obuf_inst(
  .clk(ap_clk),
  .reset(ap_rst),
  .cdata(cdata),
  .cstop(cstop),
  .odata(odata),
  .ostop(ostop)
);

assign idata = {vld_in, data_in};
assign ack_in = ~istop;

assign vld_out = odata[W-1];
assign data_out = odata[W-2:0];
assign ostop = ~ack_out;
// count, indicate how many data in the regslice.
// 00 - null
// 10 - 0
// 11 - 1
// 01 - 2
always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        count <= 2'd0;
    end else begin
        if ((((2'd2 == count) & (1'b0 == vld_in)) | ((2'd3 == count) & (1'b0 == vld_in) & (1'b1 == ack_out)))) begin
            count <= 2'd2;
        end else if ((((2'd1 == count) & (1'b0 == ack_out)) | ((2'd3 == count) & (1'b0 == ack_out) & (1'b1 == vld_in)))) begin
            count <= 2'd1;
        end else if (((~((1'b0 == vld_in) & (1'b1 == ack_out)) & ~((1'b0 == ack_out) & (1'b1 == vld_in)) & (2'd3 == count)) | ((2'd1 == count) & (1'b1 == ack_out)) | ((2'd2 == count) & (1'b1 == vld_in)))) begin
            count <= 2'd3;
        end else begin
            count <= 2'd2;
        end
    end
end

assign apdone_blk = ((count == 2'd3 && ack_out == 1'b0) | (count == 2'd1));

endmodule // both


module regslice_forward_w1 
#(parameter 
    DataWidth=1
)(
    input ap_clk ,
    input ap_rst,

    input data_in , 
    input vld_in , 
    output ack_in ,
    output data_out, 
    output vld_out,
    input ack_out,
    output apdone_blk
);
 
localparam W = 2;

wire [W-1:0] cdata;
wire cstop;
wire [W-1:0] idata;
wire istop;
wire [W-1:0] odata;
wire ostop;

obuf #(
  .W(W)
)
obuf_inst(
  .clk(ap_clk),
  .reset(ap_rst),
  .cdata(idata),
  .cstop(istop),
  .odata(odata),
  .ostop(ostop)
);

assign idata = {vld_in, data_in};
assign ack_in = ~istop;

assign vld_out = odata[W-1];
assign data_out = odata[W-2:0];
assign ostop = ~ack_out;

assign apdone_blk = ((ap_rst == 1'b0)&(1'b0 == ack_out)&(1'b1 == vld_out));

endmodule //forward


module regslice_reverse_w1 
#(parameter 
    DataWidth=1
)(
    input ap_clk ,
    input ap_rst,

    input data_in , 
    input vld_in , 
    output ack_in ,
    output data_out, 
    output vld_out,
    input ack_out,
    output apdone_blk
);
 
localparam W = 2;

wire [W-1:0] cdata;
wire cstop;
wire [W-1:0] idata;
wire istop;
wire [W-1:0] odata;
wire ostop;

ibuf #(
  .W(W)
)
ibuf_inst(
  .clk(ap_clk),
  .reset(ap_rst),
  .idata(idata),
  .istop(istop),
  .cdata(odata),
  .cstop(ostop)
);
 
assign idata = {vld_in, data_in};
assign ack_in = ~istop;

assign vld_out = odata[W-1];
assign data_out = odata[W-2:0];
assign ostop = ~ack_out;

assign apdone_blk = ((ap_rst == 1'b0)&(ack_in == 1'b0));

endmodule //reverse


module ibuf 
#(
    parameter W=32
)(
    input clk ,
    input reset,
    input [W-1:0] idata, 
    output istop ,
    output [W-1:0] cdata, 
    input cstop 
);
 
reg [W-1:0] ireg = {1'b0, {{W-1}{1'b0}}}; // Empty
 
assign istop = reset ? 1'b1 : ireg[W-1]; // Stop if buffering
assign cdata = istop ? ireg : idata ; // Send buffered
 
always @(posedge clk)
    if(reset)
        ireg <= {1'b0, {{W-1}{1'b0}}}; // Empty 
    else begin
        if (!cstop && ireg [W-1]) // Will core consume?
            ireg <= {1'b0, {{W-1}{1'b0}}}; // Yes: empty buffer
        else if ( cstop && !ireg[W-1]) // Core stop, empty?
            ireg <= idata; // Yes: load buffer
    end
 
endmodule

// Forward mode
module obuf 
#(
    parameter W=32
)(
    input clk ,
    input reset,
    input [W-1:0] cdata ,
    output cstop ,
    output reg [W-1:0] odata,
    input ostop 
);

// Stop the core when buffer full and output not ready
assign cstop = reset? 1'b1 : (odata[W-1] & ostop);
 
always @(posedge clk)
    if(reset)
        odata <= {1'b0, {{W-1}{1'b0}}};
    else
        if (!cstop) begin// Can we accept more data?
            odata <= cdata; // Yes: load the buffer
        end

endmodule

    
