
/*
 * Copyright (c) 2014 Xilinx, Inc.  All rights reserved.
 *
 * Xilinx, Inc.
 * XILINX IS PROVIDING THIS DESIGN, CODE, OR INFORMATION "AS IS" AS A
 * COURTESY TO YOU.  BY PROVIDING THIS DESIGN, CODE, OR INFORMATION AS
 * ONE POSSIBLE   IMPLEMENTATION OF THIS FEATURE, APPLICATION OR
 * STANDARD, XILINX IS MAKING NO REPRESENTATION THAT THIS IMPLEMENTATION
 * IS FREE FROM ANY CLAIMS OF INFRINGEMENT, AND YOU ARE RESPONSIBLE
 * FOR OBTAINING ANY RIGHTS YOU MAY REQUIRE FOR YOUR IMPLEMENTATION.
 * XILINX EXPRESSLY DISCLAIMS ANY WARRANTY WHATSOEVER WITH RESPECT TO
 * THE ADEQUACY OF THE IMPLEMENTATION, INCLUDING BUT NOT LIMITED TO
 * ANY WARRANTIES OR REPRESENTATIONS THAT THIS IMPLEMENTATION IS FREE
 * FROM CLAIMS OF INFRINGEMENT, IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE.
 *
 *
 * This file contains the top-level of the audio clock rate regenerator
 * used in the HDMI Audio Clock Regeneration model.
 *
 * MODIFICATION HISTORY:
 *
 * Ver   Who Date         Changes
 * ----- --- ----------   -----------------------------------------------
 * 1.00  RHe 2014/12/09   First release
 * 1.01  RHe 2015/01/15   Moved pulse_clkcross and data_clkcross to hdmi_acr_lib
 * 1.02  RHe 2015/04/14   Added audio reset
 * 1.03  RHe 2015/07/31   Updated CTS generation for HDMI 2.0 formats
 *
 *****************************************************************************/
 
`timescale 1 ps / 1 ps
 
module hdmi_acr_ctrl_top
(
  // AXI4-Lite bus (cpu control)
  input         axi_aclk,
  input         axi_aresetn,
  // - Write address
  input         axi_awvalid,
  output        axi_awready,
  input  [31:0] axi_awaddr,
  // - Write data
  input         axi_wvalid,
  output        axi_wready,
  input  [31:0] axi_wdata,
  input  [ 3:0] axi_wstrb,
  // - Write response
  output        axi_bvalid,
  input         axi_bready,
  output [ 1:0] axi_bresp,
  // - Read address   
  input         axi_arvalid,
  output        axi_arready,
  input  [31:0] axi_araddr,
  // - Read data/response
  output        axi_rvalid,
  input         axi_rready, 
  output [31:0] axi_rdata,
  output [ 1:0] axi_rresp,
   
  // Audio clock (512 * Fs)
  input         aud_clk,
   
  // HDMI TMDS clock 
  input         hdmi_clk,

  // Audio PLL Lock
  input         pll_lock_in,
  
  // Audio reset out
  output        aud_resetn_out,
  
  // Audio Clock Regeneration values
  input         aud_acr_valid_in,
  input  [19:0] aud_acr_cts_in,
  input  [19:0] aud_acr_n_in,
  output        aud_acr_valid_out,
  output [19:0] aud_acr_cts_out,
  output [19:0] aud_acr_n_out   
);

(* async_reg = "true" *)logic  [2:0] aud_rst_chain;
logic [1:0]  aud_pll_lock_sync;
logic [7:0]  aud_rst_cnt;
logic        aud_reset;
logic        aud_reset_out;

logic        aclk_enab_acr;
logic        aclk_acr_sel;
logic        aclk_aud_rst;
logic        aclk_tmdsclkratio;
logic [31:0] aclk_n_val;
logic [31:0] aclk_cts_val;

logic [ 1:0] aud_enab_acr_sync;
logic [ 1:0] aud_acr_sel_sync;
logic [ 1:0] aud_tmdsclkratio_sync;
logic [31:0] aud_n_capt;
logic [19:0] aud_n_val;
logic [31:0] aud_cts_capt_tmp;
logic [31:0] aud_cts_capt;
logic [19:0] aud_cts_val;

logic        aud_cke;
logic [1:0]  aud_rCKECounter = 'd0;
logic [31:0] aud_rCycleCntHigh;
integer      aud_rCycleCnt = 'd0;
logic        aud_rPulse = 1'b0;
logic        aud_acr_valid;

integer      hdmi_rCycleTimeCnt = 'd0;
logic [31:0] hdmi_rCTS_Val;
logic        hdmi_nCaptCTS;


hdmi_acr_ctrl_axi
#(
  .pVERSION_NR(32'hDEADBEEF)
)
HDMI_ACR_CTRL_AXI_INST
(
  // AXI4-Lite bus (cpu control)
  .S_AXI_ACLK    (axi_aclk),
  .S_AXI_ARESETN (axi_aresetn),
  // - Write address
  .S_AXI_AWVALID (axi_awvalid),
  .S_AXI_AWREADY (axi_awready),
  .S_AXI_AWADDR  (axi_awaddr),
  // - Write data
  .S_AXI_WVALID  (axi_wvalid),
  .S_AXI_WREADY  (axi_wready),
  .S_AXI_WDATA   (axi_wdata),
  .S_AXI_WSTRB   (axi_wstrb),
  // - Write response
  .S_AXI_BVALID  (axi_bvalid),
  .S_AXI_BREADY  (axi_bready),
  .S_AXI_BRESP   (axi_bresp),
  // - Read address   
  .S_AXI_ARVALID (axi_arvalid),
  .S_AXI_ARREADY (axi_arready),
  .S_AXI_ARADDR  (axi_araddr),
  // - Read data/response
  .S_AXI_RVALID  (axi_rvalid),
  .S_AXI_RREADY  (axi_rready), 
  .S_AXI_RDATA   (axi_rdata),
  .S_AXI_RRESP   (axi_rresp),
  .ENAB_ACR_OUT  (aclk_enab_acr),
  .ACR_SEL_OUT   (aclk_acr_sel),
  .AUD_RESET_OUT (aclk_aud_rst),
  .TMDSCLKRATIO_OUT (aclk_tmdsclkratio),
  .CTS_VAL_IN    (aclk_cts_val),
  .N_VAL_OUT     (aclk_n_val)
);

always_ff @(posedge aclk_aud_rst, posedge aud_clk)
begin
  if (aclk_aud_rst)
    aud_rst_chain <= {3{1'b1}};
  else
    aud_rst_chain <= aud_rst_chain << 1;
end

always_ff @(negedge pll_lock_in, posedge aud_clk)
begin
  // Reset
  if (!pll_lock_in)
  begin
    aud_rst_cnt  <= 0;
    aud_reset    <= 1'b1;
  end

  else
  begin
    // Reached maximum counter value
    if (&aud_rst_cnt)
      aud_reset <= 0;         // Release reset

    // Increment
    else
      aud_rst_cnt <= aud_rst_cnt + 'd1;
  end
end

// Divide the audio clock by 4
always_ff @(posedge aud_reset_out, posedge aud_clk)
begin
  if (aud_reset_out)
  begin
    aud_rCKECounter <= 0;
    aud_cke         <= 1'b0;
  end
  else
  begin
    aud_rCKECounter <= aud_rCKECounter + 1;

    if (aud_rCKECounter == 'd3)
      aud_cke       <= 1'b1;
    else
      aud_cke       <= 1'b0;  
  end
end

always_ff @(posedge aud_clk)
begin
  aud_tmdsclkratio_sync <= {aud_tmdsclkratio_sync[0], aclk_tmdsclkratio};
end
// For TMDS character rates larger than 340 Mcsc the TMDS clock oscillates at
// a 1/4 of the character rate. The Cycle High Count is thus set to 4 * N value.
//cunhua: dont change N period, but measure CTS and adjust CTS value only.
assign aud_cts_capt = (aud_tmdsclkratio_sync[1]) ? {aud_cts_capt_tmp[29:0], 2'b00} : aud_cts_capt_tmp;
// Generate a pulse every N'th cycle
assign aud_rCycleCntHigh = aud_n_capt;
always_ff @(posedge aud_reset_out, posedge aud_clk)
begin
  if (aud_reset_out)
  begin
    aud_rPulse        <= 1'b0;
    aud_rCycleCnt     <= 'd0;
  end
  else
  begin
    aud_rPulse        <= 1'b0;
    if (aud_cke)
    begin
      if (aud_rCycleCnt >= (aud_rCycleCntHigh-1))
      begin
        aud_rPulse    <= 1'b1;
        aud_rCycleCnt <= 'd0;
      end
      else
      begin
        aud_rCycleCnt <= aud_rCycleCnt + 1;
      end
    end
  end
end

always_ff @(posedge aud_clk)
begin
  aud_enab_acr_sync <= {aud_enab_acr_sync[0], aclk_enab_acr};
  aud_acr_sel_sync  <= {aud_acr_sel_sync[0], aclk_acr_sel};
  aud_acr_valid     <= 1'b0;
  
  if (aud_acr_sel_sync[1])
  begin
    aud_cts_val     <= aud_cts_capt;
    aud_n_val       <= aud_n_capt;
  end
  else
  begin
    aud_cts_val     <= aud_acr_cts_in;
    aud_n_val       <= aud_acr_n_in;
  end 
  
  if (aud_enab_acr_sync[1])
  begin
    if (aud_acr_sel_sync[1])
      aud_acr_valid <= aud_rPulse;
    else
      aud_acr_valid <= aud_acr_valid_in;
  end      
end

hdmi_acr_lib_pulse_clkcross
PULSE_CLKCROSS_INST
(
  .in_clk   (aud_clk),
  .in_pulse (aud_rPulse),
  .out_clk  (hdmi_clk),
  .out_pulse(hdmi_nCaptCTS)
);

hdmi_acr_lib_data_clkcross
#(
  .pDATA_WIDTH (32)
)
NVAL_CLKCROSS_INST
(
  .in_clk  (axi_aclk),
  .in_data (aclk_n_val),
  .out_clk (aud_clk),
  .out_data(aud_n_capt)
);

hdmi_acr_lib_data_clkcross
#(
  .pDATA_WIDTH (32)
)
CTS_CLKCROSS_ACLK_INST
(
  .in_clk  (hdmi_clk),
  .in_data (hdmi_rCTS_Val),
  .out_clk (axi_aclk),
  .out_data(aclk_cts_val)
);

hdmi_acr_lib_data_clkcross
#(
  .pDATA_WIDTH (32)
)
CTS_CLKCROSS_AUD_INST
(
  .in_clk  (hdmi_clk),
  .in_data (hdmi_rCTS_Val),
  .out_clk (aud_clk),
  .out_data(aud_cts_capt_tmp)
);

always_ff @(posedge hdmi_clk)
begin
  if (hdmi_nCaptCTS)
  begin
    hdmi_rCTS_Val      <= hdmi_rCycleTimeCnt;
    hdmi_rCycleTimeCnt <= 'd1;
  end
  else
  begin
    hdmi_rCycleTimeCnt <= hdmi_rCycleTimeCnt + 1;
  end
end

// Assign outputs
assign aud_acr_n_out     = aud_n_val;
assign aud_acr_cts_out   = aud_cts_val;
assign aud_acr_valid_out = aud_acr_valid;
assign aud_reset_out     = (aud_reset || aud_rst_chain[2]);
assign aud_resetn_out    = ~aud_reset_out;

endmodule //hdmi_acr_ctrl_top
