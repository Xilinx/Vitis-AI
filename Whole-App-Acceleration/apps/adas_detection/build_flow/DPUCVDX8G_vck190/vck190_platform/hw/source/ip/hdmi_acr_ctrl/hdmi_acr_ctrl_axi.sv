
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
 * This file contains the AXI register space of the audio clock rate 
 * regenerator used in the HDMI Audio Clock Regeneration model.
 *
 * MODIFICATION HISTORY:
 *
 * Ver   Who Date         Changes
 * ----- --- ----------   -----------------------------------------------
 * 1.00  RHe 2014/12/09   First release
 * 1.02  RHe 2015/04/14   Added audio reset
 * 1.03  RHe 2015/07/31   Added TMDS Clock Ratio bit for 
                          updated CTS generation (HDMI 2.0)
 * 1.04  YH  2017/11/06   ADDed Aud_Reset and TMDSCLKRatio to AXI4-lite read
 *
 *****************************************************************************/
 
`timescale 1ns / 1ps

module hdmi_acr_ctrl_axi
#(
  parameter pVERSION_NR = 32'hDEADBEEF
)
(
  // AXI4-Lite bus (cpu control)
  input             S_AXI_ACLK,
  input             S_AXI_ARESETN,
  // - Write address
  input             S_AXI_AWVALID,
  output reg        S_AXI_AWREADY,
  input      [ 7:0] S_AXI_AWADDR,
  // - Write data
  input             S_AXI_WVALID,
  output reg        S_AXI_WREADY,
  input      [31:0] S_AXI_WDATA,
  input      [ 3:0] S_AXI_WSTRB,
  // - Write response
  output reg        S_AXI_BVALID,
  input             S_AXI_BREADY,
  output reg [ 1:0] S_AXI_BRESP,
  // - Read address   
  input             S_AXI_ARVALID,
  output reg        S_AXI_ARREADY,
  input      [ 7:0] S_AXI_ARADDR,
  // - Read data/response
  output reg        S_AXI_RVALID,
  input             S_AXI_RREADY, 
  output reg [31:0] S_AXI_RDATA,
  output reg [ 1:0] S_AXI_RRESP,
  
  output            AUD_RESET_OUT,
  
  output            ENAB_ACR_OUT,
  output            ACR_SEL_OUT,
  output            TMDSCLKRATIO_OUT,
  input      [31:0] CTS_VAL_IN,
  output     [31:0] N_VAL_OUT  
);

// AXI Bus responses
localparam cAXI4_RESP_OKAY   = 2'b00; // Okay
localparam cAXI4_RESP_SLVERR = 2'b10; // Slave error

// Register map addresses
localparam cADDR_VER    = 'h00; // Version register
localparam cADDR_CTRL   = 'h04; // Control register
localparam cADDR_CTSVAL = 'h08; // CTS value register
localparam cADDR_NVAL   = 'h0C; // N value register

logic [31:0] rVersionNr;
logic [31:0] rNValue;
logic [31:0] rCTSValue;
logic        rEnab_ACR;
logic        rACR_Sel;
logic        rAud_Reset;
logic        rTMDSClkRatio;

// Capture the CTS value
always_ff @(negedge S_AXI_ARESETN, posedge S_AXI_ACLK)
begin
  if (!S_AXI_ARESETN)
  begin
    rCTSValue <= 32'h0;
  end
  else
  begin
    rCTSValue <= CTS_VAL_IN;
  end
end

////////////////////////////////////////////////////////
// Write channel

typedef enum { sWriteReset,
               sWriteAddr,
               sWriteData,
               sWriteResp
             } tStmAXI4L_Write;
             
tStmAXI4L_Write stmWrite;

logic [7:0] rWriteAddr;

// Statemachine for taking care of the write signals
always_ff @(negedge S_AXI_ARESETN, posedge S_AXI_ACLK)
begin
  if (!S_AXI_ARESETN)
  begin
    S_AXI_AWREADY       <= 1'b0;
    S_AXI_WREADY        <= 1'b0;
    S_AXI_BVALID        <= 1'b0;
    rWriteAddr          <=  'h0;
    stmWrite            <= sWriteReset;
  end
  else
  begin
    case (stmWrite) 
      sWriteReset :
      begin
        S_AXI_AWREADY   <= 1'b1;
        S_AXI_WREADY    <= 1'b0;
        S_AXI_BVALID    <= 1'b0;
        stmWrite        <= sWriteAddr;
      end
      
      sWriteAddr :
      begin
        S_AXI_AWREADY   <= 1'b1;
        if (S_AXI_AWVALID)
        begin
          S_AXI_AWREADY <= 1'b0;
          S_AXI_WREADY  <= 1'b1;
          rWriteAddr    <= S_AXI_AWADDR;
          stmWrite      <= sWriteData;
        end
      end
      
      sWriteData :
      begin
        S_AXI_WREADY    <= 1'b1;
        
        if (S_AXI_WVALID)
        begin
          S_AXI_WREADY  <= 1'b0;
          S_AXI_BVALID  <= 1'b1;
          stmWrite      <= sWriteResp;
        end
      end
      
      sWriteResp :
      begin
        S_AXI_BVALID    <= 1'b1;
        if (S_AXI_BREADY)
        begin
          S_AXI_BVALID  <= 1'b0;
          stmWrite      <= sWriteReset;
        end
      end 
      
      default :
        stmWrite        <= sWriteReset;
    endcase
  end
end

// Write address decoder
always_ff @(negedge S_AXI_ARESETN, posedge S_AXI_ACLK)
begin
  if (!S_AXI_ARESETN)
  begin
    S_AXI_BRESP       <= cAXI4_RESP_OKAY;
    rEnab_ACR         <= 1'b0;
    rACR_Sel          <= 1'b0;
    rAud_Reset        <= 1'b0;
    rVersionNr        <= pVERSION_NR;
    rNValue           <= 32'h0;
    rTMDSClkRatio     <= 1'b0;
  end
  else
  begin
    if (S_AXI_WREADY && S_AXI_WVALID)
    begin
      S_AXI_BRESP     <= cAXI4_RESP_OKAY;
      
      case (rWriteAddr)
        cADDR_VER :
        begin
          rVersionNr  <= S_AXI_WDATA;
        end
        
        cADDR_CTRL :
        begin
          rEnab_ACR   <= S_AXI_WDATA[0];
          rACR_Sel    <= S_AXI_WDATA[1];
          rAud_Reset  <= S_AXI_WDATA[2];
          rTMDSClkRatio <= S_AXI_WDATA[3];
        end
        
        cADDR_NVAL :
        begin
          rNValue     <= S_AXI_WDATA;
        end
        
        default :
          S_AXI_BRESP <= cAXI4_RESP_SLVERR;
      endcase
    end
  end
end

////////////////////////////////////////////////////////
// Read channel

typedef enum { sReadReset,
               sReadAddr,
               sDecodeAddr,
               sReadData
             } tStmAXI4L_Read;
             
tStmAXI4L_Read stmRead;

logic        ReadAddrNOK;
logic [ 7:0] rReadAddr;
logic [31:0] nReadData;

// Statemachine for taking care of the read signals
always_ff @(negedge S_AXI_ARESETN, posedge S_AXI_ACLK)
begin
  if (!S_AXI_ARESETN)
  begin
    S_AXI_ARREADY       <= 1'b0;    
    S_AXI_RRESP         <= cAXI4_RESP_OKAY;
    S_AXI_RVALID        <= 1'b0;
    S_AXI_RDATA         <=  'h0;
    rReadAddr           <=  'h0;
    stmRead             <= sReadReset;
  end
  else
  begin
    case (stmRead) 
      sReadReset :
      begin
        S_AXI_ARREADY   <= 1'b1;
        S_AXI_RRESP     <= cAXI4_RESP_OKAY;
        S_AXI_RVALID    <= 1'b0;
        S_AXI_RDATA     <=  'h0;
        rReadAddr       <=  'h0;
        stmRead         <= sReadAddr;
      end
      
      sReadAddr :
      begin
        S_AXI_ARREADY   <= 1'b1;
        if (S_AXI_ARVALID)
        begin
          S_AXI_ARREADY <= 1'b0;
          rReadAddr     <= S_AXI_ARADDR;
          stmRead       <= sDecodeAddr;
        end
      end
      
      sDecodeAddr :
      begin
        if (ReadAddrNOK)
          S_AXI_RRESP   <= cAXI4_RESP_SLVERR;
        else
          S_AXI_RRESP   <= cAXI4_RESP_OKAY;
          
        S_AXI_RDATA     <= nReadData;
        S_AXI_RVALID    <= 1'b1;
        stmRead         <= sReadData;
      end
      
      sReadData :
      begin
        S_AXI_RVALID    <= 1'b1;
        if (S_AXI_RREADY)
        begin
          S_AXI_RVALID  <= 1'b0;
          stmRead       <= sReadReset;
        end
      end
      
      default :
        stmRead         <= sReadReset;
    endcase
  end
end

// Read address decoder
always_comb
begin
  ReadAddrNOK      = 1'b0;
  nReadData        =  'h0;
  case (rReadAddr)
    cADDR_VER :
    begin
      nReadData    = rVersionNr;
    end
    
    cADDR_CTRL :
    begin
      nReadData[0] = rEnab_ACR;
      nReadData[1] = rACR_Sel;
	  nReadData[2] = rAud_Reset;
	  nReadData[3] = rTMDSClkRatio;
    end
    
    cADDR_NVAL :
    begin
      nReadData    = rNValue;
    end
    
    cADDR_CTSVAL :
    begin
      nReadData    = rCTSValue;
    end
    
    default : 
      ReadAddrNOK  = 1'b1;
  endcase  
end

// Assign the outputs
assign N_VAL_OUT    = rNValue;
assign ENAB_ACR_OUT = rEnab_ACR;
assign ACR_SEL_OUT  = rACR_Sel;
assign AUD_RESET_OUT = rAud_Reset;
assign TMDSCLKRATIO_OUT = rTMDSClkRatio;

endmodule
