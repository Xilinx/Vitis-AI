
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
 * This file contains the library for the audio clock rate regenerator
 * used in the HDMI Audio Clock Regeneration model.
 *
 * MODIFICATION HISTORY:
 *
 * Ver   Who Date         Changes
 * ----- --- ----------   -----------------------------------------------
 * 1.00  RHe 2015/01/15   First release
 *
 *****************************************************************************/
 
module hdmi_acr_lib_pulse_clkcross
(
  input  in_clk,
  input  in_pulse,
  input  out_clk,
  output out_pulse
);

reg rIn_PulseCap = 1'b0;
reg rIn_Toggle = 1'b0;

always @(posedge in_clk)
begin
  rIn_PulseCap <= in_pulse;
  
  if (in_pulse && !rIn_PulseCap)
    rIn_Toggle <= ~rIn_Toggle;
end

logic [2:0] rOut_Sync = 3'b000;
logic       rOut_Pulse = 1'b0;

always_ff @(posedge out_clk)
begin
  rOut_Sync  <= {rOut_Sync[1:0], rIn_Toggle};
  rOut_Pulse <= rOut_Sync[2] ^ rOut_Sync[1];
end

assign out_pulse = rOut_Pulse;

endmodule //hdmi_acr_lib_pulse_clkcross

module hdmi_acr_lib_data_clkcross #
(
  parameter pDATA_WIDTH = 32
)
(
  input                    in_clk,
  input  [pDATA_WIDTH-1:0] in_data,
  input                    out_clk,
  output [pDATA_WIDTH-1:0] out_data
);

logic                   rIn_DValid   = 1'b0;
logic [pDATA_WIDTH-1:0] rIn_Data;
logic [1:0]             rIn_ACK_Sync = 2'b00;

logic                   rOut_ACK         = 1'b0;
logic [1:0]             rOut_DValid_Sync = 2'b00;
logic [pDATA_WIDTH-1:0] rOut_Data;

always_ff @(posedge in_clk)
begin
  rIn_ACK_Sync <= {rIn_ACK_Sync[0], rOut_ACK};
  
  if (!rIn_ACK_Sync[1] && !rIn_DValid)
  begin
    rIn_Data   <= in_data;
    rIn_DValid <= 1'b1;
  end
  if (rIn_ACK_Sync[1] && rIn_DValid)
  begin
    rIn_DValid <= 1'b0;
  end  
end

always_ff @(posedge out_clk)
begin
  rOut_DValid_Sync <= {rOut_DValid_Sync[0], rIn_DValid};
  
  if (rOut_DValid_Sync[1] && !rOut_ACK)
  begin
    rOut_Data      <= rIn_Data;
    rOut_ACK       <= 1'b1;
  end
  if (!rOut_DValid_Sync[1] && rOut_ACK)
  begin
    rOut_ACK       <= 1'b0;
  end
end

assign out_data = rOut_Data;

endmodule //hdmi_acr_lib_data_clkcross
