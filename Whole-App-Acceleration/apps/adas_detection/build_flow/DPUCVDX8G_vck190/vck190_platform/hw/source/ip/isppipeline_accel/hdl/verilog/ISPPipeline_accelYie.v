
`timescale 1 ns / 1 ps

  module ISPPipeline_accelYie_DSP48_5(a, b, p);
input signed [15 - 1 : 0] a;
input [10 - 1 : 0] b;
output signed [25 - 1 : 0] p;

assign p = $signed (a) * $signed ({1'b0, b});

endmodule
`timescale 1 ns / 1 ps
module ISPPipeline_accelYie(
    din0,
    din1,
    dout);

parameter ID = 32'd1;
parameter NUM_STAGE = 32'd1;
parameter din0_WIDTH = 32'd1;
parameter din1_WIDTH = 32'd1;
parameter dout_WIDTH = 32'd1;
input[din0_WIDTH - 1:0] din0;
input[din1_WIDTH - 1:0] din1;
output[dout_WIDTH - 1:0] dout;



ISPPipeline_accelYie_DSP48_5 ISPPipeline_accelYie_DSP48_5_U(
    .a( din0 ),
    .b( din1 ),
    .p( dout ));

endmodule

