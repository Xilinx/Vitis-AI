<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1>
   </td>
 </tr>

</table>

# Vitis AI Developer Machine Requirements

The following table lists Vitis AI developer workstation system requirements:  

<table>
<tr><th colspan="2">Component</th><th>Requirement</th></tr>
<tr><td colspan="2">GPU (Optional, but strongly recommended for quantization acceleration)</td><td>	NVIDIA GPU supporting CUDA 11.0 or higher, (eg: NVIDIA P100, V100, A100) </td></tr>
<tr><td colspan="2">CUDA Driver </td><td>NVIDIA-450 or higher for CUDA 11.0</td></tr>
<tr><td colspan="2">Docker Version</td><td>19.03 or higher, nvidia-docker2</td></tr>
<tr><td rowspan="3">Operating System</td><td>Ubuntu</td><td>18.04, 20.04</td></tr>
<tr><td>CentOS</td><td>7.8, 7.9, 8.1, 8.2</td></tr>
<tr><td>RHEL</td><td>8.3, 8.4</td></tr>
<tr><td rowspan="2" colspan="2">CPU</td><td>Intel i3/i5/i7/i9/Xeon 64-bit CPU</td></tr>
<tr><td>AMD EPYC 7F52 64-bit CPU</td></tr>
</table>

# Vitis AI Supported Board Targets

The following table lists target boards that are supported natively** by Vitis AI:

<table>
<tr><th colspan="2">Component</th><th>Requirement</th></tr>
<tr><td rowspan="4">Xilinx Target</td><td>Alveo</td><td>U50, U50LV, U200, U250, U280 cards</td></tr>
  <tr><td nowrap>Zynq UltraScale+ MPSoC</td><td>ZCU102 and ZCU104 Boards</td></tr>
  <tr><td>Versal</td><td>VCK190 and VCK5000 boards</td></tr>
  <tr><td>Kria</td><td>KV260</td></tr>
<table>
 
# Alveo Card System Requirements

Please refer to the "System Requirements" section of the relevant Alveo documentation.

<table>

<table>

 
 **Custom platform support can for Edge/Embedded devices may be enabled by the developer through Vitis and Vivado workflows
