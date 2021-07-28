<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1>
   </td>
 </tr>

</table>

# System Requirements

The following table lists system requirements for running docker containers as well as Alveo cards.  

<table>
<tr><th colspan="2">Component</th><th>Requirement</th></tr>
<tr><td rowspan="4">FPGA</td><td>Alveo</td><td>U50, U50LV, U200, U250, U280 cards</td></tr>
  <tr><td nowrap>Zynq UltraScale+ MPSoc</td><td>ZCU102 and ZCU104 Boards</td></tr>
  <tr><td>Versal</td><td>ACAP VCK190 and VCK5000 boards</td></tr>
  <tr><td>KV260</td><td></td></tr>
<tr><td colspan="2">Motherboard</td><td>PCI Express 3.0-compliant x16 with one or dual slot</td></tr>
<tr><td colspan="2">System Power Supply</td><td>225W</td></tr>
<tr><td rowspan="3">Operating System</td><td>Ubuntu</td><td>16.04, 18.04, 20.04</td></tr>
  <tr><td>CentOS</td><td>7.6, 7.7, 7.8, 8.1</td></tr>
  <tr><td>RHEL</td><td>7.6, 7.7, 7.8, 8.1</td></tr>
<tr><td rowspan="2" colspan="2">CPU</td><td>Intel i3/i5/i7/i9/Xeon 64-bit CPU</td></tr>
  <tr><td>AMD EPYC 7F52 64-bit CPU</td></tr>
<tr><td colspan="2">GPU (Optional to accelerate quantization)</td><td>	NVIDIA GPU supports CUDA 9.0 or higher, like NVIDIA P100, V100</td></tr>
<tr><td colspan="2">CUDA Driver (Optional to accelerate quantization)</td><td>Driver compatible to CUDA version, NVIDIA-384 or higher for CUDA 9.0, NVIDIA-410 or higher for CUDA 10.0</td></tr>
<tr><td colspan="2">Docker Version</td><td>19.03 or higher</td></tr>
</table>
