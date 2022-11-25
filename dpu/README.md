<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

## Vitis-AI DPU IP and Reference Designs

The purpose of this page is to distribute DPU IP and reference designs.

Today, Xilinx DPU IPs are not incorporated into the Vivado IP catalog.  The only source for theses IPs is via a set of reference designs that encapsulate both the IP as well as a complete platform reference design.  These reference designs are fully functional and should be used as a template for IP integration and connectivity as well as Linux integration.

## Version and Compatibility

The designs and IP on this page are specific to Vitis AI v2.5.  The IP and designs were verified with Vivado and Vitis 2022.1.  If you are using a different version of Vitis or Vivado, please refer to the [version compatibility document](version_compatibility.md) for additional information.

## Introduction

The table below associates currently available DPU IP with the supported target, and provides links to download the reference design and documentation.  For convenience, a separate IP repo is provided for users who do not wish to download the reference design (which includes the IP repo).  

Please refer to [DPU Nomenclature](../docs/reference/dpu_nomenclature.md) for detailed information on the capabilities and device targets for Xilinx DPUs.


### Edge IP


<table>
<thead>
  <tr>
    <th width="10%" align="center"><h3><b>IP Name</b></hr></th>
    <th width="5%" align="center"><h3><b>Supported Platforms</b></hr></th>
    <th width="65%" align="center"><h3><b>Description</b></hr></th>
    <th width="10%" align="center"><h3><b>Reference Design</b></hr></th>
    <th width="5%" align="center"><h3><b>Product Guide</b></hr></th>
    <th width="5%" align="center"><h3><b>Read Me</b></hr></th>
    <th width="5%" align="center"><h3><b>IP-only Download</b></hr></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">DPUCZDX8G</td>
    <td align="center">MPSoC / Kria K26</td>
    <td align="center">Programmable logic based DPU, targeting general purpose CNN inference with full support for the Vitis AI ModelZoo. Support either the Vitis or Vivado flows on 16nm ZU+ targets.</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G.tar.gz">Download</a></td>
    <td align="center"><a href="https://docs.xilinx.com/r/en-US/pg338-dpu">PG338</a></td>
   <td align="center"><a href="ref_design_docs/README_DPUCZDX8G.md">Link</a></td>
   <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G_ip_repo.tar.gz">Get IP</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8G</td>
    <td align="center">VCK190</td>
    <td align="center"> AIE-centric DPU (requires some programmable logic), targeting general purpose CNN inference with full support for the Vitis AI ModelZoo. Supports the Vitis flow for 7nm Versal targets.</td> 
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8G.tar.gz">Download</a></td>
    <td align="center"><a href="https://docs.xilinx.com/r/en-US/pg389-dpucvdx8g">PG389</a></td>
   <td align="center"><a href="ref_design_docs/README_DPUCVDX8G.md">Link</a></td>
   <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8G_ip_repo.tar.gz">Get IP</a></td>
  </tr>
</tbody>
</table>

### Cloud IP

<table>
<thead>
  <tr>
    <th width="10%" align="center"><h3><b>IP Name</b></hr></th>
    <th width="5%" align="center"><h3><b>Supported Platforms</b></hr></th>
    <th width="50%" align="center"><h3><b>Description</b></hr></th>
    <th width="10%" align="center"><h3><b>Reference Design</b></hr></th>
    <th width="10%" align="center"><h3><b>Product Guide</b></hr></th>
    <th width="5%" align="center"><h3><b>IP-only Download</b></hr></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">DPUCAHX8H</td>
    <td align="center">U50/U50lv/U280/U55c</td>
    <td align="center">High throughput CNN inference 16nm DPU. Optimized with high bandwidth memory. DPU core is fully built with FPGA programming logic. Support Xilinx shell integration.</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCAHX8H.tar.gz">Download</a></td>
    <td align="center"><a href="https://docs.xilinx.com/r/en-US/pg367-dpucahx8h">PG367</a></td>
   <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCAHX8H_ip_repo.tar.gz">Get IP</a></td>
  </tr>
  <tr>
    <td align="center">DPUCADF8H</td>
    <td align="center">U200/U250</td>
    <td align="center">High throughput CNN inference 16nm DPU. Optimized with DDR. DPU core is fully built with FPGA programming logic. Support Xilinx shell integration.</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCADF8H.tar.gz">Download</a></td>
    <td align="center"><a href="https://docs.xilinx.com/r/en-US/pg400-dpucadf8h">PG400</a></td>
   <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCADF8H_ip_repo.tar.gz">Get IP</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_2pe_miscdwc</td>
    <td rowspan="5" align="center">VCK5000</td>
    <td rowspan="5" align="center">High throughput CNN inference 7nm DPU for ACAP platforms. All computing engines are implemented with FPGA AIE cores.  Support Xilinx shell integration.</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_2pe_miscdwc.tar.gz">Download</a></td>
    <td rowspan="5" align="center"><a href="https://docs.xilinx.com/r/en-US/pg403-dpucvdx8h">PG403</a></td>
   <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_2pe_miscdwc_ip_repo.tar.gz">Get IP</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_4pe_miscdwc</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_4pe_miscdwc.tar.gz">Download</a></td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_4pe_miscdwc_ip_repo.tar.gz">Get IP</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_6pe_misc</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_6pe_misc.tar.gz">Download</a></td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_6pe_misc_ip_repo.tar.gz">Get IP</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_6pe_dwc</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_6pe_dwc.tar.gz">Download</a></td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_6pe_dwc_ip_repo.tar.gz">Get IP</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_8pe_normal</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_8pe_normal.tar.gz">Download</a></td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_8pe_normal_ip_repo.tar.gz">Get IP</a></td>
  </tr>
</tbody>
</table>
