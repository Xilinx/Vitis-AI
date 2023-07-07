<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

## Vitis&trade; AI DPU IP and Reference Designs

The purpose of this page is to distribute DPU IP and reference designs.

Today, Xilinx&reg; DPU IPs are not incorporated into the standard Vivado&trade; IP catalog and instead, the DPU IP is released asynchronous to Vivado in two forms:

- The DPU IP is released as a reference design that is available to download from the links provided in the table below.  Users can start with the reference design and modify it to suit their requirements.
- The DPU is released as a separate IP download that can be incorporated into a new or existing design by the developer.  

The reference designs are fully functional and can be used as a template for IP integration and connectivity as well as Linux integration.

## Version and Compatibility

As the user must incorporate the IP into the Vivado IP catalog themselves, it is very important to understand that the designs and IP on this page are specific to Vitis AI v3.0 and were verified with Vivado and Vitis 2022.2.  If you are using a different version of Vitis or Vivado, please refer to [IP and Tool Version Compatibility](https://xilinx.github.io/Vitis-AI/docs/reference/version_compatibility.html) for additional information.

## Introduction

The table below associates currently available DPU IP with the supported target, and provides links to download the reference design and documentation.  For convenience, a separate IP repo is provided for users who do not wish to download the reference design.  The IP is thus included both in the reference design, but also is available as a separate download.  

Please refer to [DPU Nomenclature](https://xilinx.github.io/Vitis-AI/docs/reference/dpu_nomenclature.html) for detailed information on the capabilities and device targets for Xilinx DPUs.


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
    <td align="center">Programmable logic based DPU, targeting general purpose CNN inference with full support for the Vitis AI ModelZoo. Supports either the Vitis or Vivado flows on 16nm Zynq&reg; UltraScale+&trade; platforms.</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G_VAI_v3.0.tar.gz">Download</a></td>
    <td align="center"><a href="https://docs.xilinx.com/r/en-US/pg338-dpu">PG338</a></td>
   <td align="center"><a href="ref_design_docs/README_DPUCZDX8G.md">Link</a></td>
   <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G_ip_repo_VAI_v3.0.tar.gz">Get IP</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8G</td>
    <td align="center">VCK190</td>
    <td align="center">AIE centric DPU (requires some programmable logic), targeting general purpose CNN inference with full support for the Vitis AI ModelZoo. Supports the Vitis flow for 7nm Versal&trade; ACAPs.</td> 
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8G_VAI_v3.0.tar.gz">Download</a></td>
    <td align="center"><a href="https://docs.xilinx.com/r/en-US/pg389-dpucvdx8g">PG389</a></td>
   <td align="center"><a href="ref_design_docs/README_DPUCVDX8G.md">Link</a></td>
   <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8G_ip_repo_VAI_v3.0.tar.gz">Get IP</a></td>
  </tr>
  <tr>
    <td align="center">DPUCV2DX8G</td>
    <td align="center">VEK280</td>
    <td align="center">AIE-ML centric DPU (requires some programmable logic), targeting general purpose CNN inference with full support for the Vitis AI ModelZoo. Supports the Vitis flow for 7nm Versal AI Edge targets leveraging the AIE-ML architecture.</td> 
    <td align="center"><a href="https://www.xilinx.com/member/vitis-ai-vek280.html">Early Access</a></td>
    <td align="center"><b>---</b></td>
   <td align="center"><b>---</b></td>
   <td align="center"><b>---</b></td>
  </tr>
</tbody>
</table>

### Data Center IP

<table>
<thead>
  <tr>
    <th width="10%" align="center"><h3><b>IP Name</b></hr></th>
    <th width="5%" align="center"><h3><b>Supported Platforms</b></hr></th>
    <th width="50%" align="center"><h3><b>Description</b></hr></th>
    <th width="10%" align="center"><h3><b>Reference Design</b></hr></th>
    <th width="10%" align="center"><h3><b>Product Guide</b></hr></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">DPUCV2DX8G</td>
    <td align="center">V70</td>
    <td align="center">AIE-ML centric DPU (requires some programmable logic), targeting general purpose CNN inference with full support for the Vitis AI ModelZoo. Supports the Vitis flow for 7nm Versal AI Edge targets leveraging the AIE-ML architecture.</td> 
    <td align="center"><a href="https://www.xilinx.com/member/v70.html">Early Access</a></td>
    <td align="center"><b>---</b></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_2pe_miscdwc</td>
    <td rowspan="5" align="center">VCK5000</td>
    <td rowspan="5" align="center">High throughput CNN inference 7nm DPU for ACAP platforms. All computing engines are implemented with FPGA AIE cores.  Support Xilinx shell integration.</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_2pe_miscdwc_VAI_v3.0.tar.gz">Download</a></td>
    <td rowspan="5" align="center"><a href="https://docs.xilinx.com/r/en-US/pg403-dpucvdx8h">PG403</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_4pe_miscdwc</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_4pe_miscdwc_VAI_v3.0.tar.gz">Download</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_6pe_misc</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_6pe_misc_VAI_v3.0.tar.gz">Download</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_6pe_dwc</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_6pe_dwc_VAI_v3.0.tar.gz">Download</a></td>
  </tr>
  <tr>
    <td align="center">DPUCVDX8H_8pe_normal</td>
    <td align="center"><a href="https://www.xilinx.com/bin/public/openDownload?filename=DPUCVDX8H_8pe_normal_VAI_v3.0.tar.gz">Download</a></td>
  </tr>
</tbody>
</table>
