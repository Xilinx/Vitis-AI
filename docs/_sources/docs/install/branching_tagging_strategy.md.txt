<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

## Branching / Tagging Strategy

Each updated release of Vitis AI is pushed directly to [master](https://github.com/Xilinx/Vitis-AI/tree/master) on the day of release.  In addition, at that time, a tag is created for the repo, for example, see the tag for [v2.5](https://github.com/Xilinx/Vitis-AI/tree/v2.5).

Following release, the tagged version remains static, and additional inter-version updates are pushed to master.  Thus, master is always the latest release and will have the latest fixes and documentation.  The branch associated with a specific release (which will be "master" during the lifecycle of that release) will become a branch at the time of the next release.

Similarly, if you are using a previous version of Vitis AI, the branch associated with that previous revision will contain updates that were applied to the tagged release for that same version.  For instance, in the case of release 2.0, the [branch](https://github.com/Xilinx/Vitis-AI/tree/2.0) contains updates that the [tag](https://github.com/Xilinx/Vitis-AI/tree/v2.0) does not.

The diagram below depicts the overall workflow:

<div align="center">
  <img width="100%" height="100%" src="../reference/images/branching_strategy.PNG">
</div>

