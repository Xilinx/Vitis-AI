<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


### Alveo DPUCADF8H Tutorial

The purpose of this example is to demonstrate how the developer can deploy a trained model on the Alveo U200 or U250, leveraging the DPUCADF8H.  This is an end-to-end tutorial that includes the quantization, compilation and deployment workflow.

With this example, you can potentially run additional models from the [Model Zoo](../../model-zoo), or deploy your own custom model.

----------------------
📌Note: 
- The DPUCADF8H currently supports many classification and detection models.  Depthwise convolution is not supported, with the implication that MobileNets cannot be deployed on this DPU
- DPUCADF8H uses a default batch size of 4. If the original batch size for the deployed model was something other than 4, you can specify the batch size using the `--options` argument with all vai\_c\_\* compilers. e.g. `--options '{"input_shape": "4,224,224,3"}'`.
----------------------
