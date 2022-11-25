<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

### NAS, Once-for-All Examples

The purpose of this folder is to provide the user with an example end-to-end tutorial that illustrates how to implement a form of NAS known as Once-for-All (OFA). The tutorial provides the user with the workflow used to search for a neural network architecture leveraging on Once-for-All techniques.  The tutorial includes the process of quantization, compilation and deployment on the Zynq Ultrascale+ DPUCZDX8G.  Step-by-step the tutorial walks the developer through this process, leveraging OFA_ResNet50, and supports deployment of the final model on either the ZCU102 or ZCU104 evaluation boards.

Compared with traditional NAS techniques, OFA decouples the training and searching processes, with the result that the search algorithm takes a very short time.  Additional details of this technique can be found in the paper authored by [Cai et al](https://arxiv.org/pdf/1908.09791.pdf).  The associated repository maintained by that team can be found [here](https://github.com/mit-han-lab/once-for-all)

OFA currently supports three design spaces: MobileNetv3, Proxyless (similiar to MobileNetv2), and ResNet50.

You can find an additional tutorial OFA/ResNet50 [tutorial](https://github.com/mit-han-lab/once-for-all/blob/master/tutorial/ofa_resnet50_example.ipynb) on the MIT Github.

