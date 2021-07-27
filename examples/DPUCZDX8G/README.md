### Example end-to-end tutorial for searching a neural architecture based on Once-for-all networks and quantizing/compiling/running it on DPUCZDX8G

The directories in this area are demonstrating step by step how to search a neural architecture from Once-for-all/Resnet50 design space and deploy it on DPUCZDX8G on Zynq ZCU102/ZCU104 boards.

Once-for-all(ofa) is in the domain of Neural Architecture Search (NAS). It decouples training and searching process. Thus, searching takes very short time. The details can be found https://github.com/mit-han-lab/once-for-all.

Once-for-all currently has three design spaces: mobilenet_v3, proxyless (similiar to mobilnet_v2), and resnet50.

You can find the tutorial of ofa/resnet50 example in https://github.com/mit-han-lab/once-for-all/blob/master/tutorial/ofa_resnet50_example.ipynb



Note: DPUCZDX8G currently supports many classification and detection models. You can find more models in the AI Model Zoo and vitis/demo/Vitis-AI-Library/samples/ for DPUCZDX8G.