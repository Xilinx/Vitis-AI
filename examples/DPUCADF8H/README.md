### Example end-to-end tutorial for quantizing/compiling/running DPUCADF8H

The directories in this area are demonstrating step by step how to take trained models and deploy them on DPUCADF8H on Alveo U200/U250 cards.

You can find more models in the AI Model Zoo or run your own models using these examples as a starting point.

Note: 
- DPUCADF8H currently supports many classification and detection models.  Depthwise convolution as seen in Mobilenet and variants however is not supported.
- DPUCADF8H uses a default batchsize of 4. If the original model's batchsize is not 4, you can specify the batchsize using  `--options` argument with all vai\_c\_\* compilers. e.g. `--options '{"input_shape": "4,224,224,3"}'`.
