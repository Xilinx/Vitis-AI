# **Quantization Example with vai_q_tensorflow2.x**
## **Preparation**
**Install the python dependencies**
```
pip install -r requirements.txt
```
## **Usage**
**Pof2s  Quantization Strategy Example**
```
python mnist_cnn_ptq.py
```

**Floating-point Scaling Quantization Strategy Example**
```
python mnist_cnn_ptq_fs.py
```

**GPU Quantization Strategy Example**
```
python mnist_cnn_ptq_gpu.py
```

**Unsigned Quantization Example**
```
python mnist_cnn_ptq_unsigned_quantize.py
```

**Quantize Model with XCompiler Example**
```
python mnist_cnn_quantize_with_xcompiler.py
```

**Inspect Example**
```
python mnist_cnn_inspect.py
```

**QAT Quantization Example**
```
python mnist_cnn_qat.py
```

**Candidate Layer Quantization Example**
```
python candidate_layer_quantize.py
```

**Custom Layer Quantization Example**
```
python mnist_cnn_ptq_custom_layer.py
```

**Model Data Type Conversion Example**
```
python mnist_cnn_convert_datatype.py
```

**Mixed Precision Example**
```
python mnist_cnn_ptq_mix_precision.py
```

**Quantization and Exporting to ONNX Format**
```
python export_onnx.py
```

**Dumping and Loading Quantization Strategy Configurations Example**
```
python mnist_cnn_ptq_config_quantize_strategy.py
```

**BFP Quantization Strategy Example**
```shell
$ cd fashion_mnist_bfp
$ bash run.sh
```

**BERT Model Quantization Example**
```shell
$ cd transformers_bert_qa
$ bash run_eval.sh # Evaluate float model
$ bash run_train.sh # Train float model
$ bash run_quantize.sh # For PTQ (Post Training Quantization)
$ bash run_qatrain.sh # For QAT (Quantization-Aware Training)
$ bash run_quantize_eval.sh # Evaluate quantized model
```
