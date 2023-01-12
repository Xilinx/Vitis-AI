  # --graphdef ./quantize_results_0719/quantize_eval_model.pb  \
python -m tf2onnx.convert \
  --graphdef ./added_nodes.pb  \
  --output quantized.onnx \
  --inputs MobilenetV2/input:0 \
  --outputs MobilenetV2/Predictions/Reshape_1:0 \
  --opset 13 \
  --load_op_libraries /proj/rdi/staff/xieyi/anaconda3/envs/vai_q_tensorflow/lib/python3.6/site-packages/vai_q_tensorflow/gen_files/fix_neuron_op.so \
  --verbose \

