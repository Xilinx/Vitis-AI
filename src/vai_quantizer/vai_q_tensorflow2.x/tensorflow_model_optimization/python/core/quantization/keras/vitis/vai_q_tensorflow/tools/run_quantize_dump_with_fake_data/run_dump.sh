set -ex
# only support channel=3(?,H,W,3)
# only support one input_node
quantized_pb=./quantize_results/quantize_eval_model.pb
input_nodes=
output_nodes=
height=
width=
channel=
bash dump.sh \
  $quantized_pb \
  $input_nodes \
  $output_nodes \
  $height \
  $width \
  $channel \
