set -ex
# only support one input_node


float_pb=../graph.pb
input_nodes=Placeholder
output_nodes=Identity_1
# output_nodes=Relu_1
height=224
width=224
channel=3
bash quantize.sh \
  $float_pb \
  $input_nodes \
  $output_nodes \
  $height \
  $width \
  $channel \
