set -ex
# only support one input_node


float_pb=
input_nodes=
output_nodes=
height=
width=
channel=
bash quantize.sh \
  $float_pb \
  $input_nodes \
  $output_nodes \
  $height \
  $width \
  $channel \
