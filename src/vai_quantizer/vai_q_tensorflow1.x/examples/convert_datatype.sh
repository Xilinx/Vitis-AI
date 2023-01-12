# 0 skip convert
# 1 convert to float16
# 2 convert to double
# 3 convert to bfloat16
# 4 convert to float32

DTYPE=3
vai_q_tensorflow quantize \
   --input_frozen_graph $FLOAT_MODEL \
   --input_nodes $Q_INPUT_NODE \
   --input_shapes ?,$INPUT_HEIGHT,$INPUT_WIDTH,$CHANNEL_NUM \
   --output_nodes $Q_OUTPUT_NODE \
   --output_dir $QUANTIZE_DIR \
   --convert_datatype $DTYPE \
