#ls object_detection/protos/ > file

#cat file | while read line
for i in  anchor_generator  grid_anchor_generator  mean_stddev_box_coder  region_similarity_calculator  argmax_matcher  hyperparams  model  square_box_coder  bipartite_matcher  eval  image_resizer  multiscale_anchor_generator  ssd_anchor_generator  box_coder  faster_rcnn_box_coder  input_reader  optimizer  ssd  box_predictor  faster_rcnn  keypoint_box_coder  pipeline  string_int_label_map  calibration  flexible_grid_anchor_generator  losses  post_processing  graph_rewriter  matcher  preprocessor  train 
do 
echo "$i"
protoc --cpp_out=./ --proto_path=/usr/include/ object_detection/protos/${i}.proto
done
