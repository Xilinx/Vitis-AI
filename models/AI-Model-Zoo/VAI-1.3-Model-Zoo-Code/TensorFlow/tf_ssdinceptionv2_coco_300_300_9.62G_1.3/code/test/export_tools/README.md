How to export models of tensorflow/models/research/object_detection without the preprocess and postprocess parts?
  1. First you should back up your exporter.py in YourPath/tensorflow/models/research/object_detection
  2. Then copy exporter_without_pre_post_process.py to YourPath/tensorflow/models/research/object_detection/exporter.py
  3. Finally run frozon_without_pre_post_process.sh to export the inference model without the preprocess and postprocess
