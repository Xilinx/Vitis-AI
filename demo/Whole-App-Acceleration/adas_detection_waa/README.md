## 1. Build & run for ZCU102
  #### Build
  ```
  cd ~/adas_detection_waa
  ./build.sh
  mkdir output #Will be written to the picture after processing
  ```
  #### Run adas_detection without waa
  ```
  ./adas_detection_waa /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 0
  ```
  #### Run adas_detection with waa
  ```
  env XILINX_XRT=/usr ./adas_detection_waa /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 1
   ```

## 2. Build & run for U50
  #### Build
  ```
  cd ./adas_detection_waa
  ./build.sh
  mkdir output #Will be written to the picture after processing
  ```
  #### Run adas_detection without waa
  ```
  ./adas_detection_waa /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 0
  ```
  #### Run adas_detection with waa
  ```
./adas_detection_waa /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel 1
   ```