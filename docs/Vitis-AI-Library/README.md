Vitis AI Library v1.0
======================

Please check <https://www.xilinx.com/products/design-tools/ai-inference/ai-developer-hub.html#edge> for updates.


Vitis AI Library directory structure introduction
--------------------------------------------------

```
vitis_ai_library
├── demo                         # Application demo, including classification, yolov3,
|   |                            # seg_and_pose_detect and segs_roadline_detect
│   ├── classification
│   ├── seg_and_pose_detect
│   ├── segs_and_roadline_detect
│   └── yolov3
├── libsrc                       # AI library open source code
│   ├── libdpbase                # dpbase library using Vitis unified APIs
│   ├── libdpclassification
│   ├── libdpfacedetect
│   ├── libdpfacelandmark
│   ├── libdpmultitask
│   ├── libdpopenpose
│   ├── libdpposedetect
│   ├── libdprefinedet
│   ├── libdpreid
│   ├── libdproadline
│   ├── libdpsegmentation
│   ├── libdpssd
│   ├── libdptfssd               # Tensorflow SSD library
│   ├── libdpyolov2
│   └── libdpyolov3
├── LICENSE
├── README.md                    # This README
└── samples                      # Model test samples, including jpeg test, video test, performance test
    ├── classification
    ├── facedetect
    ├── facelandmark
    ├── multitask
    ├── openpose
    ├── posedetect
    ├── refinedet
    ├── reid
    ├── roadline
    ├── segmentation
    ├── ssd
    ├── tfssd
    ├── yolov2
    └── yolov3
```
