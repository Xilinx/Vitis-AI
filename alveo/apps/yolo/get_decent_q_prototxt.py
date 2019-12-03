#!/usr/bin/env python
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

def get_train_prototxt_deephi(caffe_path, src_prototxt, train_prototxt, image_list, directory_path):
    sys.path.insert(0, caffe_path)
    import caffe
    from caffe import layers as L
    from caffe import params as P
    from caffe.proto import caffe_pb2
    import google.protobuf.text_format as tfmt

    net_shape = []
    net_parameter = caffe.proto.caffe_pb2.NetParameter()
    with open(src_prototxt, "r") as f:
        tfmt.Merge(f.read(), net_parameter)
    net_shape = net_parameter.layer[0].input_param.shape[0].dim

    print(net_shape[2], net_shape[3] )
    n = caffe.NetSpec()
    print(type(n))
    n.data = L.ImageData(top='label', include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), transform_param=dict(mirror=False, yolo_height=net_shape[2], yolo_width=net_shape[3]),
                                      image_data_param=dict(source=image_list,batch_size=1, shuffle=False,root_folder=directory_path))
    with open(train_prototxt, 'w') as f:
        f.write(str(n.to_proto()))
        print(n.to_proto())

    net_parameter = caffe.proto.caffe_pb2.NetParameter()
    with open(src_prototxt, "r") as f, open(train_prototxt, "a") as g:
        tfmt.Merge(f.read(), net_parameter)
        print("before\n", (net_parameter))
        #L = next(L for L in net_parameter.layer if L.name == 'data')

        print(net_parameter.layer[0])
        print(net_parameter.layer[0].input_param.shape[0].dim)

        #L.ImageData(include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), transform_param=dict(mirror=False, yolo_height=608, yolo_width=608),
        #                                  image_data_param=dict(source="/home/arunkuma/deephi/Image1.txt",batch_size=1, shuffle=False,root_folder="/wrk/acceleration/shareData/COCO_Dataset/val2014_dummy/"))

        del net_parameter.layer[0]
        print("after\n", (net_parameter))
        g.write(tfmt.MessageToString(net_parameter))


def main():
    print(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    get_train_prototxt_deephi(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

if __name__ == '__main__':
    main()
