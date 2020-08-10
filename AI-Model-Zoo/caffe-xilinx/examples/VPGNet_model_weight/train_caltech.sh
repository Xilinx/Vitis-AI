./../../build/tools/caffe train --solver=./solver_caltech.prototxt -gpu=0,1,2,3 2>&1 | tee ./output_VGG_GSTiling.log

