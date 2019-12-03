# MNIST LeNet Tutorial
This Tutorial is meant to be ran inside of ML Suite's Caffe Container  
Every step is meant to be performed inside of `/opt/ml-suite/examples/caffe/mnist`
## Get MNIST Dataset
The MNIST dataset is grey scale 28x28 images of handwritten digits 0-9.  
Use the provided script to download the mnist dataset.  

```
# If not already there:
$ cd /opt/ml-suite/examples/caffe/mnist

$ ./get_mnist.sh
```

## Convert the MNIST Dataset
MNIST is distributed as gzipped binary files.  
Caffe requires that training data be either JPEG, or in an "LMDB" database file.  
Lets not worry about the conversion process.  
The developers of caffe gave us C++ code, and a script many moons ago to convert the MNIST dataset to an LMDB file.  
Use the provided script to convert the mnist dataset into an LMDB file.  
```
$ ./create_mnist_lmdb.sh
```

## Download the model definition
We are about to train a famous light weight model called LeNet.  
Input -> Conv1 -> Pool1 -> Conv2 -> Pool2 -> InnerProduct -> ReLU -> InnerProduct -> ReLU -> Softmax  
To do so, we could write the caffe prototxt, ourselves, but instead lets grab it from the web.  
Also, lets grab the solver prototxt which communicates to caffe our hyperparameters for training.  
For more background info, checkout info.md  and also Google is your friend.  
```
$ wget https://raw.githubusercontent.com/BVLC/caffe/master/examples/mnist/lenet_solver.prototxt
$ wget https://raw.githubusercontent.com/BVLC/caffe/master/examples/mnist/lenet_train_test.prototxt
```

## Edit the prototxts to correct path pointers
Caffe embeds a lot of information in its prototxt files, and when grabbing an open source model, typically you will have to massage.  
Most commonly, the paths pointing to training images on disk will be invalid, because everyone is free to put images where they want to.  
Do the following:  
1. Edit `lenet_solver.prototxt`  
  a. change `net:` at line 2  
    `net: "./lenet_train_test.prototxt"`  
  b. change `snapshot_prefix:` at line 23  
    `snapshot_prefix: "./lenet"`  
  c. change `solver_mode:` at line 25  
    `solver_mode: CPU`  
2. Edit `lenet_train_test.prototxt`  
  a. change `name:` at line 3  
    `name: "data"`  
  b. change `name:` at line 20  
    `name: "data"`  
  c. change `source:` at line 14  
    `source: "./mnist_train_lmdb"`  
  d. change `source:` at line 31  
    `source: "./mnist_test_lmdb"`  

## Train the model!!
Training this model on the CPU should take just a few minutes.  
Caffe prints the loss, and test every so often.  
Running this step will produce a trained model with weights, and the file will be called `lenet_iter_10000.caffemodel`  
Note that in this method we are running caffe as an executable (Compiled C++ code), whereas down the road, we will start using the convienient python wrappers for targetting the FPGA. "pycaffe".  
```
$ $CAFFE_ROOT/build/tools/caffe train --solver=./lenet_solver.prototxt
```

## Prepare the model for FPGA Inference
In this directory, we have provided you with a bunch of code.  
run.py is boiler plate code to help you quickly take a trained caffe model, and run the ml-suite tools to get ready for inference  
We will use the --prepare switch to make this quick and easy, but please do look at run.py and try to understand whats happening.  
Quantization, Compilation, and Subgraph Cutting take place during this step.  
```
$ ./run.py --prototxt lenet_train_test.prototxt --caffemodel lenet_iter_10000.caffemodel --prepare
```

## Validate the accuracy of the model on the FPGA
Its good practice to make sure that the compiled model can acheive good accuracy when targeting the FPGA.  
We are running with INT8 preciscion versus FP32, so we expect some miniscule accuracy degredation.  
```
$ ./run.py --validate --numBatches 10
``` 
You should see accuracy around 99%.  
You can quickly compare with the CPU by doing:  
```
$ ./run.py --prototxt lenet_train_test.prototxt --caffemodel lenet_iter_10000.caffemodel --validate_cpu --numBatches 10
```

## Lets examine a test image
As an easy way to visualize the test images, lets install a JPEG to ascii utility

```
$ sudo apt-get update && sudo apt-get install jp2a 
```
Now, lets take a look at an individual test image:
```
$ jp2a test_images/img_200.jpg

# You should see something like this:


                                           .,,:lllc::'...
                                         ..:xxKWWWNXXo....
                                     ...;ook000000000kdddl;;....
                                    .:::kWWWWW0dddxkkKWWW0xx:....
                                 .::okkkKNNOoo:,,,,;;lkkkkOOxlll,..
                             .;;;xNNXXXXxcc'          ...,cckXXXkll'
                           ..cOOOXMMXOOO;                 ..lNNNX00:
                          'ccxXXXXXXx;;;.                   'lllx00c
                          c00XWWWKOO;                           :KKl...
                      .,,,dNNKkkko;;.                       .;;;dXXo....
                      ,xxxKMMOccc.                       .,,oKKKKKKl...
                      ;OOOXMMk:::.                    ...:xxKMMMKkk,
                      'ooo0WW0ddd:......      .....,,cdddkKKNMMMk::.
                      .:::kWWX000o;;'.........,:::ldd0WWWMMMMMMMo..
                      ....lKKXXXX0kkxdddddddddxxxxkOOXWWWMMMXOOO:
                          ;kkKMMMMWWMMMMMMMMMMWNNNNNNNWWWMMMOccc.
                          .,,lxxxdddxOOOOOOkxxdooolcckXXXNWWx,,,.
                              ...........''...    .''dNNNKOO:
                                                  'llOWWWOll'
                                                  ;OOKNNNd''.
                                                  lWWNXXXl
                                              .'''dWWKkkk;
                                              ,xxx0WWd'''.
                                              cKKKXNNl
                                           ...dNNN0kk;
                                           .::kWWWOcc.
                                           ,dd0MMMk;;.
                                           :KKXWWWo..
                                           :KKXNNNl
                                         ..oNN0xxx;
                                        .;;kWWk;;;.
                                        .::kMMk;;;.
                                        .ccOMMO:::.
                                        .;;dKKd,,,.
                                         ...;;....


```

## Run this test image through our network on the FPGA
We are ready to run a single inference, but first we need to take care of something that caffe ignores...  
Labels!
Our boiler plate code is set up to use a labels file called `mnist_words.txt`.

Create a file in this directory called `mnist_words.txt`, and make sure it has the following content:
```
zero
one
two
three
four
five
six
seven
eight
nine
```
As you can see this is a label for each of the ten output classes.

Finally, you can run a single inference on the FPGA like this:

```
./run.py --image test_images/img_200.jpg
```
