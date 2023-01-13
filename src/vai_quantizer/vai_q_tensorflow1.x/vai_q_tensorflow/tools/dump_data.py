"""
Dump weights and activations

Usage: python dump_data.py --input_pb_file=xxx.pb [flags]
Example: python dump_data.py --input_pb_file=./models/resnet_fuse_conv_bn.pb --input_names=input --output_names=resnet_v1_50/predictions/Reshape_1
"""
import pdb
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.tools.graph_transforms import TransformGraph as GTT
from progressbar import ProgressBar
# local modules
from ..examples import load_image

tf.load_op_library(os.path.join(os.path.dirname(__file__), "lib/fix_neuron.so"))

# inputs flags
tf.app.flags.DEFINE_string('input_pb_file', '../examples/quantize_results/vgg16/wa_quantized_model.pb', 'input pb file')
tf.app.flags.DEFINE_string('input_names', 'input', 'input node names')
tf.app.flags.DEFINE_string('output_names', 'vgg_16/Predictions/Reshape_1', 'output node names')
# preprocess flags
tf.app.flags.DEFINE_string('preprocess', 'resnet_preprocess', 'image preprocess module name, must contains function [load_image], see ../util/preprocess/ for examples')
tf.app.flags.DEFINE_integer('class_num', 1000, 'number of classes, 1000 for imagenet(1001 for Inception V1)')
# environment flags
tf.app.flags.DEFINE_string('output_dir', './dump_results/', 'output dir')
tf.app.flags.DEFINE_string('gpu', '0', 'gpus used for quantization, comma-separated')
# evaluation flags
tf.app.flags.DEFINE_string('image_list', '../dataset/imagenet_image/val1.txt', 'input evaluation image list')
tf.app.flags.DEFINE_string('image_dir', '../dataset/imagenet_image/val_resize_256/', 'input evaluation image dir')
tf.app.flags.DEFINE_integer('total_batch', 1, 'input evaluation total batch')
tf.app.flags.DEFINE_integer('batch_size', 1, 'input evaluation batch size')
FLAGS = tf.app.flags.FLAGS


def dump_activation(graph_def,
             input_node_name,
             output_node_name,
             batch_size, 
             total_batch,
             class_num,
             eval_image_dir,
             eval_image_list):
    """dump activation of classification network graph_def's accuracy, need evaluation dataset"""

    # Initialization
    text = open(eval_image_list)
    line = text.readlines()
    base = 0
    sum_acc = 0

    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def,name = '')


        input_x = sess.graph.get_tensor_by_name(input_node_name)
        output = sess.graph.get_tensor_by_name(output_node_name)
        input_y = tf.placeholder(tf.float32,shape = [None,class_num])
        prediction = tf.argmax(tf.reshape(output, [batch_size, class_num]),1)
        label_prediction = tf.argmax(input_y,1)
        correct_prediction = tf.equal(prediction,label_prediction)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

        tensors = []
        tensors_name = []
        for node in graph_def.node:
            print(node.name)
            tmp_tensor = sess.graph.get_tensor_by_name(node.name+":0")
            tensors.append(tmp_tensor)
            tensors_name.append(node.name+":0")
        print(tensors_name)
        tensors.append(input_x)
        tensors.append(output)
        tensors_name.append("input_x")
        tensors_name.append("output")

        print("Start Evaluation for {} Batches...".format(total_batch))
        progress = ProgressBar()
        for i in progress(range(0,total_batch)):
            batch_xs,batch_ys = load_image(base,line,batch_size,class_num,eval_image_dir)
            base +=  batch_size
            #  results = sess.run(node_names,{input_x:batch_xs,input_y:batch_ys})
            results = sess.run(tensors,{input_x:batch_xs,input_y:batch_ys})
            for name,res in zip(tensors_name,results):
                filename = os.path.join(FLAGS.output_dir, name.replace("/","_"))
                print("output: " , os.path.join(FLAGS.output_dir, name.replace("/","_")))
                res.tofile(filename+".bin")
                np.savetxt(filename+".txt", res, fmt="%s", delimiter=",")


def printUsageMessage():
    print(""" Usage: python dump_data.py command [flags]
          Example: 
          1.dump_weights:       python dump_data.py dump_weights --input_pb_file=./models/resnet_fuse_conv_bn.pb --input_names=input --output_names=resnet_v1_50/predictions/Reshape_1 
          2.dump_activation:    python dump_data.py dump_activation  --input_pb_file=./models/resnet_fuse_conv_bn.pb --input_names=input --output_names=resnet_v1_50/predictions/Reshape_1 
          """)


def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    os.environ["TF_CPP_MIN_VLOG_LEVEL"]='0'
    #  os.environ["TF_CPP_MIN_LOG_LEVEL"]='1'

    # Parse command
    if len(unused_argv) is not 2:
        printUsageMessage()
        raise ValueError("Must specify an explicit `command`")

    if FLAGS.input_pb_file is None or FLAGS.input_pb_file == "":
        raise ValueError("Must specify an explicit `input_pb_file`")
    if FLAGS.input_names is None or FLAGS.input_names == "":
        raise ValueError("Must specify an explicit `input_names`")
    if FLAGS.output_names is None or FLAGS.output_names == "":
        raise ValueError("Must specify an explicit `output_names`")

    if unused_argv[1] == 'dump_weights':
        dump_weights()
    elif unused_argv[1] == 'dump_activation':
        input_graph_def = tf.Graph().as_graph_def()
        input_graph_def.ParseFromString(gfile.FastGFile(FLAGS.input_pb_file, "rb").read())
        os.system('mkdir -p {}'.format(FLAGS.output_dir))
        dump_activation(input_graph_def,
                        FLAGS.input_names + ":0",
                        FLAGS.output_names + ":0",
                        FLAGS.batch_size,
                        FLAGS.total_batch,
                        FLAGS.class_num,
                        FLAGS.image_dir,
                        FLAGS.image_list)
    else:
        printUsageMessage()
        raise ValueError("Wrong `command`: " + unused_argv[1])


if __name__ ==  "__main__":
    tf.app.run()
