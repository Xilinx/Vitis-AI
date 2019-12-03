"""
Resnet_v1_50 Evaluation Script
"""
import os
import argparse
import sys
import tensorflow as tf
from progressbar import ProgressBar
from input_fn import eval_input
import tensorflow.contrib.decent_q

FLAGS = None

def resnet_v1_50_eval(input_graph_def, input_node, output_node):
    """Evaluate classification network graph_def's accuracy, need evaluation dataset"""
    tf.import_graph_def(input_graph_def,name = '')

    # Get input tensors
    input_tensor = tf.get_default_graph().get_tensor_by_name(input_node+':0')
    input_labels = tf.placeholder(tf.float32,shape = [None,FLAGS.class_num])

    # Calculate accuracy
    output = tf.get_default_graph().get_tensor_by_name(output_node+':0')
    prediction = tf.argmax(tf.reshape(output, [FLAGS.batch_size, FLAGS.class_num]),1)
    label_prediction = tf.argmax(input_labels,1)
    correct_prediction = tf.equal(prediction,label_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

    # Start evaluation
    print("Start Evaluation for {} Batches...".format(FLAGS.eval_batches))
    with tf.Session() as sess:
        progress = ProgressBar()
        sum_acc = 0
        for iter in progress(range(0,FLAGS.eval_batches)):
            input_data = eval_input(iter, FLAGS.eval_image_dir, FLAGS.eval_image_list, FLAGS.class_num)
            images = input_data['input']
            labels = input_data['labels']
            feed_dict = {input_tensor: images, input_labels: labels}
            acc = sess.run(accuracy,feed_dict)
            sum_acc +=  acc
    final_accuracy = sum_acc/FLAGS.eval_batches
    print("Accuracy: {}".format(final_accuracy))
    return final_accuracy


def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    input_graph_def = tf.Graph().as_graph_def()
    input_graph_def.ParseFromString(tf.gfile.FastGFile(FLAGS.input_frozen_graph, "rb").read())
    resnet_v1_50_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_frozen_graph', type=str,
                        default='frozen_resnet_v1_50.pb',
                        help='frozen pb file.')
    parser.add_argument('--input_node', type=str,
                        default='input',
                        help='input node.')
    parser.add_argument('--output_node', type=str,
                        default='resnet_v1_50/predictions/Reshape_1',
                        help='output node.')
    parser.add_argument('--class_num', type=int,
                        default=1000,
                        help='number of classes.')   
    parser.add_argument('--eval_batches', type=int,
                        default=1000,
                        help='number of total batches for evaluation.')   
    parser.add_argument('--batch_size', type=int,
                        default=50,
                        help='number of batch size.')   
    parser.add_argument('--eval_image_dir', type=str,
                        default="",
                        help='evaluation image directory.')   
    parser.add_argument('--eval_image_list', type=str,
                        default="",
                        help='evaluation image list file.')   
    parser.add_argument('--gpu', type=str,
                        default='0',
                        help='gpu device id.')   
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
