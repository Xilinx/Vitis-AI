export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
conda activate rnn_tf_2.0
sudo cp backup/hdf5_format.py /opt/vitis_ai/conda/envs/rnn_tf_2.0/lib/python3.7/site-packages/tensorflow_core/python/keras/saving/hdf5_format.py
