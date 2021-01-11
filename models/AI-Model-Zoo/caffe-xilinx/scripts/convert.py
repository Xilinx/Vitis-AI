
'''
Copyright (c) 2015 Preferred Infrastructure, Inc.
Copyright (c) 2015 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

from __future__ import print_function
import sys
sys.path.append('caffe/python')
import caffe
import numpy as np
from collections import OrderedDict
from six import moves

DEBUG = True
format_type = True
def parse_cfg(cfgfile):
    def erase_comment(line):
        line = line.split('#')[0]
        return line
    blocks = []
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = OrderedDict()
            block['type'] = line.lstrip('[').rstrip(']')
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            line = erase_comment(line)
            key,value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks
try:
    import caffe.proto.caffe_pb2 as caffe_pb2
except:
    try:
        import caffe_pb2
    except:
        print ('caffe_pb2.py not found. Try:')
        print ('  protoc caffe.proto --python_out=.')
        exit()

def parse_caffemodel(caffemodel):
    model = caffe_pb2.NetParameter()
    print ('Loading caffemodel: ', caffemodel)
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())

    return model

def parse_prototxt(protofile):
    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_block(fp):
        block = OrderedDict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                line = line.split('#')[0]
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if key in block:
                    if type(block[key]) == list:
                        block[key].append(value)
                    else:
                        block[key] = [block[key], value]
                else:
                    block[key] = value
            elif ltype == 1: # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
            line = line.split('#')[0]
        return block

    fp = open(protofile, 'r')
    props = OrderedDict()
    layers = []
    line = fp.readline()
    while line != '':
        line = line.strip().split('#')[0]
        if line == '':
            line = fp.readline()
            continue
        ltype = line_type(line)
        if ltype == 0: # key: value
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            if key in props:
               if type(props[key]) == list:
                   props[key].append(value)
               else:
                   props[key] = [props[key], value]
            else:
                props[key] = value
        elif ltype == 1: # blockname {
            key = line.split('{')[0].strip()
            if key == 'layer':
                layer = parse_block(fp)
                layers.append(layer)
            else:
                props[key] = parse_block(fp)
        line = fp.readline()

    if len(layers) > 0:
        net_info = OrderedDict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info
    else:
        return props

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def print_prototxt(net_info):
    # whether add double quote
    def format_value(value):
        if is_number(value):
            return value
        elif value == 'true' or value == 'false' or value == 'MAX' or value == 'SUM' or value == 'AVE':
            return value
        else:
            return '\"%s\"' % value

    def print_block(block_info, prefix, indent):
        blanks = ''.join([' ']*indent)
        print('%s%s {' % (blanks, prefix))
        for key,value in block_info.items():
            if type(value) == OrderedDict:
                print_block(value, key, indent+4)
            elif type(value) == list:
                for v in value:
                    print('%s    %s: %s' % (blanks, key, format_value(v)))
            else:
                print('%s    %s: %s' % (blanks, key, format_value(value)))
        print('%s}' % blanks)

    props = net_info['props']
    layers = net_info['layers']
    print('name: \"%s\"' % props['name'])
    print('input: \"%s\"' % props['input'])
    print('input_dim: %s' % props['input_dim'][0])
    print('input_dim: %s' % props['input_dim'][1])
    print('input_dim: %s' % props['input_dim'][2])
    print('input_dim: %s' % props['input_dim'][3])
    print('')
    for layer in layers:
        print_block(layer, 'layer', 0)

def save_prototxt(net_info, protofile, region=True):
    fp = open(protofile, 'w')
    def format_value(value):
        if is_number(value):
            return value
        elif value == 'true' or value == 'false' or value == 'MAX' or value == 'SUM' or value == 'AVE':
            return value
        else:
            return '\"%s\"' % value

    def print_block(block_info, prefix, indent):
        blanks = ''.join([' ']*indent)
        print ( '%s%s {' % (blanks, prefix), file=fp)
        for key,value in block_info.items():
            if type(value) == OrderedDict:
                print_block(value, key, indent+4)
            elif type(value) == list:
                for v in value:
                    print ('%s    %s: %s' % (blanks, key, format_value(v)),file=fp)
            else:
                print ('%s    %s: %s' % (blanks, key, format_value(value)), file=fp)
        print ('%s}' % blanks, file=fp)

    props = net_info['props']
    layers = net_info['layers']
    print ('name: \"%s\"' % props['name'], file=fp)
    print ('input: \"%s\"' % props['input'], file=fp)
    print ('input_dim: %s' % props['input_dim'][0], file=fp)
    print ('input_dim: %s' % props['input_dim'][1], file=fp)
    print ('input_dim: %s' % props['input_dim'][2], file=fp)
    print ('input_dim: %s' % props['input_dim'][3], file=fp)
    print ('',file=fp)
    for layer in layers:
        if layer['type'] != 'Region' or region == True:
            print_block(layer, 'layer', 0)
    fp.close()
def darknet2caffe(cfgfile, weightfile, protofile, caffemodel):
    net_info = cfg2prototxt(cfgfile)
    save_prototxt(net_info , protofile, region=False)

    net = caffe.Net(protofile, caffe.TEST)
    params = net.params

    blocks = parse_cfg(cfgfile)
    fp = open(weightfile, 'rb')
    header = np.fromfile(fp, count=5, dtype=np.int32)
    format_type = True if header[2] < 0 else False
    buf = np.fromfile(fp, dtype = np.float32)
    fp.close()
    layers = []
    layer_id =0 
    start = 0
    for block in blocks:
        if start >= buf.size:
            break

        if block['type'] == 'net':
            continue
        elif block['type'] == 'convolutional':
            batch_normalize = int(block['batch_normalize'])
            if 'name' in block:
                conv_layer_name = block['name']
                bn_layer_name = '%s-bn' % block['name']
            else:
                conv_layer_name = 'layer%d-conv' % layer_id
                bn_layer_name = 'layer%d-bn' % layer_id

            if batch_normalize:
                start = load_conv_bn2caffe(buf, start, params[conv_layer_name], params[bn_layer_name],format_type)
            else:
                start = load_conv2caffe(buf,start, params[conv_layer_name])
            layer_id = layer_id+1
        elif block['type'] == 'connected':
            if 'name' in block:
                fc_layer_name = block['name']
            else:
                fc_layer_name = 'layer%d-fc' % layer_id
            start = load_fc2caffe(buf, start, params[fc_layer_name])
            layer_id = layer_id+1
        elif block['type'] == 'maxpool':
            layer_id = layer_id+1
        elif block['type'] == 'avgpool':
            layer_id = layer_id+1
        elif block['type'] == 'region':
            layer_id = layer_id + 1
        elif block['type'] == 'reorg':
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            layer_id = layer_id + 1
        elif block['type'] == 'shortcut':
            layer_id = layer_id + 1
        elif block['type'] == 'yolo':
            layer_id = layer_id + 1
        elif block['type'] == 'softmax':
            layer_id = layer_id + 1
        elif block['type'] == 'upsample':
            layer_id = layer_id + 1
        elif block['type'] == 'cost':
            layer_id = layer_id + 1
        else:
            print('unknow layer type %s ' % block['type'])
            layer_id = layer_id + 1
    print('save prototxt to %s' % protofile)
    print(start,buf.size)
    save_prototxt(net_info , protofile, region=True)
    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel, 1 if format_type else 0 )

def load_conv2caffe(buf, start, conv_param):
    weight = conv_param[0].data
    bias = conv_param[1].data
    conv_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    conv_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start

def load_fc2caffe(buf, start, fc_param):
    weight = fc_param[0].data
    bias = fc_param[1].data
    fc_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    fc_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start


def load_conv_bn2caffe(buf, start, conv_param, bn_param, format_type):
    conv_weight = conv_param[0].data
    scale_weight = bn_param[0].data
    scale_bias = bn_param[1].data
    running_mean = bn_param[2].data
    running_var = bn_param[3].data
    bn_param[1].data[...] = np.reshape(buf[start:start+scale_bias.size], scale_bias.shape); start = start + scale_bias.size
    bn_param[0].data[...] = np.reshape(buf[start:start+scale_weight.size], scale_weight.shape); start = start + scale_weight.size
    bn_param[2].data[...] = np.reshape(buf[start:start+running_mean.size], running_mean.shape); start = start + running_mean.size
    bn_param[3].data[...] = np.reshape(buf[start:start+running_var.size], running_var.shape); start = start + running_var.size
    if format_type:
        bn_param[4].data[...] = np.array(buf[start:start + 1]); start = start + 1
    else:
        bn_param[4].data[...] = np.array([1.0])
    conv_param[0].data[...] = np.reshape(buf[start:start+conv_weight.size], conv_weight.shape); start = start + conv_weight.size
    return start

def cfg2prototxt(cfgfile):
    blocks = parse_cfg(cfgfile)

    layers = []
    props = OrderedDict() 
    bottom = 'data'
    layer_id = 0
    topnames = dict()
    for bidx in moves.range(len(blocks)):
        block = blocks[bidx]
        if block['type'] == 'net':
            props['name'] = 'Darkent2Caffe'
            props['input'] = 'data'
            props['input_dim'] = ['1']
            props['input_dim'].append(block['channels'])
            props['input_dim'].append(block['height'])
            props['input_dim'].append(block['width'])
            continue
        elif block['type'] == 'convolutional':
            conv_layer = OrderedDict()
            conv_layer['bottom'] = bottom
            if 'name' in block:
                conv_layer['top'] = block['name']
                conv_layer['name'] = block['name']
            else:
                conv_layer['top'] = 'layer%d-conv' % layer_id
                conv_layer['name'] = 'layer%d-conv' % layer_id
            conv_layer['type'] = 'Convolution'
            convolution_param = OrderedDict()
            convolution_param['num_output'] = block['filters']
            convolution_param['kernel_size'] = block['size']
            if block['pad'] == '1':
                convolution_param['pad'] = str(int(convolution_param['kernel_size'])//2)
            convolution_param['stride'] = block['stride']
            if block['batch_normalize'] == '1':
                convolution_param['bias_term'] = 'false'
            else:
                convolution_param['bias_term'] = 'true'
            conv_layer['convolution_param'] = convolution_param
            layers.append(conv_layer)
            bottom = conv_layer['top']

            if block['batch_normalize'] == '1':
                bn_layer = OrderedDict()
                bn_layer['bottom'] = bottom
                bn_layer['top'] = bottom
                if 'name' in block:
                    bn_layer['name'] = '%s-bn' % block['name']
                else:
                    bn_layer['name'] = 'layer%d-bn' % layer_id
                bn_layer['type'] = 'BatchNorm'
                batch_norm_param = OrderedDict()
                batch_norm_param['use_global_stats'] = 'true'
                bn_layer['batch_norm_param'] = batch_norm_param
                layers.append(bn_layer)

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if 'name' in block:
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'maxpool':
            max_layer = OrderedDict()
            max_layer['bottom'] = bottom
            if 'name' in block:
                max_layer['top'] = block['name']
                max_layer['name'] = block['name']
            else:
                max_layer['top'] = 'layer%d-maxpool' % layer_id
                max_layer['name'] = 'layer%d-maxpool' % layer_id
            max_layer['type'] = 'Pooling'
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = block['size']
            pooling_param['stride'] = block['stride']
            pooling_param['pool'] = 'MAX'
            if 'pad' in block and int(block['pad']) == 1:
                pooling_param['pad'] = str((int(block['size'])-1)//2)
            elif int(block['stride']) == 1:
                pooling_param['pad'] = str((int(block['size'])-1)//2)
            max_layer['pooling_param'] = pooling_param
            print (max_layer)
            layers.append(max_layer)
            bottom = max_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'avgpool':
            avg_layer = OrderedDict()
            avg_layer['bottom'] = bottom
            if 'name' in block:
                avg_layer['top'] = block['name']
                avg_layer['name'] = block['name']
            else:
                avg_layer['top'] = 'layer%d-avgpool' % layer_id
                avg_layer['name'] = 'layer%d-avgpool' % layer_id
            avg_layer['type'] = 'Pooling'
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = 7
            pooling_param['stride'] = 1
            pooling_param['pool'] = 'AVE'
            avg_layer['pooling_param'] = pooling_param
            layers.append(avg_layer)
            bottom = avg_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            if DEBUG:
                print("block:%s" % block)       
            from_layers = block['layers'].split(',')
            bottom_tmp = []
            if len(from_layers) == 1:
                if int(from_layers[0]) < 0:
                    prev_layer_id = layer_id + int(from_layers[0])
                else:
                    prev_layer_id = int(from_layers[0])
                bottom = topnames[prev_layer_id]
                topnames[layer_id] = bottom
                layer_id = layer_id + 1
            else:
                for index in range(len(from_layers)):
                    if int(from_layers[index]) < 0:
                        prev_layer_id = layer_id + int(from_layers[index])
                    else:
                        prev_layer_id = int(from_layers[index])
                    bottom = topnames[prev_layer_id]
                    bottom_tmp.append(bottom)
                concat_layer = OrderedDict()
                concat_layer['bottom'] = bottom_tmp
                if 'name' in block:
                    concat_layer['top'] = block['name']
                    concat_layer['name'] = block['name']
                else:
                    concat_layer['top'] = 'layer%d-concat' % layer_id
                    concat_layer['name'] = 'layer%d-concat' % layer_id
                concat_layer['type'] = 'Concat'
                
                if DEBUG:
                    print("concat_layer: %s" % concat_layer)
                    
                layers.append(concat_layer)
                bottom = concat_layer['top']
                topnames[layer_id] = bottom
                layer_id = layer_id+1
        elif block['type'] == 'shortcut':
            prev_layer_id1 = layer_id + int(block['from'])
            prev_layer_id2 = layer_id - 1
            bottom1 = topnames[prev_layer_id1]
            bottom2= topnames[prev_layer_id2]
            shortcut_layer = OrderedDict()
            shortcut_layer['bottom'] = [bottom1, bottom2]
            if 'name' in block:
                shortcut_layer['top'] = block['name']
                shortcut_layer['name'] = block['name']
            else:
                shortcut_layer['top'] = 'layer%d-shortcut' % layer_id
                shortcut_layer['name'] = 'layer%d-shortcut' % layer_id
            shortcut_layer['type'] = 'Eltwise'
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            shortcut_layer['eltwise_param'] = eltwise_param
            layers.append(shortcut_layer)
            bottom = shortcut_layer['top']
 
            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if 'name' in block:
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1           
            
        elif block['type'] == 'connected':
            fc_layer = OrderedDict()
            fc_layer['bottom'] = bottom
            if 'name' in block:
                fc_layer['top'] = block['name']
                fc_layer['name'] = block['name']
            else:
                fc_layer['top'] = 'layer%d-fc' % layer_id
                fc_layer['name'] = 'layer%d-fc' % layer_id
            fc_layer['type'] = 'InnerProduct'
            fc_param = OrderedDict()
            fc_param['num_output'] = int(block['output'])
            fc_layer['inner_product_param'] = fc_param
            layers.append(fc_layer)
            bottom = fc_layer['top']

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if 'name' in block:
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1

        ##add reorg layer yolov2###############
        elif block['type'] == 'reorg':
            reorg_layer = OrderedDict()
            reorg_layer['bottom'] = bottom
            if 'name' in block:
                reorg_layer['top'] = block['name']
                reorg_layer['name'] = block['name']
            else:
                reorg_layer['top'] = 'layer%d-reorg' % layer_id
                reorg_layer['name'] = 'layer%d-reorg' % layer_id
            reorg_layer['type'] = 'Reorg'
            reorg_param = OrderedDict()
            reorg_param['stride'] = 2
            reorg_layer['reorg_param'] = reorg_param
            layers.append(reorg_layer)
            bottom = reorg_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1
        ##add upsample layer yolov3###############
        elif block['type'] == 'upsample' :
            upsample_layer = OrderedDict()
            upsample_layer['bottom'] = bottom
            if 'name' in block:
                upsample_layer['top'] = block['name']
                upsample_layer['name'] = block['name']
            else:
                upsample_layer['top'] = 'layer%d-upsample' % layer_id
                upsample_layer['name'] = 'layer%d-upsample' % layer_id
            upsample_layer['type'] = 'DeephiResize'
            upsample_param = OrderedDict()
            upsample_param['scale_h'] = 2
            upsample_param['scale_w'] = 2
            upsample_layer['deephi_resize_param'] = upsample_param
            layers.append(upsample_layer)
            bottom = upsample_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1
        elif block['type'] == 'region' or block['type'] == 'yolo':
            layer_id = layer_id + 1
        #######################################
        else:
            print('unknow layer type %s ' % block['type'])
            topnames[layer_id] = bottom
            layer_id = layer_id + 1

    net_info = OrderedDict()
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print('try:')
        print('python convert.py yolov3.cfg yolov3.weights yolov3.prototxt yolov3.caffemodel')
        print('')
        print('please add name field for each block to avoid generated name')
        exit()
    cfgfile = sys.argv[1]
    weightfile = sys.argv[2]
    protofile = sys.argv[3]
    caffemodel = sys.argv[4]
    darknet2caffe(cfgfile, weightfile, protofile, caffemodel)

