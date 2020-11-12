#!/usr/bin/env python

import argparse
import os

def get_layer_ops(filename):
    dir = os.path.dirname(filename)
    cmd = "./build/tools/get_operations_num %s -output %s >/dev/null 2>&1"%(filename, os.path.join( dir, 'ops.log' ))
    output = os.system(cmd)
    assert output==0
    layer_ops=dict()
    f = open(os.path.join( dir, 'ops.log'), 'r')
    for line in f:
        ln_sp = line.split()
        if(str(ln_sp[0])=="Total"):
            layer_ops[str(ln_sp[0])]=int(ln_sp[1])
            #  print str(ln_sp[0]),layer_ops[ln_sp[0]]
        else:
            layer_ops[str(ln_sp[0])]=int(ln_sp[2])
            #  print str(ln_sp[0]),layer_ops[ln_sp[0]]

    return layer_ops

def get_top_ops(filename):
    dir = os.path.dirname(filename)
    cmd = "./build/tools/get_operations_num %s -output %s >/dev/null 2>&1"%(filename, os.path.join( dir, 'ops.log' ))
    output = os.system(cmd)
    assert output==0
    top_ops=dict()
    f = open(os.path.join( dir, 'ops.log'), 'r')
    for line in f:
        ln_sp = line.split()
        if(str(ln_sp[0])=="Total"):
            top_ops[str(ln_sp[0])]=int(ln_sp[1])
            #  print str(ln_sp[0]),top_ops[ln_sp[0]]
        else:
            if not top_ops.has_key(str(ln_sp[1])):
                top_ops[str(ln_sp[1])]=int(ln_sp[2])
                #  print str(ln_sp[1]),top_ops[ln_sp[1]]
            else:
                print "Duplicated top: ",str(ln_sp[1]),", layer: ",str(ln_sp[0])

    return top_ops

def main():
    parser = argparse.ArgumentParser(description="Get operation numbers from prototxt")
    parser.add_argument('proto', help='input net prototxt file')
    parser.add_argument('-k','--key', help='use layer name or top name as key', default='top', choices=['top','layer'])
    args = parser.parse_args()

    if args.key=='top':
        ops = get_top_ops(args.proto)
    elif args.key=='layer':
        ops = get_layer_ops(args.proto)
    return ops

if __name__ == '__main__' :
    main()
    
