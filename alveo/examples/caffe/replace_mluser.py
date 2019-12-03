from __future__ import print_function
import os, argparse

homedir = os.environ['HOME']
mlsroot = os.environ['VAI_ALVEO_ROOT']

def substitute(filepath, replacee, replacement):
    # Read in the file
    with open(filepath, 'r') as f :
        filedata = f.read()
    # Replace the target string
    filedata = filedata.replace(replacee, replacement)
    # Write the file out again
    with open(filepath, 'w') as f:
        f.write(filedata)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace CK-TOOLS Root paths in prototxt(s)')
    parser.add_argument('--rootdir', type=str, default=os.environ['HOME'], 
            help='Root path to CK-TOOLS directory. if not provided, "/home/mluser" gets replaced by "/home/username"')
    parser.add_argument('--modelsdir', type=str, default=mlsroot+"/models/container/caffe", 
            help='Root path to MODELS directory.')
    args = parser.parse_args()

    for Dir, subDirs, files in os.walk(args.modelsdir):
        success = [substitute(os.path.join(Dir, f), "/home/mluser", args.rootdir) for f in files if f.endswith(".prototxt")]
