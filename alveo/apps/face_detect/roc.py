import numpy as np
import subprocess
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'analysis densebox model, must be deploy, not train')
    parser.add_argument('--fddbList', type = str, help = 'FDDB testset list', default='FDDB/FDDB_list.txt')
    parser.add_argument('--fddbPath', type = str, help = 'FDDB testset path', default='FDDB/')
    parser.add_argument('--results', type = str,  help = 'FDDB results path', default='FDDB_results.txt')
    parser.add_argument('--fddbAnno', type = str, help = 'FDDB testset annotations', default='FDDB/FDDB_annotations.txt')
    args = parser.parse_args()

    work_dir = os.getcwd() + '/'

    # evaluate
    subprocess.getstatusoutput('rm -f DiscROC.txt')
    subprocess.getstatusoutput('rm -f ContROC.txt')
    cmd = '%sevaluation/evaluate -a %s -d %s -i %s -l %s -r %s -z .jpg' % (work_dir, args.fddbAnno, args.results, args.fddbPath, args.fddbList, work_dir) 
    print (cmd)
    [status, _] = subprocess.getstatusoutput(cmd)
    
    DiscROC = np.loadtxt('DiscROC.txt')

    for i in range(96, 109):
        index = np.where(DiscROC[:, 1] == i)[0]
        if index :
            break

    print ( "Recall :" + str(np.mean(DiscROC[index], axis = 0)) )
