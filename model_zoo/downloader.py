# Copyright 2021 Xilinx Inc.
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

#!/usr/bin/env python3
import os
import urllib.request
import re

listpath="./model-list"
def find_Filename(keyword):
    model_list=[]
    for filename in os.listdir(listpath):
        model_file=filename.casefold().split("_")
        if keyword[0]=="all":
            model_list.append(filename)
        elif keyword[0]!=model_file[0]:
            continue
        else:
            if len(keyword)==1:
                model_list.append(filename)
            else:
                l=len(keyword[1])
                if l>len(model_file[1]):
                    continue
                elif keyword[1]==model_file[1][:l]:
                    model_list.append(filename)
    return model_list

def yaml2list(txt):
    yaml_list=[]
    for line in txt:
        line=line[:-1]
        line_list=line.split(":")
        if len(line_list)>=2:
            new_line_list=[]
            hppts_flag=0
            for x in line_list:
                if hppts_flag==0:
                    new_line_list.append(x)
                else:
                    new_line_list[-1]=new_line_list[-1]+":"+x
                if x==" https" or x==" http":
                    hppts_flag=1
            new_line=[]
            for x in new_line_list:
                blank_l=0
                for c in x:
                    if c==' ' or c=="-":
                        blank_l=blank_l+1
                    else:
                        break
                new_line.append(x[blank_l:])
            yaml_list.append(new_line)

    return yaml_list


def read_Ymal(yamlPathList):
    loadLinkDict={}
    for yamlPath in yamlPathList:
        yaml_txt=open(yamlPath)
        yaml_list=yaml2list(yaml_txt)
        index=''
        for i,ss in enumerate(yaml_list):
            if ss[0]=='board':
                index=ss[1]
                if index not in loadLinkDict:
                    loadLinkDict.update({index:[]})
            elif ss[0]=='download link':
                loadLinkDict[index].append(ss[1])
    loadLinkList=[]
    print("chose model type")
    print('0:','all')
    loadLinkDictName=[]
    for i, name in enumerate(loadLinkDict):
        print(i+1,":",name)
        loadLinkDictName.append(name)
    num = int(input("input num:"))

    if num==0:
        for i, name in enumerate(loadLinkDict):
            for link in loadLinkDict[name]:
                loadLinkList.append(link)
    else:
        name=loadLinkDictName[num-1]
        loadLinkList=loadLinkDict[name]

    return loadLinkList

def process_bar(percent, start_str='', end_str='', total_length=0):
    bar = ''.join(["\033[31m%s\033[0m" % '   '] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent * 100) + end_str
    print(bar, end='', flush=True)

def download(loadLinkList):

    def Schedule(a,b,c):
        per=100.0*a*b/c
        if per >100:
            per=100
        end_str = '100%'
        process_bar(per/100, start_str='', end_str=end_str, total_length=15)
    for load_Link in loadLinkList:
        f = load_Link.index("filename=")
        name = load_Link[f + 9:]
        print(name)
        urllib.request.urlretrieve(load_Link, name,Schedule)
        print()
    print("done")

def main():
    print('Tip:')
    print("you need to input framework and model name, use space divide such as tf vgg16")
    print("tf:tensorflow1.x  tf2:tensorflow2.x  cf:caffe  dk:darknet  pt:pytorch  all: list all model")
    framework_list=['tf','tf2','cf','dk','pt','torchvision','all']
    keyword=list(input("input:").casefold().split())
    if keyword[0] not in framework_list:
        print("Please use correct framework keyword and framework ahead of model name")
        return 0
    model_list=find_Filename(keyword)
    yamlPathList=[]
    if len(model_list)==0:
        print("None")
        return 0
    elif len(model_list)==1:
        yamlPath=listpath+"/"+model_list[0]+"/model.yaml"
        yamlPathList.append(yamlPath)
    else:
        #chose model
        print("chose model")
        print(0,":",'all')
        for i,model in enumerate(model_list):
            print(i+1,":",model)
        num=int(input("input num:"))

        if num==0:
            for i, model in enumerate(model_list):
                yamlPath = listpath + "/" + model_list[i] + "/model.yaml"
                yamlPathList.append(yamlPath)
        else:
            yamlPath = listpath + "/" + model_list[num-1] + "/model.yaml"
            yamlPathList.append(yamlPath)

    loadLinkList=read_Ymal(yamlPathList)
    download(loadLinkList)

main()
