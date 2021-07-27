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
        if keyword[0]!=model_file[0]:
            continue
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


def read_Ymal(yamlPath):
    yaml_txt=open(yamlPath)
    yaml_list=yaml2list(yaml_txt)
    index_dict={}
    index=0
    for i,ss in enumerate(yaml_list):
        if ss[0]=='type':
            index=index+1
            print(str(index)+': ',ss,end='')
        elif ss[0]=='board':
            print(ss)
        elif ss[0]=='download link':
            index_dict.update({index:[i,ss[1]]})
    num = int(input("input num:"))
    load_Link = index_dict[num][1]
    f=load_Link.index("filename=")
    name=load_Link[f+9:]

    return load_Link,name


def download(load_Link,name):
    def Schedule(a,b,c):
        per=100.0*a*b/c
        if per >100:
            per=100
        print('%.2f%%'%per)
    urllib.request.urlretrieve(load_Link, name,Schedule)
    print("done")

def main():
    print('Tip:')
    print("you need to input framework and model name, use space divide such as tf vgg16")
    print("tf:tensorflow1.x  tf2:tensorflow2.x  cf:caffe  dk:darknet  pt:pytorch")
    framework_list=['tf','tf2','cf','dk','pt','torchvision']
    keyword=list(input("input:").casefold().split())
    if keyword[0] not in framework_list:
        print("Please use correct framework keyword and framework ahead of model name")
        return 0
    model_list=find_Filename(keyword)
    if len(model_list)==0:
        print("None")
    elif len(model_list)==1:
        yamlPath=listpath+"/"+model_list[0]+"/model.yaml"
    else:
        for i,model in enumerate(model_list):
            print(i,":",model)
        num=int(input("input num:"))
        yamlPath=listpath+"/"+model_list[num]+"/model.yaml"
    load_Link,name=read_Ymal(yamlPath)
    download(load_Link,name)

main()