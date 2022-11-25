#!/usr/bin/env python

import os, re
import fnmatch
import json


def get_testcases(dir):
  testcase_list = []
  for root, dirnames, filenames in os.walk(dir):
    for filename in fnmatch.filter(filenames, 'description.json'):
      testcase_list.append(os.path.join(root, filename))
  return testcase_list

def get_drives(dir):
  folders = []
  while 1:
    dir, folder = os.path.split(dir)
    if folder != "" and folder != ".":
        folders.append(folder)
    else:
        break
  folders.reverse()
  return folders

def get_immediate_subdirectories(dir):
    return [name for name in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, name))]

def gen_category(dir ,outfile, subdircount):

  links = "[" + dir +"]:"+ dir + "\n"
  testcaselist = get_testcases(dir);
  for testcase in testcaselist:
    drives = get_drives(testcase)
    link = ""
    if len(drives) <= subdircount : 
      continue
    for drive in drives:
      if drive == "description.json":
        continue
      link += drive +"/"
    links += "[" + link + "]:" + link + "\n"
    
    outfile.write("["+link+"][]")
    outfile.write("|")
    desc = open(testcase,'r')
    data = json.load(desc)
    outfile.write(('\n').join(data["overview"]))
    outfile.write("|")
    if 'key_concepts' in data:
        outfile.write("__Key__ __Concepts__")
        key_concepts = data["key_concepts"]
        for i, kc in enumerate(key_concepts):
            outfile.write("<br>")
            outfile.write(" - ")
            outfile.write(kc)
        outfile.write("<br>")
    if 'keywords' in data:    
        outfile.write("__Keywords__")
        keywords = data["keywords"]
        for  i, kw in enumerate(keywords):
            outfile.write("<br>")
            outfile.write(" - ")
            outfile.write(kw)
    outfile.write("\n")
    desc.close()
  return links

def genReadMe(dir):
  desc = open(os.path.join(dir,"summary.json"),'r')
  data = json.load(desc)
  outfile = open(os.path.join(dir, "README.md"), "w")
  outfile.write(('\n').join((data["overview"])))
  outfile.write("\n")
  outfile.write("==================================\n")
  outfile.write(('\n').join((data["description"])))
  outfile.write("\n")
  if 'subdirs' in data:
    subDirs = data['subdirs'];
  else:
    subDirs = get_immediate_subdirectories(dir);
  outfile.write("\nS.No.   | Category  | Description \n")
  outfile.write("--------|-----------|-----------------------------------------\n")
  counter = 1;
  links = ""
  
  for subDir in subDirs:
    desc_file = os.path.join(subDir,"summary.json")
    if os.path.exists(desc_file):
        subDirDesc = open(os.path.join(subDir,"summary.json"),'r')
        subDirData = json.load(subDirDesc)
        outfile.write(str(counter));
        outfile.write(" | [" +subDir +"][]      |")
        outfile.write(('\n').join(subDirData["description"]))
        outfile.write("\n")
        counter = counter + 1;
      
  outfile.write("\n __Examples Table__ \n")
  table_header = """
Example        | Description           | Key Concepts / Keywords 
---------------|-----------------------|-------------------------
"""
  outfile.write(table_header)
  for subDir in subDirs:
    links = links + gen_category(subDir,outfile,2);

  outfile.write("\n")
  outfile.write(links)
  outfile.close();

def genReadMe2(dir):
  desc = open(os.path.join(dir,"summary.json"),'r')
  data = json.load(desc)
  outfile = open(os.path.join(dir, "README.md"), "w")
  outfile.write(('\n').join((data["overview"])))
  outfile.write("\n")
  outfile.write("==================================\n")
  outfile.write(('\n').join((data["description"])))
  outfile.write("\n")
  outfile.write("\n __Examples Table__ \n")
  table_header = """
Example        | Description           | Key Concepts / Keywords 
---------------|-----------------------|-------------------------
"""
  outfile.write(table_header)
  links = gen_category(dir,outfile,1)
  outfile.write("\n")
  outfile.write(links)
  outfile.close();

