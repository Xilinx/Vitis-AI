#!/usr/bin/env python

#
# This searches and indexes all of the example designs with description.json
# index files. To omit an example from the index, add "sdx_gui": "false" to
# the description.json.
#
# This should be run at the example root directory. The argument is the
# name of the output index file.
#

import os
import json
import collections
import sys
import subprocess

if len(sys.argv) > 2:
    print "Usage: %s [<catalog>.json]" % sys.argv[0]
    sys.exit(os.EX_USAGE)

echo_progress = (len(sys.argv) == 2)

def addexample(path):
    example = collections.OrderedDict()
    example["name"] = os.path.basename(path)
    example["commit_id"] = get_commit_id(path)
    example_json_file = os.path.join(path, "description.json")
    if (os.path.isfile(example_json_file)):
        with open(example_json_file) as json_file:
            data = json.load(json_file)
            if data.get("sdx_gui") or data.get("sdx_gui") == False:
                supported_in_gui = data.get("sdx_gui")
                if supported_in_gui == False:
                    return None
                if supported_in_gui.lower() not in [ "true", "1", "yes" ]:
                    return None
            if data.get("overview"):
                example["description"] = data.get("overview")
            if data.get("example"):
                example["displayName"] = data.get("example")
            if data.get("keywords"):
                example["keywords"] = data.get("keywords")
            if data.get("key_concepts"):
                example["key_concepts"] = data.get("key_concepts")
            if data.get("revision"):
                revisions = data.get("revision")
                highestVersion = ""
                for r in range(len(revisions)):
                    revision = revisions[r]
                    if revision["version"]:
                       highestVersion = revision["version"]
                # TODO: choose latest revision
                example["version"] = highestVersion
            if data.get("contributors"):
                contributors = data.get("contributors")
                for r in range(len(contributors)):
                    contributor = contributors[r]
                    author = contributor["group"]
                    example["author"] = author
    if echo_progress:
        print "adding %s" % path
    return example

def get_git_root_directory():
    # git rev-parse --show-toplevel
    p = subprocess.Popen(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE)
    dir = p.communicate()[0].strip()
    returncode = p.returncode
    if returncode == 0:
        return dir
    else:
        sys.stderr.write("Not a git repository. Could not find .git\n")
        sys.exit(os.EX_CONFIG)

def get_commit_id(filename):
    # git log -n 1 --pretty=format:%H $filename
    id = subprocess.Popen(["git", "log", "-n", "1", "--pretty=format:%H", filename], stdout=subprocess.PIPE).communicate()[0]
    return id

def get_git_branch():
    branch = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE).communicate()[0]
    # only works on python 2.7+:
    #branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    branch = branch.strip()
    return branch

def searchdir(dir):
    if (not os.path.isdir(dir)):
        sys.stderr.write("%s not found, but indexed in parent summary.json\n" % dir)
        emptycat = collections.OrderedDict()
        emptycat["categories"] = []
        emptycat["examples"] = []
        return emptycat
    dirnames = [entry for entry in os.listdir(dir) if os.path.isdir(os.path.join(dir,entry))]
    index = collections.OrderedDict()
    subdirs = {}
    category = collections.OrderedDict()
    category["name"] = os.path.basename(dir)
    category["displayName"] = ""
    category["description"] = ""
    category["categories"] = []
    category["examples"] = []
    summary_json_file = os.path.join(dir, "summary.json")
    if (os.path.isfile(summary_json_file)):
        with open(summary_json_file) as json_file:
            data = json.load(json_file)
            if data.get("overview"):
                category["displayName"] = data.get("overview")
            if data.get("description"):
                category["description"] = data.get("description")
            if data.get("subdirs"):
                subdirs = data["subdirs"]
    if (not subdirs):
        subdirs = dirnames
    subdirs = sorted(subdirs)
    # TODO: Sort subdirs
    for subdir in subdirs:
        if (subdir.startswith(".")):
            continue
        path = os.path.join(dir, subdir)
        example_file = os.path.join(path, "description.json")
        if (os.path.isfile(example_file)):
            example = addexample(path)
            if example is not None:
                category["examples"].append(example)
        else:
            subcategory = searchdir(path)
            if (subcategory is not None):
                category["categories"].append(subcategory)
    if (len(category["categories"]) + len(category["examples"]) > 0):
        return category
    return None

root = get_git_root_directory()
index = searchdir(root)
index["branch"] = get_git_branch()

if len(sys.argv) == 2:
    index_filename = sys.argv[1]
    indexdir = os.path.dirname(index_filename)
    if (indexdir):
        if (not os.path.exists(indexdir)):
            os.makedirs(indexdir)
    index_file = open(index_filename, "w")
else:
    index_file = sys.stdout
    
json.dump(index, index_file, indent = 4)
