#!/usr/bin/python3
# Copyright 2022-2023 Advanced Micro Devices Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import platform

tmp_dir = "/workspace/tmp/"
tmp_lib_dir = os.path.join(tmp_dir, "lib")
ori_lib_dir = "/usr/lib/."

user_cmd = ""
for argv in sys.argv[1:]:
    user_cmd += " %s" % argv
vaitrace_cmd = "vaitrace"
sudo_cmd = "sudo -E"
env_cmd = ""

# 1. Env check, arch & overlayfs


def overlay_fs_bug_exist():
    pl = platform.uname().release.split('.')
    kernel_maj, kernel_min = int(pl[0]), int(pl[1])
    if kernel_maj >= 4 and kernel_min >= 8:
        return False

    overlay_fs = os.system('o=`df /usr/lib/ |grep overlay`;test X\"$o\" = X')
    if overlay_fs == 0:
        return False
    print("overlay_fs found! work it round")
    return True

# if inside_docker:
#  ...
# else:
#  ...

# check pth [/sys/kernel/debug/tracing]


def check_ftrace_path():
    return True


def overlay_fs_workaround():
    # 1. Create temp dir and copy libs
    os.system("mkdir -p %s" % tmp_lib_dir)
    print("Copying libraries")
    os.system("cp -r %s %s" % (ori_lib_dir, tmp_lib_dir))
    print("Copy finished")

    # 2. Change $LD_LIBRARY_PATH
    LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH")
    print(LD_LIBRARY_PATH)
    NEW_LD_LIBRARY_PATH = tmp_lib_dir + ':' + LD_LIBRARY_PATH
    print("new LD_LIBRARY_PATH:\n", NEW_LD_LIBRARY_PATH)
    global env_cmd
    env_cmd = 'env LD_LIBRARY_PATH="%s"' % NEW_LD_LIBRARY_PATH


def clean_tmp():
    print("Removing temp files")
    os.system("rm -rf %s" % tmp_dir)

# 4. Issue the command with sudo -E env $LD_LIBRARY_PATH $PATH $LIBRARY_PATH

# 5. Change mode of current dir for *.csv files


def chmod_cur():
    print("Changing mode of current dir")
    os.system("chmod 777 ./")


if not check_ftrace_path():
    print("Checking ftrace failed")

if overlay_fs_bug_exist():
    overlay_fs_workaround()
    chmod_cur()


exec_cmd = "%s %s %s %s" % (sudo_cmd, env_cmd, vaitrace_cmd, user_cmd)
print(exec_cmd)
os.system(exec_cmd)
clean_tmp()
