#!/usr/bin/env python
from sys import argv
import json
import os
import subprocess

DSA = 'xilinx:vcu1525:dynamic'
VERSION = 'SDAccel 2018.2'
DEVICES = {
    'xilinx:kcu1500:dynamic': {
       'version': '5.0',
       'name': 'Xilinx Kintex UltraScale KCU1500',
       'nae':  'nx4'
    },
    'xilinx:vcu1525:dynamic': {
       'version': '5.0',
       'name': 'Xilinx Virtex UltraScale+ VCU1525',
       'nae':  'nx5'
    }
}

def header(target,data):
    target.write(data["example"])
    target.write("\n")
    target.write("======================\n\n")
    target.write("This README file contains the following sections:\n\n")
    target.write("1. OVERVIEW\n")
    target.write("2. HOW TO DOWLOAD THE REPOSITORY\n")
    target.write("3. SOFTWARE TOOLS AND SYSTEM REQUIREMENTS\n")
    target.write("4. DESIGN FILE HIERARCHY\n")
    target.write("5. COMPILATION AND EXECUTION\n")
    target.write("6. EXECUTION IN CLOUD ENVIRONMENTS\n")
    target.write("7. SUPPORT\n")
    target.write("8. LICENSE AND CONTRIBUTING TO THE REPOSITORY\n")
    target.write("9. ACKNOWLEDGEMENTS\n\n\n")
    return

def download(target):
    target.write("## 2. HOW TO DOWNLOAD THE REPOSITORY\n")
    target.write("To get a local copy of the SDAccel example repository, clone this repository to the local system with the following command:\n")
    target.write("```\n")
    target.write("git clone https://github.com/Xilinx/SDAccel_Examples examples\n")
    target.write("```\n")
    target.write("where examples is the name of the directory where the repository will be stored on the local system.")
    target.write("This command needs to be executed only once to retrieve the latest version of all SDAccel examples. The only required software is a local installation of git.\n\n")
    return

def overview(target,data):
    target.write("## 1. OVERVIEW\n")
    target.write(('\n').join(data["overview"]))
    target.write("\n\n")
    if 'more_info' in data:
        target.write(('\n').join(data["more_info"]))
        target.write("\n\n")
    if 'perf_fields' in data:
        target.write("### PERFORMANCE\n")
        target.write(data["perf_fields"][0])
        target.write("|")
        target.write(data["perf_fields"][1])
        target.write("|")
        target.write(data["perf_fields"][2])
        target.write("\n")
        target.write("----|-----|-----\n")
        for result in data["performance"]:
            target.write(result["system"])
            target.write("|")
            target.write(result["constraint"])
            target.write("|")
            target.write(result["metric"])
            target.write("\n")
    if 'key_concepts' in data:
        target.write("***KEY CONCEPTS:*** ")
        elem_count = len(data["key_concepts"])
        for result in data["key_concepts"]:
            elem_count -= 1
            target.write(result)
            if elem_count != 0:
                target.write(", ")
        target.write("\n\n")
    if 'keywords' in data:
        target.write("***KEYWORDS:*** ")
        word_count = len(data["keywords"])
        for result in data["keywords"]:
            word_count -= 1
            target.write(result)
            if word_count != 0:
                target.write(", ")
        target.write("\n\n")
    return

def requirements(target,data):
    target.write("## 3. SOFTWARE AND SYSTEM REQUIREMENTS\n")
    target.write("Board | Device Name | Software Version\n")
    target.write("------|-------------|-----------------\n")

    boards = []
    if 'board' in data:
        board = data['board']
        boards = [word for word in DEVICES if word in board]
    else:
        nboard = []
        if 'nboard' in data:
            nboard = data['nboard']
        boards = [word for word in DEVICES if word not in nboard]

    for board in boards:
        target.write(DEVICES[board]['name'])
        target.write("|")
        target.write(board)
        target.write("|")
        for version in VERSION:
            target.write(version)
        target.write("\n")
    target.write("\n\n")
    target.write("*NOTE:* The board/device used for compilation can be changed by adding the DEVICES variable to the make command as shown below\n")
    target.write("```\n")
    target.write("make DEVICES=<device name>\n")
    target.write("```\n")
    target.write("where the *DEVICES* variable accepts either 1 device from the table above or a comma separated list of device names.\n\n")
    try:
      if data['opencv']:
                target.write("***OpenCV for Example Applications***\n\n")
                target.write("This application requires OpenCV runtime libraries. If the host does not have OpenCV installed use the Xilinx included libraries with the following command:\n\n")
                target.write("```\n")
                target.write("export LD_LIBRARY_PATH=$XILINX_SDX/lnx64/tools/opencv/:$LD_LIBRARY_PATH\n")
                target.write("```\n")
    except:
      pass
    return

def hierarchy(target):
    target.write("## 4. DESIGN FILE HIERARCHY\n")
    target.write("Application code is located in the src directory. ")
    target.write("Accelerator binary files will be compiled to the xclbin directory. ")
    target.write("The xclbin directory is required by the Makefile and its contents will be filled during compilation. A listing of all the files ")
    target.write("in this example is shown below\n\n")
    target.write("```\n")
    tree_cmd = ['git', 'ls-files']
    proc = subprocess.Popen(tree_cmd,stdout=subprocess.PIPE)
    output = proc.communicate()[0]
    target.write(output)
    target.write("```\n")
    target.write("\n")
    return

def compilation(target,data):
    target.write("## 5. COMPILATION AND EXECUTION\n")
    target.write("### Compiling for Application Emulation\n")
    target.write("As part of the capabilities available to an application developer, SDAccel includes environments to test the correctness of an application at both a software functional level and a hardware emulated level.\n")
    target.write("These modes, which are named sw_emu and hw_emu, allow the developer to profile and evaluate the performance of a design before compiling for board execution.\n")
    target.write("It is recommended that all applications are executed in at least the sw_emu mode before being compiled and executed on an FPGA board.\n")
    target.write("```\n")
    target.write("make TARGETS=<sw_emu|hw_emu> all\n")
    target.write("```\n")
    target.write("where\n")
    target.write("```\n")
    target.write("\tsw_emu = software emulation\n")
    target.write("\thw_emu = hardware emulation\n")
    target.write("```\n")
    target.write("*NOTE:* The software emulation flow is a functional correctness check only. It does not estimate the performance of the application in hardware.\n")
    target.write("The hardware emulation flow is a cycle accurate simulation of the hardware generated for the application. As such, it is expected for this simulation to take a long time.\n")
    target.write("It is recommended that for this example the user skips running hardware emulation or modifies the example to work on a reduced data set.\n")
    target.write("### Executing Emulated Application \n")
    target.write("***Recommended Execution Flow for Example Applications in Emulation*** \n\n")
    target.write("The makefile for the application can directly executed the application with the following command:\n")
    target.write("```\n")
    target.write("make TARGETS=<sw_emu|hw_emu> check\n\n")
    target.write("```\n")
    target.write("where\n")
    target.write("```\n")
    target.write("\tsw_emu = software emulation\n")
    target.write("\thw_emu = hardware emulation\n")
    target.write("```\n")
    target.write("If the application has not been previously compiled, the check makefile rule will compile and execute the application in the emulation mode selected by the user.\n\n")
    target.write("***Alternative Execution Flow for Example Applications in Emulation*** \n\n")
    target.write("An emulated application can also be executed directly from the command line without using the check makefile rule as long as the user environment has been properly configured.\n")
    target.write("To manually configure the environment to run the application, set the following\n")
    target.write("```\n")
    target.write("export LD_LIBRARY_PATH=$XILINX_SDX/runtime/lib/x86_64/:$LD_LIBRARY_PATH\n")
    target.write("export XCL_EMULATION_MODE=")
    try:
      if not data['xcl']:
            target.write('true')
      else:
            target.write('<sw_emu|hw_emu>')
    except:
        target.write('<sw_emu|hw_emu>')
    target.write('\n')
    target.write("emconfigutil --platform '" + DSA + "' --nd 1\n")
    target.write("```\n")
    target.write("Once the environment has been configured, the application can be executed by\n")
    target.write("```\n")
    target.write(data["em_cmd"])
    target.write("\n```\n")
    target.write("This is the same command executed by the check makefile rule\n")
    target.write("### Compiling for Application Execution in the FPGA Accelerator Card\n")
    target.write("The command to compile the application for execution on the FPGA acceleration board is\n")
    target.write("```\n")
    target.write("make all\n")
    target.write("```\n")
    target.write("The default target for the makefile is to compile for hardware. Therefore, setting the TARGETS option is not required.\n")
    target.write("*NOTE:* Compilation for application execution in hardware generates custom logic to implement the functionality of the kernels in an application.\n")
    target.write("It is typical for hardware compile times to range from 30 minutes to a couple of hours.\n\n")

def execution(target):
    target.write("## 6. Execution in Cloud Environments\n")
    target.write("FPGA acceleration boards have been deployed to the cloud. For information on how to execute the example within a specific cloud, take a look at the following guides.\n");
    target.write("* [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]\n")
    target.write("* [Nimbix Application Execution on Xilinx Kintex UltraScale Devices]\n")
    target.write("\n")

def nimbix(target):
    target.write("The developer instance hosting the SDAccel tools on Nimbix is not directly connected to an FPGA accelerator card.\n")
    target.write("FPGA Accelerator cards are available as part of the SDAccel Runtime application. There are several ways of executing an application on the available cards:\n\n")
    target.write("***Submit the application from the developer to runtime instances (recommended flow)***\n")
    target.write("* Create a credentials file for the runtime machine based on your Nimbix username and API key. For more information on how to obtain the API key, refer to ")
    target.write("[Nimbix Application Submission README][]. The credentials file ( ~/.nimbix_creds.json ) should look like\n")
    target.write("```\n")
    target.write("{\n")
    target.write("\t\"username\": \"<username>\",\n")
    target.write("\t\"api-key\": \"<apikey>\"\n")
    target.write("}\n")
    target.write("```\n\n")
    target.write("where the values for username and apikey have been set to the values from your Nimbix account.\n\n")
    target.write("*NOTE:* The home directory of a SDAccel developer instance is not persistent across sessions. Only files stored in the /data directory are kept between sessions.")
    target.write("It is recommended that a copy of the nimbix_creds.json file be stored in the /data directory and copied to the appropriate location in the home directory ")
    target.write("at the start of each development session.\n")
    target.write("* Launch the application\n")
    target.write("```\n")
    target.write("make check\n")
    target.write("```\n")
    target.write("***Launch the application from a remote system outside of the Nimbix environment***\n")
    target.write("* Follow the instructions in [Nimbix Application Submission README][]\n\n")
    target.write("* Use the following command to launch the application from the users terminal (on a system outside of the Nimbix environment)\n")
    target.write("```\n")
    target.write(data["hw_cmd"])
    target.write("\n")
    target.write("```\n\n")
    target.write("***Copy the application files from the Developer to Runtime instances on Nimbix***\n")
    target.write("* Copy the application *.exe file and xclbin directory to the /data directory\n")
    target.write("* Launch the application using the Nimbix web interface as described in [Nimbix Getting Started Guide][]\n")
    target.write("* Make sure that the application launch options in the Nimbix web interface reflect the applications command line syntax\n")
    target.write("```\n")
    target.write(data["em_cmd"])
    target.write("\n")
    target.write("```\n")
    return

def power(target):
    target.write("\n## 6. COMPILATION AND EXECUTION FOR IBM POWER SERVERS\n")
    target.write("View the SuperVessel [Walkthrough Video][] to become familiar with the environment.\n\n")
    target.write("Compile the application with the following command\n")
    target.write("```\n")
    target.write("make ARCH=POWER all\n")
    target.write("```\n")
    return

def support(target):
    target.write("\n## 7. SUPPORT\n")
    target.write("For more information about SDAccel check the [SDAccel User Guides][]\n\n")
    target.write("For questions and to get help on this project or your own projects, visit the [SDAccel Forums][].\n\n")
    target.write("To execute this example using the SDAccel GUI, follow the setup instructions in [SDAccel GUI README][]\n\n")
    return

def license(target):
    target.write("\n## 8. LICENSE AND CONTRIBUTING TO THE REPOSITORY\n")
    target.write("The source for this project is licensed under the [3-Clause BSD License][]\n\n")
    target.write("To contribute to this project, follow the guidelines in the [Repository Contribution README][]\n")

    return

def ack(target,data):
    target.write("\n## 9. ACKNOWLEDGEMENTS\n")
    target.write("This example is written by developers at\n")
    for contributor in data["contributors"]:
        target.write("- [")
        target.write(contributor["group"])
        target.write("](")
        target.write(contributor["url"])
        target.write(")\n")
    target.write("\n")
    return

def dirTraversal(stop_file):
    stop_search = None
    level_count = 1
    s = os.path.join('..', stop_file)
    while level_count < 20:
        s = os.path.join('..', s)
        if os.path.isfile(s):
            break
        level_count += 1
    return level_count

def relativeTree(levels):
    s = '../'
    for i in range(0,levels):
        s += '../'
    return s

def footer(target):
    relativeLevels = dirTraversal("LICENSE.txt")
    root = relativeTree(relativeLevels)
    target.write("[3-Clause BSD License]: " + root + "LICENSE.txt\n")
    target.write("[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx\n")
    target.write("[SDAccel User Guides]: http://www.xilinx.com/support/documentation-navigation/development-tools/software-development/sdaccel.html?resultsTablePreSelect=documenttype:SeeAll#documentation\n")
    target.write("[Nimbix Getting Started Guide]: http://www.xilinx.com/support/documentation/sw_manuals/xilinx2016_2/ug1240-sdaccel-nimbix-getting-started.pdf\n")
    target.write("[Walkthrough Video]: http://bcove.me/6pp0o482\n")
    target.write("[Nimbix Application Submission README]: " + root + "utility/nimbix/README.md\n")
    target.write("[Repository Contribution README]: " + root + "CONTRIBUTING.md\n")
    target.write("[SDaccel GUI README]: " + root + "GUIREADME.md\n")
    target.write("[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md\n")
    target.write("[Nimbix Application Execution on Xilinx Kintex UltraScale Devices]: " + root + "utility/nimbix/README.md\n")
    return

# Get the argument from the description
script, desc_file = argv

# load the description file
print "SDAccel README File Genarator"
print "Opening the description file '%s'" % desc_file
desc = open(desc_file,'r')

# load the json data from the file
print "Parsing the description file"
data = json.load(desc)
desc.close()

assert("OpenCL" in data['runtime'])

print "Generating the README for %s" % data["example"]
target = open("README.md","w")
header(target,data)
overview(target,data)
download(target)
requirements(target,data)
hierarchy(target)
compilation(target,data)
execution(target)
#nimbix(target)
#power(target)
support(target)
license(target)
ack(target,data)
footer(target)
target.close

