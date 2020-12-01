import os
import sys
import shutil
from setuptools import setup, find_packages
#from setuptools import Extension
#from setuptools.command.build_ext import build_ext

project_name = 'dctc_lstm'
'''
extra_compile_args = ['-std=c++11', '-fPIC']
include_dirs = ['./instruction_generator/src/']

src_dir = './instruction_generator/src/'
sources = [src_dir + f for f in os.listdir(src_dir) if f.endswith('.cc')]

op_ext = Extension(
    'instruction_generator.kernel._c_src',
    sources=sources,
    language='c++',
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args
)
'''

src_path = "instruction_generator/lib/"
dst_path = sys.prefix + "/lib/"
all_lib_file = os.listdir(src_path)

for file in all_lib_file:
    shutil.copyfile(src_path+file, dst_path+file)

setup(
    name=project_name,
    version="1.0.0",
    description="A library Xilinx Compile Tool ",
    url="fill in later",
    author="donghan, kewang",
    author_email="donghan@xilinx.com, kewang@xilinx.com",
    license="Xilinx",
    packages=find_packages(),
    zip_safe=False,
    #cmdclass={'build_ext': build_ext},
    #ext_modules=[op_ext]
)

try:
    #os.system('rm -rf build dist '+project_name+'.egg-info')
    for dir_name in ['build','dist',project_name+'.egg-info']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
except:
    print("failed to do the cleaning, please clean up manully")

