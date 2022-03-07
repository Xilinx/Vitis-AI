from setuptools import setup, find_packages
from distutils.dir_util import copy_tree
import os
import shutil

notebooks_dir = "/usr/share/notebooks/base-trd"

def copy_notebooks():
    src_nb_dir = 'notebooks/'
    dst_nb_dir = notebooks_dir
    if os.path.exists(dst_nb_dir):
        shutil.rmtree(dst_nb_dir)
    copy_tree(src_nb_dir, dst_nb_dir)

copy_notebooks()

setup(
    name="base-trd",
    license='BSD 3-Clause License',
    author="Nikhil Ayyala",
    author_email="nikhil.ayyala@xilinx.com",
    packages=find_packages(),
    description="Versal Base TRD Example Designs in Jupyter Notebooks"
)
