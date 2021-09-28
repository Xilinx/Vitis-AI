import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

setup(
    name='libxperfmon',
    version='0.0.1',
    description='Example Module',
    packages=[''],
    package_dir={'': '/usr/lib'},
    package_data={'' : ['libxperfmon.so']},
)
