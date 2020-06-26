# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import os
import platform
import sys
import subprocess

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

libraries = ['sox']
include_dirs = []
extra_objects = []
extra_cflags=['-O3']

# Creating the version file
cwd = os.path.dirname(os.path.abspath(__file__))
version = '0.2'
sha = 'Unknown'

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

pytorch_package_dep = 'torch'

setup(
    name="augment",
    version=version,
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    ext_modules=[
        CppExtension(
            'augment_cpp',
            ['augment/augment.cpp'],
            libraries=libraries,
            include_dirs=include_dirs,
            extra_objects=extra_objects,
            extra_compile_args=extra_cflags)
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[pytorch_package_dep]
)
