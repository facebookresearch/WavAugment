# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import os
import subprocess

from setuptools import setup, find_packages

# Creating the version file
cwd = os.path.dirname(os.path.abspath(__file__))
version = '0.2'
sha = 'Unknown'

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    pass

setup(
    name="augment",
    version=version,
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    install_requires=['torch', 'torchaudio']
)
