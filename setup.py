
# -- local --
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pointcept",
    py_modules=["pointcept"],
    install_requires=[],
    package_dir={"": "."},
    packages=find_packages("."),
)
