import os

from setuptools import Command, find_packages, setup

from xquant.defs import XQUANT_CONFIG


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


def version():
    with open("VERSION_NUMBER", encoding="utf-8") as f:
        content = f.read()
    return content


def license():
    with open("LICENSE", encoding="utf-8") as f:
        content = f.read()
    return content


setup(
    author="SpacemiT",
    author_email="xquant@spacemit.com",
    version=version(),
    description="XQuant is an offline quantization tools based on PPQ",
    long_description=readme(),
    long_description_content_type="text/markdown",
    install_requires=open("requirements.txt").readlines(),
    python_requires=">=3.6",
    name="xquant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],
    license=license(),
    include_package_data=True,
    zip_safe=False,
)
