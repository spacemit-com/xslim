from setuptools import find_packages, setup
from xquant.defs import XQUANT_CONFIG


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


setup(
    author="SpacemiT",
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    license="Apache License 2.0",
    include_package_data=True,
    version=XQUANT_CONFIG.version,
    zip_safe=False,
)
