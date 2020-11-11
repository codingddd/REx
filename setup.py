import os
import setuptools

readme_filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "README.md")
with open(readme_filepath, "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pytorch-rex',  
    version='0.0.0',
    author="Tong Zhu",
    author_email="tzhu1997@outlook.com",
    description="A toolkit for Relation Extraction and more...",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/Spico197/REx",
    packages=[
        "rex"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    install_requires=[
        "torch>=1.2.0",
        "numpy>=1.19.0"
    ],
    # package_data={
    #     'rex' : [
    #         'models/*.pth'
    #     ],
    # },
    # include_package_data=True,
)
