import os
import setuptools

dir_repo = os.path.abspath(os.path.dirname(__file__))
# read the contents of REQUIREMENTS file
with open(os.path.join(dir_repo, "requirements.txt"), "r") as f:
    requirements = f.read().splitlines()
# read the contents of README file
with open(os.path.join(dir_repo, "README.md"), encoding="utf-8") as f:
    readme = f.read()

setuptools.setup(
    name="arnet",
    version="1.2.0",
    description="A simple auto-regressive Neural Network for time-series",
    author="Oskar Triebe",
    url="https://github.com/ourownstory/AR-Net",
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["black"],
    },
    # setup_requires=[""],
    scripts=["scripts/arnet_dev_setup"],
    long_description=readme,
    long_description_content_type="text/markdown",
)
