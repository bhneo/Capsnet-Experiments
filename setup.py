from setuptools import setup
from setuptools import find_packages

install_requires = {
    'numpy',
    'scipy',
    'tqdm'
}

setup(
    name="capslayer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    author="Naturomics Liao",
    author_email="naturomics.liao@gmail.com",
    url="https://github.com/naturomics/Experiments",
    license="Apache-2.0",
    install_requires=install_requires,
    description="Experiments: An advanced library for capsule theory",
    keywords="capsule, capsNet, deep learning, tensorflow",
    platform=['any']
)
