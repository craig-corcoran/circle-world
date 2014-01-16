import os
import setuptools

setuptools.setup(
    name='rsis',
    version='0.0.1',
    packages=setuptools.find_packages(),
    author='Craig Corcoran',
    author_email='ccor@cs.utexas.edu',
    license='MIT',
    url='http://github.com/craig-corcoran/circle-world',
    install_requires=['numpy', 'plac'],
    )
