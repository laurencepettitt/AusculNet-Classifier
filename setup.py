#!/usr/bin/env python3
"""The setup script."""

from setuptools import setup

requirements = [
    'pandas==0.25.3',
    'librosa==0.7.1',
    'keras==2.3.1',
    'tensorflow==2.5.3',
    'scikit-learn==0.21.3',
    'numpy==1.17.4',
    'respiratory-sounds @ git+ssh://git@github.com/laurencepettitt/RespiratorySounds-DataSet@master'
]

setup(
    name='AusculNet',
    version='0.1dev',
    packages=['ausculnet'],
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=requirements
)
