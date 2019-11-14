#!/usr/bin/env python3
"""The setup script."""

from setuptools import setup

reqs=[
    'pandas==0.25.3',
    'librosa==0.7.1',
    'keras==2.3.1',
    'tensorflow==2.0.0',
    'scikit-learn==0.21.3',
    'numpy==1.17.4',
    'respiratory-sounds'
]

setup(
    name='AusculNet',
    version='0.1dev',
    packages=['ausculnet'],
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=reqs,
    dependency_links=[
        'git+ssh://git@github.com/laurencepettitt/RespiratorySounds-DataSet.git#egg=respiratory-sounds'
    ]
)