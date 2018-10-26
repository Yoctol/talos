from pathlib import Path

from setuptools import setup

readme = Path(__file__).parent.joinpath('README.md')
if readme.exists():
    with readme.open() as f:
        long_description = f.read()
else:
    long_description = '-'

setup(
    name='talos',
    version='0.1.0',
    description='Powerful Neural Network Builder',
    long_description=long_description,
    python_requires='>=3.6',
    packages=[
        'talos',
    ],
    author='Jsaon',
    author_email='jsaon@yoctol.com',
    url='',
    license='MIT',
    install_requires=[],
)
