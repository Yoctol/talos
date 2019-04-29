from pathlib import Path

from setuptools import setup

readme = Path(__file__).parent.joinpath('README.md')
if readme.exists():
    with readme.open() as f:
        long_description = f.read()
else:
    long_description = '-'

REQUIRED_PACKAGES = [
    'tensorflow==1.11.0',
]


setup(
    name='talos',
    version='1.0.14',
    description='Powerful Neural Network Builder',
    long_description=long_description,
    python_requires='>=3.6',
    packages=[
        'talos',
    ],
    install_requires=REQUIRED_PACKAGES,
    author='Jsaon',
    author_email='jsaon@yoctol.com',
    url='',
    license='MIT',
)
