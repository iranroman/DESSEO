from setuptools import setup
import imp


with open('README.md') as file:
    long_description = file.read()

version = imp.load_source('desseo.version', 'desseo/version.py')

setup(
    name='desseo',
    version=version.version,
    description='A library to optimize the ASHLE model for synchronization with speech envelope onsets',
    author='Iran R. Roman',
    author_email='iran@ccrma.stanford.edu',
    url='',
    download_url='http://github.com/iranroman/DESSEO/releases',
    packages=['desseo'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='speech dynamical systems oscillator nonlinear pytorch torchdiffeq Hopf',
    license='Creative Commons Attribution',
    classifiers=[
            "License :: Creative Commons Attribution 4.0",
            "Programming Language :: Python",
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Theoretical Neuroscience :: Sound/Audio :: Nonlinear resonance",
            "Topic :: Theoretical Neuroscience :: Sound/Audio :: Oscillatory synchronization",
        ],
    install_requires=[
        'torchdiffeq>=0.2.3'
    ],
)

