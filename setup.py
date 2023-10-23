from setuptools import find_packages
from setuptools import setup
from gadme import __version__

def requirements():
     with open("requirements.txt", "r") as file:
         lines = file.readlines()
         lines = [line.rstrip() for line in lines]
     return lines

setup(
    name='gadme',
    version=__version__,
    description='General Avian Monitoring Evaluation',
    author='Lukas Rauch',
    author_email='lukas.rauch@uni-kassel.de',
    url='https://github.com/DBD-research-group/Bird2Vec/tree/toolbox-structure',
    license='BSD 3-Clause',
    packages=find_packages(),
    install_requires=requirements(),
    extras_require={
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD 3-Clause License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='avian monitoring',
    python_requires=">=3.9",
)
