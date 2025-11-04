from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md') as f:
    long_description = f.read()

setup(
    name='qspire',
    version='1.0.0',

    packages=find_packages(where='qspire'),  # Look for packages inside 'qspire' folder
    package_dir={'': 'qspire'},  # Root package is in 'qspire' directory

    install_requires=[
        'click',
        requirements
    ],
    python_requires='>=3.10',

    description='QSpire is a Tool to detect Quantum Code Smells in the Qiskit framework',
    long_description=long_description,

    entry_points={
        'console_scripts': [
            'qspire = util.CLIModule:qspire',  # Back to original path since we set package_dir
        ],
    },
)