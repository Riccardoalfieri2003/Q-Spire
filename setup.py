from setuptools import setup, find_packages

setup(
    name='qspire',
    version='1.0.0',
    packages=find_packages(),  # 👈 this includes the util/ folder
    install_requires=[
        'click',
    ],
    entry_points={
        'console_scripts': [
            'qspire = qspire.util.CLIModule:qspire',  # util/CLIModule.py -> def qspire()
        ],
    },
)