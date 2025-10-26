from setuptools import setup, find_packages

setup(
    name='qspire',
    version='1.0.0',
    packages=find_packages(where='qspire'),  # Look for packages inside 'qspire' folder
    package_dir={'': 'qspire'},  # Root package is in 'qspire' directory
    install_requires=[
        'click',
    ],
    entry_points={
        'console_scripts': [
            'qspire = util.CLIModule:qspire',  # Back to original path since we set package_dir
        ],
    },
)