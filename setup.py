from setuptools import setup, find_packages

setup(
    name="eigen-lang",
    version="2.7.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20",
    ],
    entry_points={
        "console_scripts": [
            "eigen=src.main:main",
        ],
    },
)
