from setuptools import setup, find_packages

setup(
    name='pvbm',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "pillow"
        
    ],
    author='Jonathan Fhima, Yevgeniy Men',
    author_email='jonathanfh@campus.technion.ac.il',
    description='Python Vasculature BioMarker toolbox',
    url='https://github.com/aim-lab/PVBM',
)
