from setuptools import setup, find_packages
def read_readme(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return file.read()

long_description = read_readme("README.md")


setup(
    name='pvbm',
    version='2.1.6',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "pillow"
        
    ],
    author='Jonathan Fhima, Yevgeniy Men',
    author_email='jonathanfh@campus.technion.ac.il',
    description="Python Vasculature Biomarker toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Add this line
    url='https://github.com/aim-lab/PVBM',
)
