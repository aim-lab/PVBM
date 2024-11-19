from setuptools import setup, find_packages


def read_readme(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return file.read()


long_description = read_readme("README.md")

setup(
    name='pvbm',
    version='2.9.9.5',
    packages=find_packages(),
    include_package_data=True,  # Ensure non-Python files are included
    package_data={
        "": ["*.onnx"],  # Include all .onnx files in all subdirectories
    },
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "pillow",
        "gdown",
        "onnxruntime",
        "torchvision",
        "opencv-python",
        "matplotlib"
    ],
    author='Jonathan Fhima, Yevgeniy Men',
    author_email='jonathanfh@campus.technion.ac.il',
    description="Python Vasculature Biomarker toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/aim-lab/PVBM',
)
