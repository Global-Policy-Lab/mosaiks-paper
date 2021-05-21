from setuptools import find_packages, setup

version = {}
exec(open("version.py").read(), version)

setup(
    name="mosaiks",
    version=version["__version__"],
    description="MOSAIKS Tool for Prediction with Satellite Imagery and "
    + "Machine Learning",
    packages=find_packages(),
    include_package_data=True,
)
