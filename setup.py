from setuptools import find_packages, setup

version = "0.0.1.dev0"

setup(
    name="procthor",
    packages=find_packages(),
    include_package_data=True,
    version=version,
    description="ProcTHOR: Large-Scale Embodied AI Using Procedural Generation",
    long_description="ProcTHOR: Large-Scale Embodied AI Using Procedural Generation",
    long_description_content_type="text/markdown",
    author_email="mattd@allenai.org",
    author="Matt Deitke",
    install_requires=[
        "numpy",
        "matplotlib",
        "trimesh",
        "python-sat",
        "python-fcl",
        "canonicaljson",
    ],
    url="https://procthor.allenai.org/",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 1 - Planning",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
