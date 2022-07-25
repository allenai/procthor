from setuptools import find_packages, setup

version = "<REPLACE_WITH_VERSION>"

setup(
    name="procthor",
    packages=find_packages(),
    include_package_data=True,
    version=version,
    description="None",
    long_description="None",
    long_description_content_type="text/markdown",
    author_email="anon@gmail.com",
    author="Anon",
    install_requires=["numpy", "matplotlib", "trimesh", "python-sat"],
    url="https://google.com",
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
