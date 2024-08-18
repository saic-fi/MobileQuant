import io, pathlib, pkg_resources
from setuptools import setup, find_packages 


with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_txt)]


setup(
    name='mobilellm',
    packages=["mobilellm"],
    version='0.1',
    description='MobileQuant',
    author='Fuwen Tan',
    url='https://github.com/saic-fi/MobileQuant',
    install_requires=install_requires,
    entry_points={"console_scripts": []},
    package_data={},
    classifiers=["Programming Language :: Python :: 3"],
)