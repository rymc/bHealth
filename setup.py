from distutils.core import setup
from distutils.util import convert_path

with open("README.md", 'r') as f:
    long_description = f.read()

main_ns = {}
ver_path = convert_path('bhealth/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
        name='bHealth',
        version=main_ns['__version__'],
        packages=['bhealth',],
        license='Creative Commons Attribution-Noncommercial-Share Alike license',
        long_description=long_description,
        author='Michal Kozlowski, Miquel Perello-Nieto, Ryan McConville',
        author_email='miquel.perellonieto@bristol.ac.uk',
        url='https://github.com/rymc/bHealth',
)
