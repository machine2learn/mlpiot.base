from setuptools import find_packages, setup
import os.path


def read(*rnames):
    return open(os.path.join(os.path.dirname(__file__), *rnames)).read()


short_desc = 'Machine Learning Platform for Internet Of Things -- Base classes'
long_desc = \
    read('README.rst') + '\n\n' + \
    read('CHANGES.rst') + '\n\n' + \
    read('LICENSE')

version = '0.0.1'

setup(
    name='mlpiot.base',
    version=version,
    description=short_desc,
    long_description=long_desc,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Framework :: MLPIOT',
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='ML IOT',
    author='Machine2Learn BV',
    author_email='mlpiot@machine2learn.nl',
    url='http://mlpiot.com/',
    license='Proprietary',
    packages=find_packages('py_src'),
    package_dir={'': 'py_src'},
    namespace_packages=['mlpiot'],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy>=1.16,<2',
        'protobuf>=3.10.0,<4'
        'setuptools>=36.2',
    ],
    platforms='Any',
)
