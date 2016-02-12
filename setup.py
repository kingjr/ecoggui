#! /usr/bin/env python
#
# Copyright (C) 2015 Jean-Remi King
# <jeanremi.king@gmail.com>

import os
#  from numpy.distutils.core import setup
from setuptools import setup

version = None
with open(os.path.join('ecoggui', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = """Tools to facilitate Ecog localization"""

DISTNAME = 'ecoggui'
DESCRIPTION = descr
MAINTAINER = 'Jean-Remi King'
MAINTAINER_EMAIL = 'jeanremi.king@gmail.com'
URL = 'https://github.com/kingjr/ecoggui'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/kingjr/ecoggui'
VERSION = version


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=False,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Linux'],
          platforms='any',
          packages=['ecoggui'],
          package_data={},
          scripts=[])
