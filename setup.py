#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('hicitco', ['hicitco.pyx'], include_dirs=[np.get_include()])
]

setup(name='HiCitco',
      version='0.1',
      description='Hi-C Iterative Correction',
      author='Matthias Blum',
      author_email='mat.blum@gmail.com',
      url='https://github.com/matthiasblum/HiCitco',
      version='1.0',
      zip_safe=False,
      scripts=['bin/hicitco'],
      ext_modules=cythonize(extensions),
      requires=['Cython', 'numpy', 'scipy', 'pandas', 'matplotlib']
      )
