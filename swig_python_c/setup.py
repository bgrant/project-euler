#!/usr/bin/env python

from setuptools import setup, Extension


ext = Extension('_euler',
                sources=['interface.i', '../c/euler.c'],
                )

setup(name='_euler',
      ext_modules=[ext],
      )
