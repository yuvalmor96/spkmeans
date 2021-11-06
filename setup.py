from setuptools import Extension, setup

module = Extension("myspkmeans",
                  sources=[
                    'spkmeans.c',
                    'spkmeansmodule.c'
                  ])
setup(name='myspkmeans',
     version='1.0',
     description='Python wrapper for custom C extension',
     ext_modules=[module])
