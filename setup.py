from distutils.core import setup


setup(name='dpmm',
      version='1.0',
      description='Dirichlet Process Mixture Model',
      url='https://github.com/hannawallach/dpmm/',
      author='Hanna Wallach',
      author_email='hanna@dirichlet.net',
      license='Apache 2.0',
      packages=['dpmm'],
      install_requires=['kale', 'matplotlib', 'numpy', 'scipy'])
