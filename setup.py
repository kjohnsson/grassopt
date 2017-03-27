import src
from distutils.core import setup

setup(name='grassopt',
      version=src.__version__,
      description='Optimization over Grassmann manifolds',
      author='Kerstin Johnsson',
      author_email='kerstin.johnsson@hotmail.com',
      url='https://github.com/kjohnsson/grassopt',
      license='MIT',
      packages=['grassopt'],
      package_dir={'grassopt': 'src'}
      )
