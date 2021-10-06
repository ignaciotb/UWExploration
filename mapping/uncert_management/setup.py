## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['uncert_management'],
    package_dir={'': 'src'},
)

setup(**setup_args)

#  from setuptools import setup
  #
#  setup(name="bathymetric_svgp",
      #  version="0.0.0",
      #  packages=["gp_mapping"],
      #  package_dir={'': 'src'},
      #  )

