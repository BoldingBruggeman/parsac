from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='acpy',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description='AutoCalibration tool written in Python',
      long_description=readme(),
      url='http://github.com/BoldingBruggeman/acpy',
      author='Jorn Bruggeman',
      author_email='jorn@bolding-bruggeman.com',
      license='GPL',
      install_requires=['netCDF4'],
      classifiers=[ # Note: classifiers MUST match options on https://pypi.org/classifiers/ for PyPI submission
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Programming Language :: Python',
      ],
      entry_points={
          'console_scripts': [
                'acpy=acpy.acpy_run:main',
          ]
      },
      packages=find_packages(exclude=['need_updates']),
      package_data={'acpy': ['service.txt', 'examples/*']},
      include_package_data=True,
      zip_safe=False)
