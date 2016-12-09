from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='acpy',
      version='0.1',
      description='AutoCalibration tool in Python',
      long_description=readme(),
      url='http://github.com/BoldingBruggeman/acpy',
      author='Jorn Bruggeman',
      author_email='jorn@bolding-bruggeman.com',
      license='GPL',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Numerical Models :: Calibration Tools',
          'License :: OSI Approved :: GPL License',
          'Programming Language :: Python :: 2.7',
      ],
#      scripts=['bin/acpy.py.py', 'bin/animate_2d.py', 'plotbest.py', 'plot.py'],
      scripts=['bin/acpy.py',],
#      packages=['acpy', 'acpy/job', 'acpy/optimize', 'acpy/result', 'acpy/transport'],
#      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)
