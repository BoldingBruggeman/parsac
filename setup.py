import subprocess
from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

def get_sha():
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd='.')
    return sha[0:6]

setup(name='acpy',
      version=get_sha(),
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
      scripts=['bin/run_acpy.py'],
#      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      packages=find_packages(exclude=['need_updates']),
      include_package_data=True,
      zip_safe=False)
