#!/usr/bin/env python

import os
import argparse
import ConfigParser

user = None
key = None
parallel = None

path = os.path.dirname(os.path.realpath(__file__))
license_file  = os.path.join(path, 'license.txt')

config = ConfigParser.SafeConfigParser()

def read():
    try:
        config.read(license_file)
        user = config.get('User','user')
        key = config.get('Key','key')
        parallel = config.get('Features','parallel')
    except ConfigParser.ParsingError, err:
        print 'Could not parse:', err

#def write():
#    print('Write cfg-file')

def print_c():
   for section_name in config.sections():
       print 'Section:', section_name
       for name, value in config.items(section_name):
           print '  %s = %s' % (name, value)
       print

def main(args):
    read()
    print_c()

if __name__ == "__main__":
    print('Reading license from:')
    print('   '+license_file)
    print
    args = None
    main(args)




