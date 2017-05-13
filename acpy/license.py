#!/usr/bin/env python

import os
import argparse
import ConfigParser

user = None
email = None
key = None
parallel = None
have_license = False

path = os.path.dirname(os.path.realpath(__file__))
license_file  = os.path.join(path, 'license.txt')

config = ConfigParser.SafeConfigParser()

def read():
    global user
    global email
    global key
    global parallel
    global have_license
    if os.path.isfile(license_file):
        try:
            config.read(license_file)
            user = config.get('User','user')
            email = config.get('User','email')
            key = config.get('Key','key')
            parallel = config.get('Features','parallel')
            have_license = True
        except ConfigParser.ParsingError, err:
            print 'Could not parse:', err

#def write():
#    print('Write cfg-file')

def print_c():
    if have_license:
        print('Reading license from:')
        print('   '+license_file)
        for section_name in config.sections():
            print 'Section:', section_name
            for name, value in config.items(section_name):
                print '  %s = %s' % (name, value)
    else:
        print("No license file found!")

def main(args):
    read()
    print_c()

if __name__ == "__main__":
    args = None
    main(args)
