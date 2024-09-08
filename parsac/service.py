#!/usr/bin/env python

from __future__ import print_function
import os
import argparse
import configparser

user = None
email = None
key = None
parallel = None
have_service = False

path = os.path.dirname(os.path.realpath(__file__))
service_file  = os.path.join(path, 'service.txt')

config = configparser.ConfigParser()

def read():
    global user
    global email
    global key
    global parallel
    global have_service
    if os.path.isfile(service_file):
        try:
            config.read(service_file)
            user = config.get('User','user')
            try:
               email = config.get('User','email')
            except:
                email = 'none'
            try:
                expire = config.get('User','expire')
            except:
                expire = 'never'
            key = config.get('Key','key')
            parallel = config.get('Features','parallel')
            have_service = True
        except configparser.ParsingError as err:
            print('Could not parse:', err)

#def write():
#    print('Write cfg-file')

def print_c():
    if have_service:
        print('Reading service from:')
        print('   '+service_file)
        for section_name in config.sections():
            print('Section:', section_name)
            for name, value in config.items(section_name):
                print('  %s = %s' % (name, value))
    else:
        print("No service file found!")

def main(args):
    read()
    print_c()

if __name__ == "__main__":
    args = None
    main(args)
