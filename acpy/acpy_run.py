#!/usr/bin/env python2

import sys
import os.path

# Add parent directory to Python path to support importing acpy from source distribution.
#KBsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import acpy
from acpy.license.license import user,key, parallel

def main():
    acpy.main()

if __name__ == "__main__":
    # execute only if run as a script
    main()

