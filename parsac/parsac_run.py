#!/usr/bin/env python

import sys
import os.path

# Add parent directory to Python path to support importing from source distribution.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import parsac

def main():
    parsac.main()

if __name__ == "__main__":
    # execute only if run as a script
    main()

