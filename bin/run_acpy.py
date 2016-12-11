#!/usr/bin/env python2

import sys
import os.path

# Add parent directory to Python path to support importing acpy from source distribution.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import acpy
acpy.main()
