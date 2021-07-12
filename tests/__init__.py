"""
Modules defining package tests.
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
try:
    range = xrange
except NameError:
    pass
    
# Path manipulation to get OrvilleImageDB.py into the path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.basename(__file__), '..')))

__version__   = "0.1"
__author__    = "Jayce Dowell"

from . import test_oims
