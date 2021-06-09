"""
Unit test for OrvilleImageDB module.
"""

# Python2 compatibility
from __future__ import print_function, division, absolute_import
try:
    range = xrange
except NameError:
    pass
    
import os
import numpy
import tempfile
import unittest

import OrvilleImageDB


__version__  = "0.1"
__author__    = "Jayce Dowell"


oimsFile = os.path.join(os.path.dirname(__file__), 'data', 'test.oims')


class oims_tests(unittest.TestCase):
    """A unittest.TestCase collection of unit tests for the OrvilleImageDB
    module."""
    
    testPath = None

    def setUp(self):
        """Turn off all numpy warnings and create the temporary file directory."""

        numpy.seterr(all='ignore')
        self.testPath = tempfile.mkdtemp(prefix='test-oims-', suffix='.tmp')
        
    def test_oims_read(self):
        """Test reading in an image from a OrvilleImage file."""

        db = OrvilleImageDB.OrvilleImageDB(oimsFile, 'r')
        
        # Read in the first image with the correct number of elements
        hdr, data = db.read_image()
        ## Image
        self.assertEqual(data.shape[0], db.header.nchan)
        self.assertEqual(data.shape[1], len(db.header.stokes_params.split(b',')))
        self.assertEqual(data.shape[2], db.header.ngrid)
        self.assertEqual(data.shape[3], db.header.ngrid)
        
        db.close()
        
    def test_oims_loop(self):
        """Test reading in a collection of images in a loop."""
        
        db = OrvilleImageDB.OrvilleImageDB(oimsFile, 'r')
        
        # Go
        for i,(hdr,data) in enumerate(db):
            i
            
        db.close()
        
    def test_oims_write(self):
        """Test saving data to the OrvilleImageDB format."""
        
        # Setup the file names
        testFile = os.path.join(self.testPath, 'test.oims')
        
        db = OrvilleImageDB.OrvilleImageDB(oimsFile, 'r')
        db.header.station = b'LWA-SV'       # TODO: Need to fix this in orvile_imager.py
        nf = OrvilleImageDB.OrvilleImageDB(testFile, 'w', imager_version=db.header.imager_version, 
                                                          station=db.header.station)
                                            
        # Fill it
        for rec in db:
            nf.add_image(*rec)
            
        # Done
        db.close()
        nf.close()
        
        # Re-open
        db0 = OrvilleImageDB.OrvilleImageDB(oimsFile, 'r')
        db1 = OrvilleImageDB.OrvilleImageDB(testFile, 'r')
        
        # Validate
        ## File header
        for attr in ('imager_version', 'station', 'stokes_params', 'ngrid', 'nchan', 'flags'):
            print('0:', getattr(db0.header, attr, None))
            self.assertEqual(getattr(db0.header, attr, None), getattr(db1.header, attr, None))
        for attr in ('pixel_size', 'start_time',):
            # TODO: What's up with stop_time?
            self.assertAlmostEqual(getattr(db0.header, attr, None), getattr(db1.header, attr, None), 6)
        ## First image
        ### Image header
        hdr0, img0 = db0.read_image()
        hdr1, img1 = db1.read_image()
        for attr in ('stokes_params', 'ngrid', 'pixel_size', 'ngrid'):
            self.assertEqual(getattr(hdr0, attr, None), getattr(hdr1, attr, None))
        for attr in ('start_time', 'int_len', 'lst', 'start_freq', 'stop_freq', 'bandwidth', 'fill', 'center_ra', 'center_dec'):
            self.assertAlmostEqual(getattr(hdr0, attr, None), getattr(hdr1, attr, None), 6)
        ### Image
        for i in range(img0.shape[0]):
            for j in range(img0.shape[1]):
                for k in range(img0.shape[2]):
                    for l in range(img0.shape[3]):
                        self.assertAlmostEqual(img0[i,j,k,l], img1[i,j,k,l], 6)
                        
        db0.close()
        db1.close()
        
    def tearDown(self):
        """Remove the test path directory and its contents"""
        
        tempFiles = os.listdir(self.testPath)
        for tempFile in tempFiles:
            os.unlink(os.path.join(self.testPath, tempFile))
        os.rmdir(self.testPath)
        self.testPath = None


class oims_test_suite(unittest.TestSuite):
    """A unittest.TestSuite class which contains all of the OrvilleImageDB units 
    tests."""
    
    def __init__(self):
        unittest.TestSuite.__init__(self)
        
        loader = unittest.TestLoader()
        self.addTests(loader.loadTestsFromTestCase(oims_tests)) 


if __name__ == '__main__':
    unittest.main()
