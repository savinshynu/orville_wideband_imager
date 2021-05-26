from __future__ import print_function, division, absolute_import
import sys
if sys.version_info > (3,):
    xrange = range
    
import os
import numpy
import ctypes
import struct
import shutil
import tempfile


class PrintableLittleEndianStructure(ctypes.LittleEndianStructure):
    """
    Sub-class of ctypes.LittleEndianStructure that adds a as_dict()
    method for accessing all of the fields as a dictionary.
    """
    
    def as_dict(self):
        """
        Return all of the structure fields as a dictionary.
        """
        
        out = {}
        for field in self._fields_:
            out[field[0]] = getattr(self, field[0], None)
        return out
        
    def __repr__(self):
        return repr(self.as_dict())


class OrvilleImageDB(object):
    """
    Encapsulates a OrvilleImageDB binary file.
    
    This class can be used for both reading and writing OrvilleImageDB files.
    For reading, initialize with mode = "r" and use the read_image() function.  
    For writing, use mode = "a" or "w" and the add_image() function.  Be sure 
    to always call the close() method after adding images in order to update 
    the file's header information.
    
    Public module variables:
      header -- a class with member variables describing the data, including:
        imager_version, imager version used to create the images
        station, the station name
        stokes_params, the comma-delimited parameters (e.g., 'I,Q,U,V')
        ngrid, the dimensions of the image in pixels
        pixel_size, the physical size of a pixel, in degrees
        nchan, the number of channels for each image
        flags, a bitfield: 0x1 = sorted; others zero
        start_time, the earliest time of data covered by this file, in MJD UTC
        stop_time, the latest time of data, in MJD UTC
    """
    
    # The OrvilleImageDB files start with a 24 byte string that specifies the
    # data file's format version.  Following it is that format's header block,
    # defined by the _FileHeader structure.  After the header are the 
    # integration blocks.  An integration block starts with a header, defined
    # by the _EntryHeader structure.  Following that are the images, packed as 
    # [chan, stokes, x, y] and float32s.  The alignment of the images is little 
    # endian.
    #
    # All absolute times are in MJD UTC, LSTs are in days (i.e., from 0 to
    # 0.99726957), and integration lengths are in days.  Sky directions
    # (including RA) and pixel sizes are in degrees.  All other entries are in
    # standard mks units.
    
    _FORMAT_VERSION = b'OrvilleImageDBv002'
    
    class _FileHeader_v1(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('imager_version', ctypes.c_char*24),
                    ('station',        ctypes.c_char*24),
                    ('stokes_params',  ctypes.c_char*24),
                    ('ngrid',          ctypes.c_int),
                    ('pixel_size',     ctypes.c_double),
                    ('nchan',          ctypes.c_int),
                    ('flags',          ctypes.c_uint),
                    ('start_time',     ctypes.c_double),
                    ('stop_time',      ctypes.c_double)]
    _FileHeader_v2 = _FileHeader_v1
    
    FLAG_SORTED = 0x0001
    
    class _EntryHeader_v1(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('sync_word',  ctypes.c_uint),
                    ('start_time', ctypes.c_double),
                    ('int_len',    ctypes.c_double),
                    ('lst',        ctypes.c_double),
                    ('start_freq', ctypes.c_double),
                    ('stop_freq',  ctypes.c_double),
                    ('bandwidth',  ctypes.c_double),
                    ('center_ra',  ctypes.c_double),
                    ('center_dec', ctypes.c_double)]
    class _EntryHeader_v2(PrintableLittleEndianStructure):
        _pack_   = 1
        _fields_ = [('sync_word',  ctypes.c_uint),
                    ('start_time', ctypes.c_double),
                    ('int_len',    ctypes.c_double),
                    ('fill',       ctypes.c_double),
                    ('lst',        ctypes.c_double),
                    ('start_freq', ctypes.c_double),
                    ('stop_freq',  ctypes.c_double),
                    ('bandwidth',  ctypes.c_double),
                    ('center_ra',  ctypes.c_double),
                    ('center_dec', ctypes.c_double),
                    ('center_az',  ctypes.c_double),
                    ('center_alt', ctypes.c_double)]
        
    _TIME_OFFSET_v1 = 4
    _TIME_OFFSET_v2 = _TIME_OFFSET_v1
    
    def __init__(self, filename, mode='r', imager_version='', station=''):
        """
        Constructs a new OrvilleImageDB.
        
        Optional arguments specify the file mode (must be 'r', 'w', or, 'a';
        defaults to 'r') and strings providing the imager version and the 
        station name, all of which are truncated at 24 bytes.  These optional 
        strings are only relevant when opening a file for writing.
        """
        
        self.name = ''
        self.file = None
        self.curr_int = -1
        
        self._FileHeader = self._FileHeader_v2
        self._EntryHeader = self._EntryHeader_v2
        self._TIME_OFFSET = self._TIME_OFFSET_v2
        
        # 'station' is a required keyword
        if mode[0] == 'w' and station == '':
            raise RuntimeError("'station' is a required keyword for 'mode=w'")
            
        # For read mode, we do not create a new file.  Raise an error if it
        # does not exist, and create an empty OrvilleImageDB object if its length
        # is zero.
        if mode == 'r':
            self._is_new = False
            if not os.path.isfile(filename):
                raise OSError('The specified file, "%s", does not exist.' % filename)
            fileSize = os.path.getsize(filename)
            if fileSize == 0:
                self.version = self._FORMAT_VERSION
                self.header = self._FileHeader()
                self.curr_int = 0
                self.nint = 0
                self.nstokes = 0
                return
                
        # For append mode, check if the file exists and is at least longer
        # than the initial 24 byte version string.  If that's the case, switch
        # to 'r+' mode, since we may need to read and/or write to the header,
        # and some Unix implementations don't allow this with 'a' mode.
        # Otherwise, switch to write mode.
        elif mode == 'a':
            fileSize = os.path.getsize(filename) if os.path.isfile(filename) else 0
            self._is_new = (fileSize <= 24)
            mode = 'w' if self._is_new else 'r+'
            
        # Write mode: pretty straightforward.
        elif mode == 'w':
            self._is_new = True
            
        else:
            raise ValueError("Mode must be 'r', 'w', or 'a'.")
            
        # Now read or create the file header.
        mode += 'b'
        self.name = filename
        self.file = open(filename, mode)
        self._is_outdated = False
        
        if not self._is_new:
            self.version = self.file.read(24).rstrip(b'\x00')
            if self.version != self._FORMAT_VERSION:
                if self.version == b'OrvilleImageDBv001':
                    self._FileHeader = self._FileHeader_v1
                    self._EntryHeader = self._EntryHeader_v1
                    self._TIME_OFFSET = self._TIME_OFFSET_v1
                else:
                    raise KeyError('The file "%s" does not appear to be a '
                                   'OrvilleImageDB file.  Initial string: "%s"' %
                                   (fileName, self.version))
            
            file_header = self._FileHeader()
            
            if mode != 'r' and fileSize <= 24 + ctypes.sizeof(file_header):
                # If the file is too short to have any data in it, close it
                # and start a new one.  This one is probably corrupt.
                self.file.close()
                self._is_new = True
                mode = 'w'
                self.file = open(filename, mode)
            
            else:
                # It looks like we should have a good header, at least ....
                self.header = self._FileHeader()
                self.file.readinto(self.header)
                self.nstokes = len(self.header.stokes_params.split(b','))
                
                entry_header = self._EntryHeader()
                int_size = ctypes.sizeof(entry_header) \
                          + 4*self.header.nchan*(0 + self.nstokes*self.header.ngrid**2)
                if (fileSize - 24 - ctypes.sizeof(self.header)) % int_size != 0:
                    raise RuntimeError('The file "%s" appears to be '
                                       'corrupted.' % filename)
                self.nint = \
                    (fileSize - 24 - ctypes.sizeof(self.header)) // int_size
                
                if mode == 'r+b':
                    self.file.seek(0, os.SEEK_END)
                    self.curr_int = self.nint
                else:
                    self.curr_int = 0
                    
        if self._is_new:
            # Start preparing a file header, but don't write it until we
            # receive the first image, which will fill in some information
            # (e.g., resolution) that isn't yet available.
            self.version = self._FORMAT_VERSION
            self.header = self._FileHeader()
            self.header.flags = self.FLAG_SORTED     # Sorted until it's not
            self.nint = 0
            
    def __del__(self):
        if self.file is not None and not self.file.closed:
            self.close()
            
    def close(self):
        """
        Closes the database file.  If the header information is outdated, it
        writes the new file header.
        """
        
        if self.file is None or self.file.closed:  return
        
        if self._is_outdated:
            self.file.seek(24, os.SEEK_SET)
            self.file.write(self.header)
            
        self.file.close()
        self.curr_int = -1
        
    def closed(self):
        return self.file is None or self.file.closed
        
    def getpos(self):
        return self.curr_int
        
    def eof(self):
        return self.curr_int >= self.nint
        
    def seek(self, index):
        if index < 0:
            index += self.nint
        if index < 0 or index >= self.nint:
            raise IndexError('OrvilleImageDB index %d outside of range [0, %d)' %
                             (index, self.nint))
        if self.curr_int != index:
            entry_header = self._EntryHeader()
            int_size = ctypes.sizeof(entry_header) \
                       + 4*self.header.nchan*(0 + self.nstokes*self.header.ngrid**2)
            file_header = self._FileHeader()
            headerSize = 24 + ctypes.sizeof(file_header)
            self.file.seek(headerSize + int_size * index, os.SEEK_SET)
            self.curr_int = index
            
    def _check_header(self, stokes_params, ngrid, pixel_size, nchan):
        """
        For new files, adds the given information to the file header and
        writes the header to disk.  For existing files, compares the given
        information to the expected values and raises a ValueError if there's
        a mismatch.
        """
        
        if type(stokes_params) is list:
            stokes_params = ','.join(stokes_params)
            
        if self._is_new:
            # If this is the file's first image, fill in values of the file
            # header based on the image properties, then write the header.
            self.header.stokes_params = stokes_params
            self.header.ngrid         = ngrid
            self.header.pixel_size    = pixel_size
            self.header.nchan         = nchan
            self.file.write(struct.pack('<24s', self.version))
            self.file.write(self.header)
            self.nstokes = len(self.header.stokes_params.split(','))
            self._is_new = False
            
        else:
            # Make sure that the Stokes parameters match expectations.
            if stokes_params != self.header.stokes_params:
                raise ValueError(
                    'The Stokes parameters for this image (%s) do not match '
                    'this file\'s parameters (%s).' %
                    (stokes_params, self.header.stokes_params))
                
            # Make sure that the dimensions of the data match expectations.
            if ngrid != self.header.ngrid:
                raise ValueError(
                    'The spatial resolution of this image (%d x %d) does not '
                     'match this file\'s resolution (%d x %d).' %
                    (ngrid, ngrid, self.header.ngrid, self.header.ngrid))
                
            if pixel_size != self.header.pixel_size:
                raise ValueError(
                    'The pixel size of this image (%r deg x %r deg) does not '
                     'match this file\'s resolution (%r deg x %r deg).' %
                    (pixel_size, pixel_size,
                     self.header.pixel_size, self.header.pixel_size))
                
            # Make sure that the size of the images matches expectations.
            if nchan != self.header.nchan:
                raise ValueError(
                    'The channel count for this image (%d) does not '
                    'match this file\'s channel count (%d).'
                    % (nchan, self.header.nchan))
                
    def _update_file_header(self, interval):
        """
        To be called at the end of the add_image functions.  Updates the header
        information to reflect the new data.
        """
        
        self.nint += 1
        
        # Has this image expanded the time range covered by the file?
        if self.header.start_time == 0 or \
           self.header.start_time > interval[0]:
            self.header.start_time = interval[0]
            self._is_outdated = True
            
        if self.header.stop_time < interval[1]:
            self.header.stop_time = interval[1]
            self._is_outdated = True
            
        # If the new image isn't later than all the others, and the file is
        # currently marked as sorted, then remove the sorted flag.
        elif self.header.flags & self.FLAG_SORTED:
            self.header.flags &= ~self.FLAG_SORTED
            self._is_outdated = True
            
    def add_image(self, info, data):
        """
        Adds an integration to the database.  Returns the index of the newly
        added image.
        
        Arguments:
        info -- a dictionary with the following keys defined:
            start_time -- MJD UTC at which this integration began
            int_len -- integration length, in days
            fill -- packet fill fraction
            lst -- mean local sidereal time of the observation, in days
            start_freq -- frequency of first channel in the integration, in Hz
            stop_freq -- frequency of last channel in the integration, in Hz
            bandwidth -- bandwidth of each channel in the integrated data, in Hz
            center_ra -- RA of the image phase center, in degrees
            center_dec -- Declination of image phase center, in degrees
            center_az -- azimuth of the image phase center, in degrees
            center_alt -- altitude of image phase center, in degrees
            pixel_size -- Real-world size of a pixel, in degrees
            stokes_params -- a list or comma-delimited string of Stokes params
        data -- a 4D float array of image data indexed as [chan, stokes, x, y]
        """
        
        assert(data.shape[2] == data.shape[3])
        self._check_header(info['stokes_params'], data.shape[2], 
                           info['pixel_size'], data.shape[0])
        
        # Write it out.
        entry_header = self._EntryHeader()
        entry_header.sync_word = 0xC0DECAFE
        for key in ('start_time', 'int_len', 'fill', 'lst', 'start_freq', 'stop_freq',
                    'bandwidth', 'center_ra', 'center_dec', 'center_az', 'center_alt'):
            if key in ('fill', 'center_az', 'center_alt') and self.version != self._FORMAT_VERSION:
                continue
            setattr(entry_header, key, info[key])
        self.file.write(entry_header)
        data.astype('<f4').tofile(self.file)
        self.file.flush()
        
        interval = [info['start_time'], info['start_time'] + info['int_len']]
        self._update_file_header(interval)
        return self.nint - 1
        
    def read_image(self):
        """
        Reads an integration from the database.
        
        Returns a 3-tuple containing:
        info -- a dictionary with the following keys defined:
            start_time -- MJD UTC at which this integration began
            int_len -- integration length, in days
            fill -- packet fill fraction, if available
            lst -- mean local sidereal time of the observation, in days
            start_freq -- frequency of first channel in the integration, in Hz
            stop_freq -- frequency of last channel in the integration, in Hz
            bandwidth -- bandwidth of each channel in the integrated data, in Hz
            center_ra -- RA of the image phase center, in degrees
            center_dec -- Declination of image phase center, in degrees
            center_az -- azimuth of the image phase center, in degrees
            center_alt -- altitude of image phase center, in degrees
            pixel_size -- Real-world size of a pixel, in degrees
            stokes_params -- a list or comma-delimited string of Stokes params
        data -- a 4D float array of image data indexed as [chan, stokes, x, y]
        """
        
        if self.curr_int >= self.nint:
            raise IOError("end of file reached")
            
        entry_header = self._EntryHeader()
        self.file.readinto(entry_header)
        if entry_header.sync_word != 0xC0DECAFE:
            raise RuntimeError("Database corrupted")
        info = {}
        for key in ('start_time', 'int_len', 'fill', 'lst', 'start_freq', 'stop_freq',
                    'bandwidth', 'center_ra', 'center_dec', 'center_az', 'center_alt'):
            info[key] = getattr(entry_header, key, None)
            
        nchan, nstokes, ngrid = self.header.nchan, self.nstokes, self.header.ngrid
        data = numpy.fromfile(self.file, '<f4', nchan*nstokes*ngrid*ngrid)
        data = data.reshape(nchan, nstokes, ngrid, ngrid)
        
        self.curr_int += 1
        return info, data
        
    @staticmethod
    def sort(filename):
        """
        Sorts the integrations in a DB file to be time-ordered.
        """
        
        # Open the input database.  If it's already sorted, stop.
        inDB = OrvilleImageDB(filename, 'r')
        if inDB.header.flags & OrvilleImageDB.FLAG_SORTED:
            inDB.close()
            return
            
        # Read the entire input database into memory.
        inIntHeaderStruct = inDB._EntryHeader()
        headerSize = ctypes.sizeof(inIntHeaderStruct)
        dataSize = 4*inDB.header.nchan*(0 + inDB.nstokes*inDB.header.ngrid**2)
        int_size = headerSize + dataSize
        
        data = inDB.file.read()
        inDB.file.close()
        if len(data) != int_size * inDB.nint:
            raise RuntimeError('The file "%s" appears to be corrupted.' %
                               filename)
        
        # Loop through the input DB's images, saving their image times.
        # Determine the sort order of those times.
        times = numpy.array([
                struct.unpack_from('<d', data, offset=i)[0] for i in
                xrange(inDB._TIME_OFFSET,
                       int_size * inDB.nint, int_size)])
        
        intOrder = times.argsort()
        
        # Write the sorted file.  Note that we write it using the most recent
        # header version, which may differ from the version of the input file.
        # After writing the updated file header, loop through the intervals
        # and copy (in sorted order) from the input data to the output file.
        inDB.header.flags |= OrvilleImageDB.FLAG_SORTED
        
        with tempfile.NamedTemporaryFile(mode='wb', prefix='orville-', suffix='.oims') as outFile:
            outVersion = inDB.version
            outFileHeaderStruct = inDB.header
            outFile.write(struct.pack('<24s', outVersion))
            outFile.write(outFileHeaderStruct)
            outFile.flush()
            
            outEntryHeaderStruct = inDB._EntryHeader()
            for iOut in xrange(inDB.nint):
                i = intOrder[iOut] * int_size
                entry_header = data[i:i+headerSize]
                entry_data = data[i+headerSize:i+int_size]
                
                ctypes.memmove(ctypes.addressof(outEntryHeaderStruct), entry_header, headerSize)
                outFile.write(outEntryHeaderStruct)
                outFile.write(entry_data)
                outFile.flush()
                
            # Overwrite the original file
            shutil.copy(outFile.name, filename)
            
    # Implement some built-ins to make reading images more "Pythonic" ...
    def __len__(self):
        return self.nint
        
    def __getitem__(self, index):
        if index >= self.nint:
            raise IndexError("image index out of range")
            
        self.seek(index)
        return self.read_image()
        
    def __iter__(self):
        return self
        
    def next(self):
        if self.curr_int >= self.nint:
            raise StopIteration
        else:
            return self.read_image()
