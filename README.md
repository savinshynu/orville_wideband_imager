# orville_wideband_imager

[![Paper](https://img.shields.io/badge/lwa%20memo-215-blue)](http://leo.phys.unm.edu/~lwa/memos/memo/lwa0215.pdf)


## Description
The Orville Wideband Imager is a realtime GPU-based all-sky imager for the output
of the Advanced Digitial Processor (ADP) broadband correlator that runs at LWA-SV.
Orville receives visibility data from ADP for 32,896 baselines, images the data,
and writes the images to the disk in a [binary frame-based format called "OIMS"](https://github.com/lwa-project/orville_wideband_imager/blob/master/OrvilleImageDB.py).
The imaging is performed using a _w_-stacking algorithm for the non-coplanarity of
the array.  For each image, the sky is projected onto the two dimensional plane using
orthographic sine projection.  To reduce the number of _w_-planes needed during _w_-stacking,
the phase center is set to a location approximately 2 degrees off zenith that minimizes
the spread in the _w_ coordinate. The gridding operation is based on the Romein gridder
implemented as part of the [EPIC project](https://github.com/epic-astronomy/EPIC).  Every 5
seconds, the imager produces 4 Stokes (I, Q, U and V) images in 198 channels, each with
100 kHz bandwidth.

## Data Archive
Orville data with reduced spectral resolution (six 3.3 MHz channels) are available at the [LWA data archive](https://lda10g.alliance.unm.edu/Orville/).

## Reading OIMS Files
You can use the `OrvilleImageDB.py` Python module to read the data stored in an OIMS file:
```
import OrvilleImageDB

db = OrvilleImageDB.OrvilleImageDB(oimsFile, 'r')

#getting parameters from the input file

ints = db.nint #number of integration
station =  db.header.station #Station info
stokes = db.header.stokes_params #Stokes parameter info
inp_flag = db.header.flags # Flag info
file_start = db.header.start_time # File start time
file_end = db.header.stop_time   # File end time
ngrid = db.header.ngrid #size of images (x-axis)
psize = db.header.pixel_size #angular size of a pixel (at zenith)
nchan = db.header.nchan # number of channels

# Collecting header and data from the whole file

for i,(hdr,data) in enumerate(db):
    print(i, hdr)

# Below will give more info on the headers and data array
# Collecting data and header from a particular image integration (Usually 720 integrations (each 5 seconds) in an hour) 

hdr, dat = db.__getitem__(710) # collecting header and data from the 710 th integration
t = hdr['start_time'] # starting time in MJD
int_len = hdr['int_len'] # Length of each integration
lst = hdr['lst'] # Starting LST time of observation
start_freq = hdr['start_freq']/1e+6 # Starting Frequency in MHz
stop_freq = hdr['stop_freq']/1e+6  # Stopping Frequency in MHz
bandwidth = hdr['bandwidth']/1e+6 # Bandwidth of the observation
cent_ra = hdr['center_ra'] # Phase center RA
cent_dec = hdr['center_dec'] # Phase center Dec
cent_az = hdr['center_az']   # Phase center coordinates, Azimuth, Ideally towards zenith, changed due to W-Projection 
cent_alt = hdr['center_alt'] # Phase center coordinates, Elevation, Ideally towards zenith, changed due to W-Projection 

# dat array contains the image data in a four dimensional array of the form [nchan,stokes,xgrid,ygrid]
# where nchan = 198 frequency channels, stokes = 4 [stokes (I, Q, U,V)], xgrid = grid size in x direction, ygrid = grid size in y direction and mostly xgrid = ygrid = ngrid
# Copy over to numpy arrays for further processing such as image subtraction and transient searches.

db.close()
```
