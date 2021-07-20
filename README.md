# orville_wideband_imager

[![Paper](https://img.shields.io/badge/lwa%20memo-215-blue)](http://leo.phys.unm.edu/~lwa/memos/memo/lwa0215.pdf)


## Description
The Orville Wideband Imager is a realtime GPU-based all-sky imager for the output
of the ADP broadband correlator that runs at LWA-SV.  Orville receives visibility
data from ADP for 32,896 baselines, images the data, and writes the images to the
disk in a binary frame-based format called "OIMS".  The imaging is performed using
a _w_-stacking algorithm for the non-coplanarity of the array.  For each image, the
sky is projected onto the two dimensional plane using orthographic sine projection.
To reduce the number of _w_-planes needed during _w_-stacking, the phase center is
set to a location approximately 2 degrees off zenith that minimizes the spread in
the _w_ coordinate. The gridding operation is based on the Romein gridder implemented
as part of the [EPIC project](https://github.com/epic-astronomy/EPIC).  Every 5
seconds, the imager produces 4 Stokes (I, Q, U and V) images in 198 channels, each with
100 kHz bandwidth.

## Data Archive
Orville data with reduced spectral resolution (six 3.3 MHz channels) are available at the [LWA data archive](https://lda10g.alliance.unm.edu/Orville/).

## Reading OIMS Files
You can use the `OrvilleImageDB.py` Python module to read the data stored in an OIMS file:
```
import OrvilleImageDB

db = OrvilleImageDB.OrvilleImageDB(oimsFile, 'r')

for i,(hdr,data) in enumerate(db):
    print(i, hdr)

db.close()
```
