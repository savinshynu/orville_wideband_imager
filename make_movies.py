#!/usr/bin/env python3

"""
Given the name of a directory containing a collection of LWATV images,
convert the images into movies.
"""

import os
import sys
import glob
import time
import shutil
import argparse
from tempfile import mkdtemp


OLD_LWATV2_MOVIES = [[51544, 51545], [57513, 57573], [57576, 57577], [57579, 57580],
                     [57584, 57584], [57589, 57666], [57668, 57698], [57700, 57717],
                     [57719, 57724], [57726, 57732], [57734, 57734], [57736, 57738],
                     [57741, 57743], [57766, 57770], [57772, 57773], [57776, 57792],
                     [57797, 57798], [57801, 57855], [57859, 57888], [57891, 57903],
                     [57905, 58110], [58112, 58121], [58127, 58148], [58150, 58402],
                     [58410, 58789], [58792, 58867], [58869, 58897], [58899, 59005],
                     [59010, 59078], [59082, 59101], [59137, 59187], [59189, 59380],
                     [59382, 59387], [59389, 59407]]


def main(args):
    # Validate and extract the MJD
    if not os.path.isdir(args.directory):
        raise RuntimeError("Expected a path to a collection of LWATV images")
    mjd = os.path.basename(args.directory)
    if mjd == '':
        args.directory = os.path.dirname(args.directory)
        mjd = os.path.basename(args.directory)
        
    # Find all of the PNGs...
    pngs = glob.glob(os.path.join(args.directory, '*.png'))
    pngs.sort()
    if len(pngs) == 0:
        print('No .png files found in "%s", exiting' % args.directory)
        sys.exit(0)

    # ... and then sort out how many image sets there are
    frames = {}
    for png in pngs:
        timestamp = png.split('_', 2)[1]
        try:
            frames[timestamp].append(png)
        except KeyError:
            frames[timestamp] = [png,]
    nsets = max([len(frames[timestamp]) for timestamp in frames])
    print("Found %i image sets amongst %i images" % (nsets, len(pngs)))

    # Make the movies, one for each set found
    timestamps = sorted(list(frames.keys()))
    success = 0
    for i in range(nsets):
        ## Set the movie name - this is in the parent directory of the LWATV image direcotry
        moviename = os.path.join(args.output_dir, "%s_%i.mov" % (mjd, i))
        
        ## Find all of the frames that contribute to that set
        pngs = []
        for timestamp in timestamps:
            try:
                pngs.append( frames[timestamp][i] )
            except IndexError:
                pass
                
        ## Skip over any "short" sets
        if len(pngs) < 0.95*len(timestamps):
            print("Image set %i has only %i image(s) of the expected %i (%.1f%%); skipping" % (i, len(pngs), len(timestamps), 100.0*len(pngs)/len(timestamps)))
            continue
            
        ## Copy the files to a temporary directory and build the movie
        print('Writing movie file "%s" from %i images' % (moviename, len(pngs)))
        t0 = time.time()
        temp_path = mkdtemp(suffix='.movie')
        for j,png in enumerate(pngs):
            shutil.copy(png, os.path.join(temp_path, "%06i.png" % j))
        os.system('ffmpeg -y -v 0 -i %s/%%06d.png -q:v 5 -r 15 -vcodec libx264 -pix_fmt yuv420p %s' % (temp_path, moviename))
        shutil.rmtree(temp_path)
        
        ## Symlink _0.mov to a flat .mov file for easy upload later
        if i == 0:
            os.system('ln -s %s %s' % (moviename, moviename.replace('_0.', '.')))
            
        success += 1
        print('-> Finished in %.1f s' % (time.time()-t0))
        
    # Update the movielist.js file
    movie_mjds = []
    for span in OLD_LWATV2_MOVIES:
        movie_mjds.extend(list(range(span[0], span[1]+1)))
    all_movies = glob.glob(os.path.join(args.output_dir, '*[0-9][0-9][0-9].mov'))
    for moviename in all_movies:
        mjd = os.path.splitext(os.path.basename(moviename))[0]
        mjd = int(mjd, 10)
        if mjd <= OLD_LWATV2_MOVIES[-1][1]:
            continue
        movie_mjds.append(mjd)
    movie_mjds.sort()
    
    # Write the new movielist.js file
    with open(os.path.join(args.output_dir, 'movielist.js'), 'w') as fh:
        fh.write("var movieMJDs = %s;" % (str(movie_mjds)))
        
    # Set an exit code
    if success == 0:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Given a directory containing a collection of LWATV images, build one or more movies"
        )
    parser.add_argument('directory', type=str,
                        help='directory containing LWATV images')
    parser.add_argument('-o', '--output-dir', type=str, default=os.getcwd(),
                        help='output directory for the movies')
    args = parser.parse_args()
    main(args)
    
