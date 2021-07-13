#!/usr/bin/env python3

import os
import sys
import copy
import json
import time
import queue
import ephem
import numpy
import ctypes
import shutil
import signal
import logging
from logging.handlers import TimedRotatingFileHandler
import argparse
import threading
import subprocess
from datetime import datetime
from collections import deque

from scipy.special import pro_ang1, iv
from scipy.stats import scoreatpercentile as percentile

from astropy.constants import c as speedOfLight
speedOfLight = speedOfLight.to('m/s').value

from lsl.common.stations import lwasv, parse_ssmif
from lsl.correlator import uvutils
from lsl.imaging import utils
from lsl.common.adp import fS, fC
from lsl.astro import MJD_OFFSET, DJD_OFFSET

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, DiskReader, UDPCapture, UDPSniffer
from bifrost.ring import Ring
from bifrost.libbifrost import bf
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.fft import Fft
from bifrost.quantize import quantize as Quantize
from bifrost.linalg import LinAlg as Correlator
from bifrost.orville import Orville as Gridder
from bifrost.proclog import ProcLog
from bifrost import map as BFMap, asarray as BFAsArray
from bifrost.DataType import DataType as BFDataType
from bifrost.transpose import transpose as BFTranspose
from bifrost.ndarray import memset_array, copy_array
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, stream_synchronize as BFSync, set_devices_no_spin_cpu as BFNoSpinZone
from bifrost import device
BFNoSpinZone()

import PIL.Image, PIL.ImageDraw, PIL.ImageFont

from OrvilleImageDB import OrvilleImageDB


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.join(BASE_PATH, 'calibration')
if not os.path.exists(CAL_PATH):
    os.mkdir(CAL_PATH)


STATION = lwasv
ANTENNAS = STATION.antennas


W_STEP = 0.1


SUPPORT_SIZE = 7
SUPPORT_OVERSAMPLE = 64


def round_up_to_even(n, maxprimes=3):
    """
    Round up the given value to minimize the number of prime factors.  Factors
    other than 2, 3, and 5 are not allowed, and the number of factors of 3 and
    5 cannot exceed maxprimes.
    """
    if n % 2 != 0:
        n += 1
    while True:
        r = n
        nPrimes = 0
        while r > 1 and r % 2 == 0:
            r //= 2
        while r > 1 and r % 3 == 0:
            r //= 3
            nPrimes += 1
        while r > 1 and r % 5 == 0:
            r //= 5
            nPrimes += 1
        if r == 1 and nPrimes <= maxprimes:
            return n
        n += 2


def timetag_to_mjdatetime(time_tag):
    """
    Convert a DP/ADP timestamp into a MJD and UTC hour, minute, and second.
    """
    
    ## Get the date
    unix_time_tag_i = time_tag // int(fS)
    unix_time_tag_f = (time_tag % int(fS)) / float(fS)
    mjd = int(40587 + unix_time_tag_i // 86400)
    unix_day_frac_s = unix_time_tag_i - (unix_time_tag_i // 86400) * 86400
    h = unix_day_frac_s // 3600
    m = unix_day_frac_s % 3600 // 60
    s = unix_day_frac_s % 60 + unix_time_tag_f
    
    return mjd, h, m, s


class MultiQueue(object):
    def __init__(self, slots, maxsize=0):
        self._lock = threading.RLock()
        self._slots = [queue.Queue(maxsize=maxsize) for i in range(slots)]
        
    def empty(self):
        with self._lock:
            is_empty = all([slot.empty() for slot in self._slots])
        return is_empty
        
    def full(self):
        with self._lock:
            is_full = any([slot.full() for slot in self._slots])
        return is_full
        
    def qsize(self):
        with self._lock:
            size = max([slot.qsize() for slot in self._slots])
        return size
        
    def put(self, item, block=True, timeout=None):
        with self._lock:
            for slot in self._slots:
                slot.put(item, block=block, timeout=timeout)
                
    def put_nowait(self, item):
        with self._lock:
            for slot in self._slots:
                slot.put_nowait(item)
                
    def get(self, slot, block=True, timeout=None):
        with self._lock:
            item = self._slots[slot].get(block=block, timeout=timeout)
        return item
        
    def get_nowait(self, slot):
        with self._lock:
            item = self._slots[slot].get_nowait()
        return item
        
    def task_done(self, slot):
        self._slots[slot].task_done()
        
    def join(self):
        for slot in self._slots:
            slot.join()


FILL_QUEUE = queue.Queue(maxsize=4)


def get_good_and_missing_rx():
    pid = os.getpid()
    statsname = os.path.join('/dev/shm/bifrost', str(pid), 'udp_capture', 'stats')
    
    good = 'ngood_bytes    : 0'
    missing = 'nmissing_bytes : 0'
    if os.path.exists(statsname):
        with open(os.path.join('/dev/shm/bifrost', str(pid), 'udp_capture', 'stats'), 'r') as fh:        
            good = fh.readline()
            missing = fh.readline()
    good = int(good.split(':', 1)[1], 10)
    missing = int(missing.split(':', 1)[1], 10)
    return good, missing


class CaptureOp(object):
    def __init__(self, log, oring, sock, *args, **kwargs):
        self.log    = log
        self.oring  = oring
        self.sock   = sock
        self.args   = args
        self.kwargs = kwargs
        self.shutdown_event = threading.Event()
        ## HACK TESTING
        #self.seq_callback = None
    def shutdown(self):
        self.shutdown_event.set()
    def cor_callback(self, seq0, time_tag, chan0, nchan, navg, nsrc, hdr_ptr, hdr_size_ptr):
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
        hdr = {'time_tag': time_tag,
            'seq0':     seq0, 
            'chan0':    chan0,
            'cfreq':    chan0*25e3,
            'nchan':    nchan,
            'bw':       nchan*4*25e3,
            'navg':     navg,
            'nstand':   int(numpy.sqrt(8*nsrc+1)-1)//2,
            'npol':     2,
            'nbl':      nsrc,
            'complex':  True,
            'nbit':     32}
        print("******** CFREQ:", hdr['cfreq'])
        hdr_str = json.dumps(hdr).encode()
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
    def main(self):
        global FILL_QUEUE
        
        seq_callback = PacketCaptureCallback()
        seq_callback.set_cor(self.cor_callback)
        
        with UDPCapture("cor", self.sock, self.oring, *self.args, sequence_callback=seq_callback, **self.kwargs) as capture:
            good, missing = get_good_and_missing_rx()
            while not self.shutdown_event.is_set():
                status = capture.recv()
                print('III', status)
                
                # Determine the fill level of the last gulp
                new_good, new_missing = get_good_and_missing_rx()
                try:
                    fill_level = float(new_good-good) / (new_good-good + new_missing-missing)
                except ZeroDivisionError:
                    fill_level = 0.0
                good, missing = new_good, new_missing
                
                try:
                    FILL_QUEUE.put_nowait(fill_level)
                except queue.Full:
                    pass
                    
        del capture

class SpectraOp(object):
    def __init__(self, log, iring, base_dir=os.getcwd(), core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.output_dir = os.path.join(base_dir, 'spectra')
        self.core = core
        self.gpu = gpu
        
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        
    def _plot_spectra(self, time_tag, freq, specs, status):
        # Plotting setup
        nchan = freq.size
        nstand = specs.shape[0]
        try:
            minval = numpy.min(specs[numpy.where(numpy.isfinite(specs))])
            maxval = numpy.max(specs[numpy.where(numpy.isfinite(specs))])
        except ValueError:
            minval = 0.0
            maxval = 1.0
            
        # Image setup
        width = height = 16
        im = PIL.Image.new('RGB', (width * 65 + 1, height * 65 + 21), '#FFFFFF')
        draw = PIL.ImageDraw.Draw(im)
        font = PIL.ImageFont.load(os.path.join(BASE_PATH, 'fonts', 'helvB10.pil'))
        
        # Axes boxes
        for i in range(width + 1):
            draw.line([i * 65, 0, i * 65, height * 65], fill = '#000000')
        for i in range(height + 1):
            draw.line([(0, i * 65), (im.size[0], i * 65)], fill = '#000000')
            
        # Power as a function of frequency for all antennas
        x = numpy.arange(nchan) * 64 // nchan
        for s in range(nstand):
            if s >= height * width:
                break
            x0, y0 = (s % width) * 65 + 1, (s // width + 1) * 65
            draw.text((x0 + 5, y0 - 60), str(s+1), font=font, fill='#000000')
            
            ## XX
            c = '#1F77B4'
            if status[2*s+0] != 33:
                c = '#799CB4'
            y = ((54.0 / (maxval - minval)) * (specs[s,:,0] - minval)).clip(0, 54)
            draw.line(list(zip(x0 + x, y0 - y)), fill=c)
            
            ## YY
            c = '#FF7F0E'
            if status[2*s+1] != 33:
                c = '#FFC28C'
            y = ((54.0 / (maxval - minval)) * (specs[s,:,1] - minval)).clip(0, 54)
            draw.line(list(zip(x0 + x, y0 - y)), fill=c)
            
        # Summary
        ySummary = height * 65 + 2
        timeStr = datetime.utcfromtimestamp(time_tag / fS)
        timeStr = timeStr.strftime("%Y/%m/%d %H:%M:%S UTC")
        draw.text((5, ySummary), timeStr, font = font, fill = '#000000')
        rangeStr = 'range shown: %.3f to %.3f dB' % (minval, maxval)
        draw.text((210, ySummary), rangeStr, font = font, fill = '#000000')
        x = im.size[0] + 15
        for label, c in reversed(list(zip(('good XX','good YY','flagged XX','flagged YY'),
                                          ('#1F77B4','#FF7F0E','#799CB4',   '#FFC28C')))):
            x -= draw.textsize(label, font = font)[0] + 20
            draw.text((x, ySummary), label, font = font, fill = c)
            
        return im
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        status = [ant.combined_status for ant in ANTENNAS]
        
        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            self.log.info('SpectraOp: Config - %s', ihdr)
            
            # Setup the ring metadata and gulp sizes
            chan0  = ihdr['chan0']
            nchan  = ihdr['nchan']
            nbl    = ihdr['nbl']
            nstand = int(numpy.sqrt(8*nbl+1)-1)//2
            npol   = ihdr['npol']
            navg   = ihdr['navg']
            time_tag0 = iseq.time_tag
            time_tag  = time_tag0
            igulp_size = nstand*(nstand+1)//2*nchan*npol*npol*8
            ishape = (nstand*(nstand+1)//2,nchan,npol,npol)
            self.iring.resize(igulp_size, igulp_size*10)
            
            # Setup the arrays for the frequencies and auto-correlations
            freq = chan0*fC + numpy.arange(nchan)*4*fC
            autos = [i*(2*(nstand-1)+1-i)//2 + i for i in range(nstand)]
            
            intCount = 0
            prev_time = time.time()
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                ## Setup and load
                t0 = time.time()
                idata = ispan.data_view(numpy.complex64).reshape(ishape)
                
                ## Pull out the auto-correlations
                adata = idata[autos,:,:,:].real
                adata = adata[:,:,[0,1],[0,1]]
                
                ## Plot
                im = self._plot_spectra(time_tag, freq, 10*numpy.log10(adata), status)
                
                ## Save
                ### Timetag stuff
                mjd, h, m, s = timetag_to_mjdatetime(time_tag)
                ### The actual save
                outname = os.path.join(self.output_dir, str(mjd))
                if not os.path.exists(outname):
                    os.mkdir(outname)
                filename = '%i_%02i%02i%02i_%.3fMHz_%.3fMHz.png' % (mjd, h, m, s, freq.min()/1e6, freq.max()/1e6)
                outname = os.path.join(outname, filename)
                im.save(outname, 'PNG')
                self.log.debug("Wrote spectra %i to disk as '%s'", intCount, os.path.basename(outname))
                
                time_tag += navg * (fS / 100.0)
                intCount += 1
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                self.log.debug('Spectra plotter processing time was %.3f s', process_time)
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0.0, 
                                          'process_time': process_time,})
        self.log.info("SpectraOp - Done")


class BaselineOp(object):
    def __init__(self, log, iring, base_dir=os.getcwd(), core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.output_dir = os.path.join(base_dir, 'baselines')
        self.core = core
        self.gpu = gpu
        
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        
        self.station = STATION
        
    def _plot_baselines(self, time_tag, freq, dist, baselines, valid):
        # Plotting setup
        nchan = freq.size
        nbl = baselines.shape[0]
        freq = freq[range(nchan//6//2,nchan,nchan//6)]
        baselines = baselines[:,range(nchan//6//2,nchan,nchan//6),:,:]
        baselines = numpy.abs(baselines[:,:,[0,1,1],[0,0,1]])
        minval = numpy.min(baselines[valid,:,:])
        maxval = numpy.max(baselines[valid,:,:])
        if minval == maxval:
            maxval = minval + 1.0
            
        mindst = 0.0
        maxdst = numpy.max(dist)
        
        # Image setup
        width, height = 2, 3
        im = PIL.Image.new('RGB', (width*500 + 1, height*300 + 21), '#FFFFFF')
        draw = PIL.ImageDraw.Draw(im)
        font = PIL.ImageFont.load(os.path.join(BASE_PATH, 'fonts', 'helvB10.pil'))
        
        # Axes boxes
        for i in range(width + 1):
            draw.line([i * 500, 0, i * 500, height * 300], fill = '#000000')
        for i in range(height + 1):
            draw.line([(0, i * 300), (im.size[0], i * 300)], fill = '#000000')
            
        # Visiblity amplitudes as a function of (u,v) distance
        for c in range(baselines.shape[1]):
            if c >= height * width:
                break
            x0, y0 = (c % width) * 500 + 1, (c // width + 1) * 300
            draw.text((x0 + 5, y0 - 295), '%.3f MHz' % (freq[c]/1e6,), font=font, fill='#000000')
            
            ## (u,v) distance as adjusted for the frequency
            x = ((499.0 / (maxdst - mindst)) * (dist[valid]*freq[c]/freq[0] - mindst)).clip(0, 499)
            
            ## XX
            y = ((299.0 / (maxval - minval)) * (baselines[valid,c,0] - minval)).clip(0, 299)
            draw.point(list(zip(x0 + x, y0 - y)), fill='#1F77B4')
            
            ## YY
            y = ((299.0 / (maxval - minval)) * (baselines[valid,c,2] - minval)).clip(0, 299)
            draw.point(list(zip(x0 + x, y0 - y)), fill='#FF7F0E')
            
            ### XY
            #y = ((299.0 / (maxval - minval)) * (baselines[valid,c,1] - minval)).clip(0, 299)
            #draw.point(zip(x0 + x, y0 - y), fill='#A00000')
            
        # Details and labels
        ySummary = height * 300 + 2
        timeStr = datetime.utcfromtimestamp(time_tag / fS)
        timeStr = timeStr.strftime("%Y/%m/%d %H:%M:%S UTC")
        draw.text((5, ySummary), timeStr, font = font, fill = '#000000')
        rangeStr = 'range shown: %.6f - %.6f' % (minval, maxval)
        draw.text((210, ySummary), rangeStr, font = font, fill = '#000000')
        x = im.size[0] + 15
        #for label, c in reversed(list(zip(('XX','XY','YY'), ('#1F77B4','#A00000','#FF7F0E')))):
        for label, c in reversed(list(zip(('XX','YY'), ('#1F77B4','#FF7F0E')))):
            x -= draw.textsize(label, font = font)[0] + 20
            draw.text((x, ySummary), label, font = font, fill = c)
            
        return im
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            self.log.info('BaselineOp: Config - %s', ihdr)
            
            # Setup the ring metadata and gulp sizes
            chan0  = ihdr['chan0']
            nchan  = ihdr['nchan']
            nbl    = ihdr['nbl']
            nstand = int(numpy.sqrt(8*nbl+1)-1)//2
            npol   = ihdr['npol']
            navg   = ihdr['navg']
            time_tag0 = iseq.time_tag
            time_tag  = time_tag0
            igulp_size = nstand*(nstand+1)//2*nchan*npol*npol*8
            ishape = (nstand*(nstand+1)//2,nchan,npol,npol)
            self.iring.resize(igulp_size, igulp_size*10)
            
            # Setup the arrays for the frequencies and baseline lengths
            freq = chan0*fC + numpy.arange(nchan)*4*fC
            t0 = time.time()
            distname = os.path.join(CAL_PATH, 'dist_%i_%i_%i.npy' % (nbl, chan0, nchan))
            try:
                if os.path.exists(distname) and os.path.getmtime(distname) < os.path.getmtime(__file__):
                    raise IOError
                dist = numpy.load(distname)
            except IOError:
                print('dist cache failed')
                uvw = uvutils.compute_uvw(ANTENNAS[0::2], HA=0, dec=self.station.lat*180/numpy.pi,
                                            freq=freq[0], site=self.station.get_observer(), include_auto=True)
                print('uvw.shape', uvw.shape)
                dist = numpy.sqrt(uvw[:,0,0]**2 + uvw[:,1,0]**2)
                numpy.save(distname, dist)
            valid = numpy.where( dist > 0.1 )[0]
            print('@dist', time.time() - t0, '@', dist.shape, dist.size*4/1024.**2)
            
            intCount = 0
            prev_time = time.time()
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                ## Setup and load
                idata = ispan.data_view(numpy.complex64).reshape(ishape)
                
                ## Plot
                im = self._plot_baselines(time_tag, freq, dist, idata, valid)
               
                ## Save
                ### Timetag stuff
                mjd, h, m, s = timetag_to_mjdatetime(time_tag)
                ### The actual save
                outname = os.path.join(self.output_dir, str(mjd))
                if not os.path.exists(outname):
                    os.mkdir(outname)
                filename = '%i_%02i%02i%02i_%.3fMHz_%.3fMHz.png' % (mjd, h, m, s, freq.min()/1e6, freq.max()/1e6)
                outname = os.path.join(outname, filename)
                im.save(outname, 'PNG')
                self.log.debug("Wrote baselines %i to disk as '%s'", intCount, os.path.basename(outname))
                
                time_tag += navg * (fS / 100.0)
                intCount += 1
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                self.log.debug('Baseline plotter processing time was %.3f s', process_time)
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0.0, 
                                          'process_time': process_time,})
        self.log.info("BaselineOp - Done")

class ImagingOp(object):
    def __init__(self, log, iring, oring, decimation=1, core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.oring = oring
        self.decimation = decimation
        self.core = core
        self.gpu = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update({'nring':1, 'ring0':self.oring.name})
        
        self.station = copy.deepcopy(STATION)
        
        self.phase_center_ha = 0.0                  # radians
        self.phase_center_dec = self.station.lat    # radians
        if self.station.id == 'SV':
            # Alternate phase center for Sevilleta that minimized the w RMS
            self.phase_center_ha = 1.0*ephem.hours("-0:07:59.82")
            self.phase_center_dec = 1.0*ephem.degrees("33:21:27.5")
            
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        pce = numpy.sin(self.phase_center_dec)*numpy.sin(self.station.lat) \
              + numpy.cos(self.phase_center_dec)*numpy.cos(self.station.lat)*numpy.cos(self.phase_center_ha)
        pce = numpy.arcsin(pce)
        pca = numpy.sin(self.phase_center_dec) - numpy.sin(pce)*numpy.sin(self.station.lat)
        pca = pca / numpy.cos(pca) / numpy.cos(self.station.lat)
        pca = numpy.arccos(pca)
        if numpy.sin(self.phase_center_ha) > 0:
            pca = 2*numpy.pi - pca
            
        phase_center = numpy.array([numpy.cos(pce)*numpy.sin(pca), 
                                    numpy.cos(pce)*numpy.cos(pca), 
                                    numpy.sin(pce)])
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=True):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                self.log.info('ImagingOp: Config - %s', ihdr)
                
                # Setup the ring metadata and gulp sizes
                chan0  = ihdr['chan0']
                nchan  = ihdr['nchan']
                nbl    = ihdr['nbl']
                nstand = int(numpy.sqrt(8*nbl+1)-1)//2
                npol   = ihdr['npol']
                navg   = ihdr['navg']
                time_tag0 = iseq.time_tag
                time_tag  = time_tag0*1
                
                # Figure out the grid size and resolution - assumes a station size of 
                # 100 m and maximum angular extent for the sky of 130 degrees
                min_lambda = 299792458.0 / ((chan0 + 4*nchan-1)*fC)     # m
                rayleigh_res = 1.22 * min_lambda / 100.0 * 180/numpy.pi # deg
                res = rayleigh_res / 4.0    # deg
                grid_size = int(numpy.ceil(130.0 / res))    # px
                grid_size = max([grid_size, 128])           # px
                grid_size = round_up_to_even(grid_size)     # px
                grid_res = 130.0 / grid_size                # deg/px
                
                # Report
                self.log.info("ImagerOp: grid is %i by %i with a resolution of %.3f deg/px", grid_size, grid_size, grid_res)
                
                ochan = nchan // self.decimation
                igulp_size = nstand*(nstand+1)//2*nchan*npol*npol*8      # complex64
                ishape = (nstand*(nstand+1)//2,nchan,npol,npol)
                ogulp_size = ochan*npol**2*grid_size*grid_size*4              # float32
                oshape = (ochan,npol**2,grid_size,grid_size)
                self.iring.resize(igulp_size, igulp_size*10)
                self.oring.resize(ogulp_size, ogulp_size*10)
                
                ohdr = ihdr.copy()
                ohdr['chan0'] = chan0 + self.decimation//2 + 0.5*(self.decimation%2-1)
                ohdr['nchan'] = ochan
                ohdr['decimation'] = self.decimation
                ohdr['ngrid'] = grid_size
                ohdr['res'] = grid_res
                ohdr['basis'] = 'Stokes'
                ohdr['phase_center_ha'] = self.phase_center_ha
                ohdr['phase_center_dec'] = self.phase_center_dec
                ohdr['phase_center_az'] = pca
                ohdr['phase_center_alt'] = pce
                ohdr_str = json.dumps(ohdr)
                
                # Setup the sorted full visibility array
                try:
                    self.rdata
                    self.sdata
                except AttributeError:
                    self.rdata = BFArray(shape=ishape, dtype=numpy.complex64, space='cuda')
                    self.sdata = BFArray(shape=(nchan,nstand,nstand,npol,npol), dtype=numpy.complex64, space='cuda')
                    
                # Setup the uvw coordinates and get them ready for gridding
                t0 = time.time()
                freq = chan0*fC + numpy.arange(nchan)*4*fC
                dfreq = freq*1.0
                dfreq.shape = (freq.size//self.decimation, self.decimation)
                dfreq = dfreq.mean(axis=1)
                uvwname = os.path.join(CAL_PATH, 'uvw_%i_%i_%i.npy' % (nbl, chan0, nchan))
                try:
                    if os.path.exists(uvwname) and os.path.getmtime(uvwname) < os.path.getmtime(__file__):
                        raise IOError
                    uvw = numpy.load(uvwname)
                except IOError:
                    print('uvw cache failed')
                    uvw = numpy.zeros((3,nchan,nstand,nstand,1,1), dtype=numpy.float32)
                    uvwT = uvutils.compute_uvw(ANTENNAS[0::2], HA=self.phase_center_ha*12/numpy.pi, dec=self.phase_center_dec*180/numpy.pi,
                                              freq=freq, site=self.station.get_observer(), include_auto=True).transpose(1,2,0)
                    uvwT.shape += (1,1)
                    k = 0
                    for i in range(nstand):
                        for j in range(i, nstand):
                            uvw[:,:,i,j,:,:] = -uvwT[:,:,k]
                            uvw[:,:,j,i,:,:] =  uvwT[:,:,k]
                            k += 1
                    uvw[1,:,:] *= -1
                    numpy.save(uvwname, uvw)
                print('@uvw', time.time() - t0, '@', uvw.shape, uvw.size*4/1024.**2)
                
                # Setup the baselines phasing terms for zenith
                t0 = time.time()
                phsname = os.path.join(CAL_PATH, 'phs_%i_%i_%i.npy' % (nbl, chan0, nchan))
                try:
                    #if os.path.exists(phsname) and os.path.getmtime(phsname) < os.path.getmtime(__file__):
                    #    raise IOError
                    phases = numpy.load(phsname)
                except IOError:
                    print('phase cache failed')
                    phases = numpy.zeros((nchan,nstand*(nstand+1)//2,npol,npol), dtype=numpy.complex64)
                    k = 0
                    for i in range(nstand):
                        ## X
                        a = ANTENNAS[2*i + 0]
                        delayX0 = a.cable.delay(freq) - numpy.dot(phase_center, [a.stand.x, a.stand.y, a.stand.z]) / speedOfLight
                        gainX0 = a.cable.gain(freq)
                        cgainX0 = numpy.exp(2j*numpy.pi*freq*delayX0) / numpy.sqrt(gainX0)
                        ## Y
                        a = ANTENNAS[2*i + 1]
                        delayY0 = a.cable.delay(freq) - numpy.dot(phase_center, [a.stand.x, a.stand.y, a.stand.z]) / speedOfLight
                        gainY0 = a.cable.gain(freq)
                        cgainY0 = numpy.exp(2j*numpy.pi*freq*delayY0) / numpy.sqrt(gainY0)
                        ## Goodness check
                        if ANTENNAS[2*i + 0].combined_status != 33 or ANTENNAS[2*i + 1].combined_status != 33:
                            cgainX0 *= 0.0
                            cgainY0 *= 0.0
                            
                        for j in range(i, nstand):
                            ## X
                            a = ANTENNAS[2*j + 0]
                            delayX1 = a.cable.delay(freq) - numpy.dot(phase_center, [a.stand.x, a.stand.y, a.stand.z]) / speedOfLight
                            gainX1 = a.cable.gain(freq)
                            cgainX1 = numpy.exp(2j*numpy.pi*freq*delayX1) / numpy.sqrt(gainX1)
                            ## Y
                            a = ANTENNAS[2*j + 1]
                            delayY1 = a.cable.delay(freq) - numpy.dot(phase_center, [a.stand.x, a.stand.y, a.stand.z]) / speedOfLight
                            gainY1 = a.cable.gain(freq)
                            cgainY1 = numpy.exp(2j*numpy.pi*freq*delayY1) / numpy.sqrt(gainY1)
                            ## Goodness check
                            if ANTENNAS[2*j + 0].combined_status != 33 or ANTENNAS[2*j + 1].combined_status != 33:
                                cgainX1 *= 0.0
                                cgainY1 *= 0.0
                                
                            phases[:,k,0,0] = cgainX0.conj()*cgainX1
                            phases[:,k,0,1] = cgainX0.conj()*cgainY1
                            phases[:,k,1,0] = cgainY0.conj()*cgainX1
                            phases[:,k,1,1] = cgainY0.conj()*cgainY1
                            k += 1
                    numpy.save(phsname, phases)
                print('@phases', time.time() - t0, '@', phases.shape, phases.size*(4+4)/1024.**2)
                
                # Build the gridding kernel
                t0 = time.time()
                kernel = numpy.zeros((SUPPORT_SIZE*SUPPORT_OVERSAMPLE,), dtype=numpy.complex64)
                kx = numpy.arange(-(SUPPORT_SIZE*SUPPORT_OVERSAMPLE)//2+1, (SUPPORT_SIZE*SUPPORT_OVERSAMPLE)//2+1, 
                                  dtype=numpy.float32) / SUPPORT_OVERSAMPLE
                kernel = numpy.sinc(kx) * iv(0, 8.6*numpy.sqrt(1-(2*numpy.abs(kx)/SUPPORT_SIZE)**2)) / iv(0, 8.6)
                
                t0 = time.time()
                weights = numpy.ones((nchan,nstand,nstand,npol,npol), dtype=numpy.complex64)
                for i in range(nstand):
                    # Mask out bad antennas
                    if ANTENNAS[2*i+0].combined_status != 33 or ANTENNAS[2*i+1].combined_status != 33:
                        weights[:,i,:,:,:] = 0.0
                    if ANTENNAS[2*i+0].combined_status != 33 or ANTENNAS[2*i+1].combined_status != 33:
                        weights[:,:,i,:,:] = 0.0
                        
                    for j in range(nstand):
                        if i == j or i == (nstand-1) or j == (nstand-1):
                             weights[:,i,j,:,:] = 0.0
                        
                print('@weights', time.time() - t0, '@', weights.shape, weights.size*(4+4)/1024.**2)
                
                # ... and get them on the GPU
                uvw = uvw.reshape(3,nchan//self.decimation,self.decimation*nstand**2)
                weights = weights.reshape(nchan//self.decimation,self.decimation*nstand**2,npol**2)
                try:
                    copy_array(self.guvws, uvw.astype(numpy.float32))
                    copy_array(self.gwgts, weights)
                    copy_array(self.gkernel, kernel.astype(numpy.complex64))
                    copy_array(self.gphases, phases)
                except AttributeError:
                    self.guvws = BFArray(uvw.astype(numpy.float32), space='cuda')
                    self.gwgts = BFArray(weights, space='cuda')
                    self.gkernel = BFArray(kernel.astype(numpy.complex64), space='cuda')
                    self.gphases = BFArray(phases, space='cuda')
                    
                # Setup the output grid
                self.grid = BFArray(shape=(nchan//self.decimation,npol**2,grid_size,grid_size), dtype=numpy.float32, space='cuda')
                    
                intCount = 0
                prev_time = time.time()
                with oring.begin_sequence(time_tag=iseq.time_tag, header=ohdr_str) as oseq:
                    for ispan in iseq.read(igulp_size):
                        if ispan.size < igulp_size:
                            continue # Ignore final gulp
                        curr_time = time.time()
                        acquire_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        with oseq.reserve(ogulp_size) as ospan:
                            curr_time = time.time()
                            reserve_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            ## Setup and load
                            idata = ispan.data_view(numpy.complex64).reshape(ishape)
                            odata = ospan.data_view(numpy.float32).reshape(oshape)
                            copy_array(self.rdata, idata)
                            
                            ## Phase and fill in the other half of the visibility matrix
                            self.sdata = self.sdata.reshape((nchan,nstand,nstand,npol,npol))
                            BFMap("""
                                if( j > i ) {
                                    auto k = i*(2*(256-1)+1-i)/2 + j;
                                    auto xx = idata(k,c,0,0).conj() * phases(c,k,0,0);
                                    auto yx = idata(k,c,0,1).conj() * phases(c,k,0,1);
                                    auto xy = idata(k,c,1,0).conj() * phases(c,k,1,0);
                                    auto yy = idata(k,c,1,1).conj() * phases(c,k,1,1);
                                    
                                    odata(c,i,j,0,0) = xx + yy;
                                    odata(c,i,j,0,1) = xx - yy;
                                    odata(c,i,j,1,0) = xy + yx;
                                    odata(c,i,j,1,1) = -Complex<float>(0.0,1.0)*(xy - yx);
                                    
                                    xx = xx.conj();
                                    xy = xy.conj();
                                    yx = yx.conj();
                                    yy = yy.conj();
                                    
                                    odata(c,j,i,0,0) = xx + yy;
                                    odata(c,j,i,0,1) = xx - yy;
                                    odata(c,j,i,1,0) = xy + yx;
                                    odata(c,j,i,1,1) = Complex<float>(0.0,1.0)*(xy - yx);   
                                }
                                """, 
                                {'idata':self.rdata, 'phases':self.gphases, 'odata':self.sdata}, 
                                axis_names=('c','i','j'), shape=(nchan,nstand,nstand))
                            self.sdata = self.sdata.reshape(nchan//self.decimation,self.decimation*nstand**2,npol**2)
                            
                            ## Grid
                            try:
                                memset_array(self.grid, 0)
                                BFSync()
                                try:
                                    bfdg.execute(self.sdata, self.grid)
                                except NameError:
                                    bfdg = Gridder()
                                    bfdg.init(self.guvws, self.gwgts, self.gkernel, 
                                            grid_size, grid_res, W_STEP, SUPPORT_OVERSAMPLE, 
                                            polmajor=False)
                                    #bfdg.set_stream(stream)
                                    bfdg.execute(self.sdata, self.grid)
                            except RuntimeError as e:
                                self.log.error("Error during imaging: %s", str(e))
                                
                            ## Save it to the ring
                            BFSync()
                            copy_array(odata, self.grid)
                            
                            time_tag += navg * (int(fS) // 100)
                            intCount += 1
                            
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            self.log.debug('Imager processing time was %.3f s', process_time)
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                        'reserve_time': reserve_time, 
                                                        'process_time': process_time,})
                            
                try:
                    del self.grid
                    del bfdg
                except (AttributeError, NameError):
                    pass
                    
        self.log.info("ImagingOp - Done")


class WriterOp(object):
    def __init__(self, log, iring, base_dir=os.getcwd(), core=-1, gpu=-1):
        self.log = log
        self.iring = iring
        self.output_dir_images = os.path.join(base_dir, 'images')
        self.output_dir_archive = os.path.join(base_dir, 'archive')
        self.output_dir_lwatv = os.path.join(base_dir, 'lwatv')
        self.core = core
        self.gpu = gpu
        
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        if not os.path.exists(self.output_dir_images):
            os.mkdir(self.output_dir_images)
        if not os.path.exists(self.output_dir_archive):
            os.mkdir(self.output_dir_archive)  
        if not os.path.exists(self.output_dir_lwatv):
            os.mkdir(self.output_dir_lwatv)
            
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        
        self.station = copy.deepcopy(STATION)
        
    def _save_image(self, station, time_tag, hdr, freq, data):
        # Get the fill level as a fraction
        global FILL_QUEUE
        try:
            fill = FILL_QUEUE.get_nowait()
            self.log.debug("Fill level is %.1f%%", 100.0*fill)
            FILL_QUEUE.task_done()
        except queue.Empty:
            fill = 0.0
            
        # Get the date
        mjd, h, m, s = timetag_to_mjdatetime(time_tag)
        
        # Figure out the LST
        mjd_f = mjd + (h + m/60.0 + s/3600.0)/24.0
        station.date = mjd_f + (MJD_OFFSET - DJD_OFFSET)
        lst = station.sidereal_time()
        
        # Fill the info dictionary that describes this image
        info = {'start_time':    mjd_f,
                'int_len':       hdr['navg'] / 100.0 / 86400.0,
                'fill':          fill,
                'lst':           lst * 0.5/numpy.pi,
                'start_freq':    freq[0],
                'stop_freq':     freq[-1],
                'bandwidth':     freq[1]-freq[0],
                'center_ra':     (lst - hdr['phase_center_ha']) * 180/numpy.pi,
                'center_dec':    hdr['phase_center_dec'] * 180/numpy.pi,
                'center_az':     hdr['phase_center_az'] * 180/numpy.pi,
                'center_alt':    hdr['phase_center_alt'] * 180/numpy.pi,
                'pixel_size':    hdr['res'],
                'stokes_params': ('I,Q,U,V' if hdr['basis'] == 'Stokes' else 'XX,XY,YX,YY')}
                
        # Write the image to disk
        outname = os.path.join(self.output_dir_images, str(mjd))
        if not os.path.exists(outname):
            os.mkdir(outname)
        filename = '%i_%02i%02i%02i_%.3fMHz_%.3fMHz.oims' % (mjd, h, 0, 0, freq.min()/1e6, freq.max()/1e6)
        outname = os.path.join(outname, filename)
        
        db = OrvilleImageDB(outname, mode='a', station=station.name)
        db.add_image(info, data)
        db.close()
        self.log.debug("Added integration to disk as part of '%s'", os.path.basename(outname))
        
    def _save_archive_image(self, station, time_tag, hdr, freq, data):
        # Get the fill level as a fraction
        global FILL_QUEUE
        try:
            fill = FILL_QUEUE.get_nowait()
            self.log.debug("Fill level is %.1f%%", 100.0*fill)
            FILL_QUEUE.task_done()
        except queue.Empty:
            fill = 0.0
            
        # Get the date
        mjd, h, m, s = timetag_to_mjdatetime(time_tag)
        
        # Figure out the LST
        mjd_f = mjd + (h + m/60.0 + s/3600.0)/24.0
        station.date = mjd_f + (MJD_OFFSET - DJD_OFFSET)
        lst = station.sidereal_time()
        
        # Fill the info dictionary that describes this image
        info = {'start_time':    mjd_f,
                'int_len':       hdr['navg'] / 100.0 / 86400.0,
                'fill':          fill,
                'lst':           lst * 0.5/numpy.pi,
                'start_freq':    freq[0],
                'stop_freq':     freq[-1],
                'bandwidth':     freq[1]-freq[0],
                'center_ra':     (lst - hdr['phase_center_ha']) * 180/numpy.pi,
                'center_dec':    hdr['phase_center_dec'] * 180/numpy.pi,
                'center_az':     hdr['phase_center_az'] * 180/numpy.pi,
                'center_alt':    hdr['phase_center_alt'] * 180/numpy.pi,
                'pixel_size':    hdr['res'],
                'stokes_params': ('I,Q,U,V' if hdr['basis'] == 'Stokes' else 'XX,XY,YX,YY')}
                
        # Write the image to disk
        outname = os.path.join(self.output_dir_archive, str(mjd))
        if not os.path.exists(outname):
            os.mkdir(outname)
        filename = '%i_%02i%02i%02i_%.3fMHz_%.3fMHz.oims' % (mjd, h, 0, 0, freq.min()/1e6, freq.max()/1e6)
        outname = os.path.join(outname, filename)
        
        db = OrvilleImageDB(outname, mode='a', station=station.name)
        db.add_image(info, data)
        db.close()
        self.log.debug("Added archive integration to disk as part of '%s'", os.path.basename(outname))
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        # Setup the figure
        ## Import
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        import matplotlib.patches as mpatches
        ## Create
        fig = plt.Figure(figsize=(5*2, 5*(400.3/390)),
                         facecolor='black')
        ax = [fig.add_axes((i/2.0, 0, 1.0/2.0, 1.005), facecolor='black') for i in range(2)]
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ## Logo-ize
        logo = PIL.Image.open(os.path.join(BASE_PATH, 'logo.png'))
        logo = logo.getchannel('A')
        lax = fig.add_axes([0.01, 0, 0.12, 0.10], frameon=False)
        lax.set_axis_off()
        lax.imshow(logo, origin='upper', cmap='gray')
        
        # Setup the A Team sources, plus the Sun and Jupiter
        srcs = [ephem.Sun(), ephem.Jupiter()]
        srcs.append( ephem.readdb('Cas A,f|J,23:23:24.00,+58:48:54,1') )
        srcs.append( ephem.readdb('Cyg A,f|J,19:59:28.36,+40:44:02,1') )
        srcs.append( ephem.readdb('Tau A,f|J,05:34:31.95,+22:00:52,1') )
        srcs.append( ephem.readdb('Vir A,f|J,12:30:49.42,+12:23:28,1') )
        srcs.append( ephem.readdb(   'GC,f|J,17:45:40.04,-29:00:28,1') )
        
        # Setup the Galactic plane
        gplane = []
        for i in range(360):
            gal = ephem.Galactic(str(i), '0', epoch=ephem.J2000)
            ra, dec = gal.to_radec()
            bdy = ephem.FixedBody()
            bdy._ra = ra
            bdy._dec = dec
            bdy._epoch = ephem.J2000
            gplane.append(bdy)
            
        for iseq in self.iring.read(guarantee=True):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            self.log.info('WriterOp: Config - %s', ihdr)
            
            # Setup the ring metadata and gulp sizes
            chan0      = ihdr['chan0']
            nchan      = ihdr['nchan']
            decimation = ihdr['decimation']
            npol       = ihdr['npol']
            navg       = ihdr['navg']
            ngrid      = ihdr['ngrid']
            res        = ihdr['res']
            time_tag0  = iseq.time_tag
            time_tag   = time_tag0
            igulp_size = nchan*npol*npol*ngrid*ngrid*4        # float32
            ishape = (nchan,npol*npol,ngrid,ngrid)
            self.iring.resize(igulp_size, igulp_size*10)
            
            clip_size = 180.0/numpy.pi/res
            
            # Setup the frequencies
            t0 = time.time()
            freq = chan0*fC + numpy.arange(nchan)*4*fC
            arc_freq = freq*1.0
            arc_freq = arc_freq.reshape(6, -1)
            arc_freq = arc_freq.mean(axis=1)
            
            # Setup the frequencies to write images for
            ichans = [nchan//2,]   ## Only make one image at the center of the band
            
            # Setup the buffer for the automatic color scale control
            vmax = [deque([], maxlen=60) for c in freq]
            
            intCount = 0
            prev_time = time.time()
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                ## Setup and load
                idata = ispan.data_view(numpy.float32).reshape(ishape)
                
                ## Write the full image set to disk
                tSave = time.time()
                self._save_image(self.station, time_tag, ihdr, freq, idata)
                self.log.debug('Save time: %.3f s', time.time()-tSave)
                
                ## Write the archive image set to disk
                tArchive = time.time()
                arc_data = idata.reshape(arc_freq.size,-1,npol*npol,ngrid,ngrid)
                arc_data = arc_data.mean(axis=1)
                self._save_archive_image(self.station, time_tag, ihdr, arc_freq, arc_data)
                self.log.debug('Archive save time: %.3f s', time.time()-tArchive)
                
                ## Timetag stuff
                unix_time_tag_s = time_tag // int(fS)
                date_str = datetime.utcfromtimestamp(unix_time_tag_s).strftime('%Y/%m/%d %H:%M:%S UTC')
                
                ## Compute the locations of the brigth sources and the Galactic plane
                self.station.date = date_str[:-4]
                for src in srcs:
                    src.compute(self.station)
                ateam_x = [numpy.cos(src.alt)*numpy.sin(src.az) for src in srcs if src.alt > 0]
                ateam_y = [numpy.cos(src.alt)*numpy.cos(src.az) for src in srcs if src.alt > 0]
                ateam_s = [src.name                             for src in srcs if src.alt > 0]
                for g in gplane:
                    g.compute(self.station)
                plane_x = [numpy.cos(g.alt)*numpy.sin(g.az) if g.alt > 0 else numpy.nan for g in gplane]
                plane_x = numpy.array(plane_x)
                plane_y = [numpy.cos(g.alt)*numpy.cos(g.az) if g.alt > 0 else numpy.nan for g in gplane]
                plane_y = numpy.array(plane_y)
                
                ## Plot
                for c in ichans:
                    for i,p,l in ((0,0,'I'), (1,3,'V')):
                        ### Pull out the data and get it ready for plotting
                        img = idata[c,p,:,:]
                        if l == 'V':
                            l = '|V|'
                            img = numpy.abs(img)
                            
                        ### Update the colorbar limits
                        if i == 0:
                            vmax[c].append( percentile(img, 98.0) )
                            
                        ### Plot the sky and clip at the horizon
                        ax[i].cla()
                        img = ax[i].imshow(img, origin='lower',
                                           vmin=0, vmax=max([1e-6, numpy.median(vmax[c])]),
                                           interpolation='bilinear', cmap='jet')
                        clip = mpatches.Circle((ngrid/2., ngrid/2.), 1.03*clip_size,
                                               facecolor='none', edgecolor='none')
                        ax[i].add_patch(clip)
                        img.set_clip_path(clip)
                        ax[i].set_xticks([])
                        ax[i].set_yticks([])
                        
                        ### Add in the locations of the A Team sources
                        for x,y,n in zip(ateam_x, ateam_y, ateam_s):
                            ax[i].text(ngrid//2-x*clip_size, ngrid//2+(y +  0.0083)*clip_size, n,
                                       color='white', fontsize=14)
                            
                        ### Add in the Galactic plane
                        ax[i].plot(ngrid//2-plane_x*clip_size, ngrid//2+plane_y*clip_size,
                                   marker='', linestyle='--', color='white', alpha=0.60)
                        
                        
                        ### Add in the polarization labels 
                        if i == 0:
                            ax[i].text(0.01, 0.94, l, verticalalignment='top',
                                       horizontalalignment='left', color='white',
                                       fontsize=14, transform=fig.transFigure)
                        else:
                            ax[i].text(0.99, 0.94, l, verticalalignment='top',
                                       horizontalalignment='right', color='white',
                                       fontsize=14, transform=fig.transFigure)
                            
                    ### Add in the timestamp and frequency information
                    ax[0].text(0.01, 0.99, date_str, verticalalignment='top',
                            horizontalalignment='left', color='white',
                            fontsize=14, transform=fig.transFigure)
                    ax[1].text(0.99, 0.99, '%.3f MHz' % (freq[c]/1e6,), verticalalignment='top',
                            horizontalalignment='right', color='white',
                            fontsize=14, transform=fig.transFigure)
                    
                    ## Save
                    mjd, h, m, s = timetag_to_mjdatetime(time_tag)
                    outname = os.path.join(self.output_dir_lwatv, str(mjd))
                    if not os.path.exists(outname):
                        os.mkdir(outname)
                    filename = '%i_%02i%02i%02i_%.3fMHz.png' % (mjd, h, m, s, freq[c]/1e6)
                    outname = os.path.join(outname, filename)
                    canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
                    canvas.print_figure(outname, dpi=78, facecolor='black')
                    
                    self.log.debug("Wrote LWATV %i, %i to disk as '%s'", intCount, c, os.path.basename(outname))
                    
                time_tag += navg * (int(fS) // 100)
                intCount += 1
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                self.log.debug('Writer processing time was %.3f s', process_time)
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})
        self.log.info("WriterOp - Done")


class UploaderOp(object):
    def __init__(self, log, base_dir=os.getcwd(), core=-1, gpu=-1):
        self.log = log
        self.output_dir_spectra = os.path.join(base_dir, 'spectra')
        self.output_dir_uvdist  = os.path.join(base_dir, 'baselines')
        self.output_dir_lwatv   = os.path.join(base_dir, 'lwatv')
        
        self.core = core
        self.gpu = gpu
        
        for output_dir in (self.output_dir_spectra, self.output_dir_uvdist, self.output_dir_lwatv):
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
                
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.shutdown_event = threading.Event()
        
    def shutdown(self):
        self.shutdown_event.set()
        
    def main(self):
        cpu_affinity.set_core(self.core)
        if self.gpu != -1:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        prev_time = time.time()
        while not self.shutdown_event.is_set():
            curr_time = time.time()
            acquire_time = curr_time - prev_time
            prev_time = curr_time
            
            # Find the latest version of the files
            ## MJD
            try:
                p = subprocess.Popen('ls -td %s/* | head -n1' % self.output_dir_spectra,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     shell=True)
                output, error = p.communicate()
                output = output.decode()
                mjd = os.path.basename(output.split('\n')[0])
            except subprocess.CalledProcessError:
                mjd = '*'
            ## Approximate time as an hour
            utc_hour = int((time.time() % 86400) / 3600)
            
            ## Spectra
            try:
                p = subprocess.Popen('ls -t %s/%s/*.png | head -n1' % (self.output_dir_spectra, mjd),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=True)
                output, error = p.communicate()
                output = output.decode()
                latest_spectra = output.split('\n')[0]
                shutil.copy2(latest_spectra, '/tmp/lwatv_spec.png')
            except (subprocess.CalledProcessError, OSError, IOError):
                pass
                
            ## (u,v) radial distribution
            try:
                p = subprocess.Popen('ls -t %s/%s/*.png | head -n1' % (self.output_dir_uvdist, mjd),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=True)
                output, error = p.communicate()
                output = output.decode()
                latest_uvdist = output.split('\n')[0]
                shutil.copy2(latest_uvdist, '/tmp/lwatv_uvdist.png')
            except (subprocess.CalledProcessError, OSError, IOError):
                pass
                
            ## LWATV image
            try:
                p = subprocess.Popen('ls -t %s/%s/*_%02i*.png | head -n1' % (self.output_dir_lwatv, mjd, utc_hour),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=True)
                output, error = p.communicate()
                output = output.decode()
                latest_lwatv = output.split('\n')[0]
                shutil.copy2(latest_lwatv, '/tmp/lwatv.png')
            except (subprocess.CalledProcessError, OSError, IOError) as e:
                pass
                
            # Upload and make active
            try:
                ## Stage
                p = subprocess.Popen('rsync -e ssh -av /tmp/lwatv*.png \
                                        mcsdr@lwalab.phys.unm.edu:/var/www/lwatv2/incoming/',
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=True)
                output, error = p.communicate()
                if p.returncode != 0:
                    self.log.warning('Error uploading: %s', error)
                    
                ## Activate
                p = subprocess.Popen("ssh mcsdr@lwalab.phys.unm.edu \
                                        'mv -f /var/www/lwatv2/incoming/* /var/www/lwatv2/'",
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    shell=True)
                output, error = p.communicate()
                if p.returncode != 0:
                    self.log.warning('Error making active: %s', error)
            except subprocess.CalledProcessError:
                pass
                
            curr_time = time.time()
            process_time = curr_time - prev_time
            self.log.debug('Uploader processing time was %.3f s', process_time)
            prev_time = curr_time
            self.perf_proclog.update({'acquire_time': acquire_time, 
                                      'reserve_time': -1, 
                                      'process_time': process_time,})
            
            time.sleep(max([10-process_time, 0]))


class LogFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, rollover_callback=None):
        days_per_file =  1
        file_count    = 21
        TimedRotatingFileHandler.__init__(self, filename, when='D',
                                          interval=days_per_file,
                                          backupCount=file_count)
        self.filename = filename
        self.rollover_callback = rollover_callback
    def doRollover(self):
        super(LogFileHandler, self).doRollover()
        if self.rollover_callback is not None:
            self.rollover_callback()

def main(args):
    # Logging setup
    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    if args.logfile is None:
        logHandler = logging.StreamHandler(sys.stdout)
    else:
        logHandler = LogFileHandler(args.logfile)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    log.setLevel(logging.DEBUG)
    
    log.info("Starting %s with PID %i", os.path.basename(__file__), os.getpid())
    log.info("Cmdline args:")
    for arg in vars(args):
        log.info("  %s: %s", arg, getattr(args, arg))
        
    # Setup the cores and GPUs to use
    cores = [0, 1, 2, 3, 4, 5, 6]
    gpus  = [0,]*len(cores)
    log.info("CPUs:         %s", ' '.join([str(v) for v in cores]))
    log.info("GPUs:         %s", ' '.join([str(v) for v in gpus]))
    
    # Setup the signal handling
    ops = []
    shutdown_event = threading.Event()
    def handle_signal_terminate(signum, frame):
        SIGNAL_NAMES = dict((k, v) for v, k in \
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and \
                            not v.startswith('SIG_'))
        log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
        try:
            ops[0].shutdown()
            ops[-1].shutdown()
        except IndexError:
            pass
        shutdown_event.set()
    for sig in [signal.SIGHUP,
                signal.SIGINT,
                signal.SIGQUIT,
                signal.SIGTERM,
                signal.SIGTSTP]:
        signal.signal(sig, handle_signal_terminate)
        
    # Create the rings that we need
    capture_ring = Ring(name="capture", space='system')
    writer_ring = Ring(name="writer", space='system')
    
    # Setup the processing blocks
    ## A reader
    nBL = len(ANTENNAS)//2*(len(ANTENNAS)//2+1)//2
    iaddr = Address(args.address, args.port)
    isock = UDPSocket()
    isock.bind(iaddr)
    isock.timeout = 5.0
    ops.append(CaptureOp(log, capture_ring,
                         isock, nBL*6, 1, 9000, 1, 1, core=cores.pop(0)))
    ## The spectra plotter
    ops.append(SpectraOp(log, capture_ring, base_dir=args.output_dir,
                         core=cores.pop(0), gpu=gpus.pop(0)))
    ## The radial (u,v) plotter
    ops.append(BaselineOp(log, capture_ring, base_dir=args.output_dir,
                          core=cores.pop(0), gpu=gpus.pop(0)))
    ## The imager
    ops.append(ImagingOp(log, capture_ring, writer_ring,
                         decimation=args.decimation,
                         core=cores.pop(0), gpu=gpus.pop(0)))
    ## The image writer and plotter for LWA TV
    ops.append(WriterOp(log, writer_ring, base_dir=args.output_dir,
                         core=cores.pop(0), gpu=gpus.pop(0)))
    ## The image uploader
    ops.append(UploaderOp(log, base_dir=args.output_dir,
                          core=cores.pop(0), gpu=gpus.pop(0)))
    
    # Launch everything and wait
    threads = [threading.Thread(target=op.main) for op in ops]
    
    log.info("Launching %i thread(s)", len(threads))
    for thread in threads:
        thread.daemon = True
        thread.start()
    log.info("Waiting for reader thread to finish")
    while threads[0].is_alive() and not shutdown_event.is_set():
        time.sleep(0.5)
    log.info("Waiting for remaining threads to finish")
    for thread in threads:
        thread.join()
    log.info("All done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Capture data from the ADP wideband correlator mode and image it",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-a', '--address', type=str, default='192.168.40.47',
                        help='IP address to listen to')
    parser.add_argument('-p', '--port', type=int, default=11000,
                        help='UDP port to listen to')
    parser.add_argument('-d', '--decimation', type=int, default=1,
                        help='additional frequecy decimation factor')
    parser.add_argument('-l', '--logfile',    default=None,
                        help='Specify log file')
    parser.add_argument('-o', '--output-dir', type=str, default=os.getcwd(),
                        help='base directory to write output data to')
    args = parser.parse_args()
    main(args)
    
