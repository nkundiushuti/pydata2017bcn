"""
    This file is based on DeepConvSep.

    Copyright (c) 2014-2017 Marius Miron  <miron.marius at gmail.com>

    DeepConvSep is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DeepConvSep is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DeepConvSep.  If not, see <http://www.gnu.org/licenses/>.
 """
 
import numpy as np
import scipy
from scipy import io
import os,sys,re

def infoAudioScipy(filein):
    sampleRate, audioObj = scipy.io.wavfile.read(filein)  
    bitrate = audioObj.dtype    
    nsamples = len(audioObj)
    audioObj = None
    return nsamples, sampleRate, bitrate

def readAudioScipy(filein):
    sampleRate, audioObj = scipy.io.wavfile.read(filein)  
    bitrate = audioObj.dtype    
    try:
        maxv = np.finfo(bitrate).max
    except:
        maxv = np.iinfo(bitrate).max
    return audioObj.astype('float')/maxv, sampleRate, bitrate

def writeAudioScipy(fileout,audio_out,sampleRate,bitrate="int16"):
    maxn = np.iinfo(bitrate).max  
    scipy.io.wavfile.write(filename=fileout, rate=sampleRate, data=(audio_out*maxn).astype(bitrate))

def saveTensor(t, out_path, suffix=''):
    """
    Saves a numpy array as a binary file
    """
    assert os.path.isdir(os.path.dirname(out_path)), "path to save tensor does not exist"
    t.tofile(out_path.replace('.data',suffix+'.data'))
    #save shapes
    save_shape(out_path.replace('.data',suffix+'.shape'),t.shape)

def loadTensor(out_path,suffix=''):
    """
    Loads a binary .data file
    """
    assert os.path.isdir(os.path.dirname(out_path)), "path to load tensor does not exist"
    f_in = np.fromfile(out_path.replace('.data',suffix+'.data'))
    shape = get_shape(out_path.replace('.data',suffix+'.shape'))
    f_in = f_in.reshape(shape)    
    return f_in
    
def save_shape(shape_file,shape):
    """
    Saves the shape of a numpy array
    """
    with open(shape_file, 'w') as fout:
        fout.write(u'#'+'\t'.join(str(e) for e in shape)+'\n')

def get_shape(shape_file):
    """
    Reads a .shape file
    """
    with open(shape_file, 'rb') as f:
        line=f.readline().decode('ascii')
        if line.startswith('#'):
            shape=tuple(map(int, re.findall(r'(\d+)', line)))
            return shape
        else:
            raise IOError('Failed to find shape in file') 
